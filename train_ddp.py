import logging
import os
import random
import sys
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
import numpy as np
from config.args import get_parser

from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from model.utils import get_model
from training.utils import unwrap_model_checkpoint
from training.losses import DiceLoss, FocalLoss, FlowLoss, FlowConsistency
from training.validation import validation_ddp as validation

from training.utils import (
    cosine_lr_scheduler_with_warmup,
    log_evaluation_result,
    get_optimizer
)

from utils import (
    configure_logger,
    save_configure,
    is_master,
    AverageMeter,
    resume_load_optimizer_checkpoint,
    resume_load_model_checkpoint,
    froze_the_feature_extractor_layer,
    froze_the_optical_flow_extractor_layer
)


from dataset.dim2.dataset_acdc import SDTDataset ,DistributedTwoStreamBatchSampler






def train_net(net, trainset, testset, args, ema_net=None, fold_idx=0):
    def worker_init_fn(worker_id):
        random.seed(args.reproduce_seed + worker_id)
    # --------------- CREATE DATA ----------------
    # Dataloader Creation
    # SSl
    # train_sampler = DistributedTwoStreamBatchSampler(trainset, shuffle=True, bs=args.batch_size) if args.distributed else None
    
    # SL
    train_sampler = DistributedSampler(trainset, shuffle=True) if args.distributed else None
    trainLoader = data.DataLoader(
        trainset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        persistent_workers=(args.num_workers > 0),
    )

    test_sampler = DistributedSampler(testset, shuffle=False) if args.distributed else None
    testLoader = data.DataLoader(testset, batch_size=1, shuffle=False, sampler=test_sampler, num_workers=args.num_workers)
    
    logging.info(f"Created Dataset and DataLoader")


    # --------------- CREATE CRITERION ----------------
    # set losses
    criterion_dict = {
        # Supervised Seg loss
        'focal': FocalLoss(args.classes),
        'dice': DiceLoss(),

        # Supervised SDT Est loss
        'l1_sdt': nn.SmoothL1Loss(),
        'skt_l1_sdt': nn.SmoothL1Loss(),
        'edge_l1_sdt': nn.SmoothL1Loss(),

        # Unsupervised Seg loss
        'unsup_focal': FocalLoss(args.classes, one_hot=True, soft_label=True),
        'unsup_dice': DiceLoss(one_hot=True, soft_label=True),
        'skt_l1_pm': nn.SmoothL1Loss(),
        'edge_l1_pm': nn.SmoothL1Loss(),

        # Unsupervised flow loss
        'unsup_flow_loss': FlowLoss(),

        # flow consistency loss
        'flow_consistency_loss': FlowConsistency()
    }

    best_Dice = np.zeros(args.classes)
    best_HD = np.ones(args.classes) * 1000
    best_ASD = np.ones(args.classes) * 1000

    trigger = 0

    scaler = GradScaler()

    # --------------- CREATE SUMMARY WRITER ----------------
    # Initialize tensorboard, optimizer, amp scaler and etc.
    writer = SummaryWriter(f"{args.log_path}{args.unique_name}/fold_{fold_idx}")

    # --------------- START TRAINING ----------------
    # Start training
    iter_num = 0


    # --------------- TRAINING PHASE: FEATURE EXTRACTING ----------------
    # feature extractor phase
    logging.info(f"Feature Extracting Phase")

    optimizer = get_optimizer(args, net)

    # training
    for epoch in range(args.start_epoch, args.epochs):
        logging.info(f"Feature Extracting Phase: Starting epoch {epoch + 1}/{args.epochs}")
        logging.info(f"Current lr: {optimizer.state_dict()['param_groups'][0]['lr']:.4e}")

        cosine_lr_scheduler_with_warmup(optimizer, args.base_lr, epoch, args.warm_epoch, args.epochs,
                                        lr_min=args.lr_min, lr_warm=args.lr_warm)

        train_sampler.set_epoch(epoch)

        current_iter_num = train_epoch(args, trainLoader, net, optimizer, scaler, epoch, writer, criterion_dict, iter_num)

        iter_num = current_iter_num
        ##################################################################################
        # Evaluation, save checkpoint and log training info
        net_for_eval = net
        
        if is_master(args):
            # save the latest checkpoint, including net, ema_net, and optimizer
            net_state_dict, ema_net_state_dict = unwrap_model_checkpoint(net, ema_net, args)

            torch.save({
                'epoch': epoch+1,
                'model_state_dict': net_state_dict,
                'ema_model_state_dict': ema_net_state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
            }, f"{args.cp_path}{args.dataset}/{args.unique_name}/fold_{fold_idx}_latest.pth")

        if (epoch+1) % args.val_freq == 0:
            if epoch+1 > args.warm_epoch:
                trigger += 1

            dice_list_test, ASD_list_test, HD_list_test = validation(net_for_eval, testLoader, args)
            if is_master(args):

                log_evaluation_result(writer, dice_list_test, ASD_list_test, HD_list_test, 'test', epoch, args)
            
                if dice_list_test.mean() >= best_Dice.mean():

                    best_Dice = dice_list_test
                    best_HD = HD_list_test
                    best_ASD = ASD_list_test

                    # Save the checkpoint with best performance
                    net_state_dict, ema_net_state_dict = unwrap_model_checkpoint(net, ema_net, args)

                    torch.save({
                        'epoch': epoch+1,
                        'model_state_dict': net_state_dict,
                        'ema_model_state_dict': ema_net_state_dict,
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, f"{args.cp_path}{args.dataset}/{args.unique_name}/fold_{fold_idx}_best.pth")

                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': net,
                        'ema_model_state_dict': ema_net_state_dict,
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, f"{args.cp_path}{args.dataset}/{args.unique_name}/fold_{fold_idx}_best_full_model.pth")

                logging.info("Evaluation Done")
                logging.info(f"Dice: {dice_list_test.mean():.4f}/Best Dice: {best_Dice.mean():.4f}")

            # early stopping
            if dice_list_test.mean() >= best_Dice.mean():
                trigger = 0
            if args.es >= 0 and trigger >= args.es:
                print("=> early stopping")
                break

            writer.add_scalar('LR', optimizer.state_dict()['param_groups'][0]['lr'], epoch + 1)

    return best_Dice, best_HD, best_ASD


def train_epoch(args, trainLoader, net, optimizer, scaler, epoch, writer, criterion_dict, iter_num):
    current_iter_num = iter_num

    epoch_loss, epoch_sup_seg_loss, epoch_sup_reg_loss, epoch_unsup_seg_loss, epoch_unsup_flow_loss, epoch_flow_consistency_loss = \
        AverageMeter("Loss", ":.2f"), \
        AverageMeter("SupSegLoss", ":.2f"), \
        AverageMeter("SupRegLoss", ":.2f"), \
        AverageMeter("UnsupSegLoss", ":.2f"), \
        AverageMeter("UnsupFlowLoss", ":.2f"), \
        AverageMeter("FlowConsistencyLoss", ":.2f")

    net.train()

    # for i, inputs in enumerate(trainLoader):
    for i, sampled_batch in enumerate(trainLoader):
        loss = torch.tensor(.0).cuda(args.proc_idx)

        # input data
        img = sampled_batch['img'].cuda(args.proc_idx)

        # label
        lab = sampled_batch['lab'].cuda(args.proc_idx)  # bs, frames, c, h, w
        lab = lab.contiguous().view(lab.shape[0] * lab.shape[1], *lab.shape[2:])

        sdt_lab = sampled_batch['sdt_lab'].cuda(args.proc_idx)  # bs, frames, c-1, h, w
        sdt_lab = sdt_lab.contiguous().view(sdt_lab.shape[0] * sdt_lab.shape[1], *sdt_lab.shape[2:])

        bs, frames = img.shape[0], img.shape[1]
        
        # get the labeled data idx
        labeled_idx = int(bs * args.labeled_rate) * frames

        optimizer.zero_grad()

        # network output
        res_dict = net(img)

        # full mask
        out_full = res_dict['out_full'][-1]  # bs, frames, c, h, w
        out_full = out_full.contiguous().view(out_full.shape[0] * out_full.shape[1], *out_full.shape[2:])

        # sdt out
        out_sdt = res_dict['out_sdt'][-1]  # bs, frames, c-1, h, w
        out_sdt = out_sdt.contiguous().view(out_sdt.shape[0] * out_sdt.shape[1], *out_sdt.shape[2:])

        # pseudo mask
        out_pm = res_dict['pseudo_mask'][-1]  # bs, frames, c, h, w
        out_pm = out_pm.contiguous().view(out_pm.shape[0] * out_pm.shape[1], *out_pm.shape[2:])


        # Supervised Seg loss
        dice_loss = criterion_dict['dice'](out_full[:labeled_idx, ...], lab[:labeled_idx, ...].long())
        # focal_loss = criterion_dict['focal'](out_full[:labeled_idx, ...], lab[:labeled_idx, ...].long())
        sup_seg_loss = dice_loss

        # Supervised Reg loss
        # global l1 loss
        sup_sdt_out = F.tanh(out_sdt[:labeled_idx, ...])
        sup_sdt_label = sdt_lab[:labeled_idx, ...]
        
        l1_loss = criterion_dict['l1_sdt'](sup_sdt_out, sup_sdt_label) * 20

        sup_reg_loss = l1_loss

        # Sup BK
        for n in range(sup_sdt_label.shape[0]):
            for c in range(sup_sdt_label.shape[1]):
                if lab[n, ...].max() >= c + 1:

                    # give the skt position a focus loss
                    skt_pos = torch.where(sup_sdt_label[n, c] == 1.0)
                    if len(skt_pos[0]) == 0:
                        continue
                    # give the edge position a focus loss
                    edge_pos = torch.where(sup_sdt_label[n, c] == .0)
                    if len(edge_pos[0]) == 0:
                        continue

                    if args.sup_K == True:
                        sup_reg_loss += criterion_dict['skt_l1_sdt'](sup_sdt_out[n, c][skt_pos], 
                                                                     torch.ones_like(sup_sdt_out[n, c])[skt_pos]) * 1 / (
                                                    sup_sdt_label.shape[0] * sup_sdt_label.shape[1])
                    if args.sup_B == True:
                        sup_reg_loss += criterion_dict['edge_l1_sdt'](sup_sdt_out[n, c][edge_pos],
                                                                      torch.zeros_like(sup_sdt_out[n, c])[edge_pos]) * 1 / (
                                                    sup_sdt_label.shape[0] * sup_sdt_label.shape[1])

                else:
                    continue


        # supervised loss
        supervised_loss = sup_seg_loss + sup_reg_loss

        # Unsupervised loss
        # Unsupervised R-S loss
        # unsupervised consistency
        consistency_weight_1 = args.consistency_weight_1
        # calculate the consistency loss between pseudo mask and full mask
        unsup_dice_loss = criterion_dict['unsup_dice'](F.tanh(out_pm[:, ...]), out_full[:, ...])
        # unsup_focal_loss = criterion_dict['unsup_focal'](F.tanh(out_pm[:, ...]), out_full[:, ...])

        unsup_seg_loss = unsup_dice_loss
        
        _, soft_pred_out = torch.max(F.softmax(out_full, dim=1), dim=1)

        # Unsup BK
        for n in range(out_pm.shape[0]):
            for c in range(1, out_pm.shape[1]):
                if soft_pred_out[n, ...].max() >= c:

                    # give the skt position a focus loss
                    skt_pos_pm = torch.where(out_pm[n, c] >= args.kk_thresh)
                    if len(skt_pos_pm[0]) == 0:
                        continue
                    # give the edge position a focus loss
                    edge_pos_pm = torch.where(out_pm[n, c] == 0)
                    if len(edge_pos_pm[0]) == 0:
                        continue

                    if args.unsup_K == True:
                        unsup_seg_loss += criterion_dict['skt_l1_pm'](F.softmax(out_full, dim=1)[n, c][skt_pos_pm],
                                                                      torch.ones_like(out_full)[n, c][skt_pos_pm]) * 1 / (
                                                    out_pm.shape[0] * out_pm.shape[1])
                    if args.unsup_B == True:
                        unsup_seg_loss += criterion_dict['edge_l1_pm'](F.softmax(out_full, dim=1)[n, c][edge_pos_pm],
                                                                   torch.zeros_like(out_full)[n, c][skt_pos_pm]) * 1 / (
                                                    out_pm.shape[0] * out_pm.shape[1])

                else:
                    continue

        unsup_seg_loss = consistency_weight_1 * unsup_seg_loss

        # Unsupervised Flow loss
        # unsupervised flow consistency
        consistency_weight_2 = args.consistency_weight_2

        # calculate the flow loss between sdt_t_1 and sdt_t
        sdt_t_1, sdt_t = F.tanh(res_dict['out_sdt'][-1][:, 0, ...]), F.tanh(res_dict['out_sdt'][-1][:, -1, ...])  # bs, c-1, h, w
        fp_t_1 = res_dict['feature_pyramid_1']
        fp_t = res_dict['feature_pyramid_2']
        flows = res_dict['optical_flow']  # list[bs, 2, h, w]

        sdt_channel = len(fp_t_1)

        unsup_flow_loss = torch.tensor(.0).cuda(args.proc_idx)
        flow_consistency_loss = torch.tensor(.0).cuda(args.proc_idx)

        for k in range(sdt_channel):
            # Unsupervised flow loss
            unsup_flow_loss += criterion_dict['unsup_flow_loss'](sdt_t_1[:, k, ...].unsqueeze(1),
                                                                 sdt_t[:, k, ...].unsqueeze(1), fp_t_1[k], fp_t[k], flows[k])
            # Flow consistency loss
            flow_consistency_loss += criterion_dict['flow_consistency_loss'](sdt_t_1[:, k, ...].unsqueeze(1),
                                                                             sdt_t[:, k, ...].unsqueeze(1), flows[k])
        if args.unsup_Flow == True:
            # unsupervised flow consistency
            consistency_weight_2 = 0.01
        else:
            # unsupervised flow consistency
            consistency_weight_2 = .0

        unsup_flow_loss = consistency_weight_2 * (unsup_flow_loss / sdt_channel * 50)
        flow_consistency_loss = consistency_weight_2 * (flow_consistency_loss / sdt_channel * 50)



        if epoch <= 40:
            loss += supervised_loss + .0 * unsup_seg_loss + .0 * unsup_flow_loss + .0 * flow_consistency_loss
        elif epoch > 40 and epoch <= 80:
            loss += supervised_loss + unsup_seg_loss + .0 * unsup_flow_loss + .0 * flow_consistency_loss
        elif epoch > 80:
            loss += supervised_loss + unsup_seg_loss + unsup_flow_loss + flow_consistency_loss
        else:
            loss += supervised_loss

        # loss backward
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if i % args.print_freq == 0:
            print('Epoch: {} [{}/{}], '
                  'Iter: [{}/{}], '
                  'Loss: {:.3f}, '
                  'SupSegLoss: {:.3f}, '
                  'SupRegLoss: {:.3f}, '
                  'UnsupSegLoss: {:.3f}, '
                  'UnsupFlowLoss: {:.3f}, '
                  'FlowConsistencyLoss: {:.3f}, '
                .format(epoch + 1, i, len(trainLoader),
                current_iter_num, args.epochs * len(trainLoader),
                loss.item(),
                sup_seg_loss.item(),
                sup_reg_loss.item(),
                unsup_seg_loss.item(),
                unsup_flow_loss.item(),
                flow_consistency_loss.item()))


        epoch_loss.update(loss.item(), img.shape[0])
        epoch_sup_seg_loss.update(sup_seg_loss.item(), img.shape[0])
        epoch_sup_reg_loss.update(sup_reg_loss.item(), img.shape[0])
        epoch_unsup_seg_loss.update(unsup_seg_loss.item(), img.shape[0])
        epoch_unsup_flow_loss.update(unsup_flow_loss.item(), img.shape[0])
        epoch_flow_consistency_loss.update(flow_consistency_loss.item(), img.shape[0])

        current_iter_num = current_iter_num + 1

        torch.cuda.empty_cache()


        if is_master(args):
            writer.add_scalar('Train/Loss', epoch_loss.avg, epoch + 1)
            writer.add_scalar('Train/SupSegLoss', epoch_sup_seg_loss.avg, epoch + 1)
            writer.add_scalar('Train/SupRegLoss', epoch_sup_reg_loss.avg, epoch + 1)
            writer.add_scalar('Train/UnsupSegLoss', epoch_unsup_seg_loss.avg, epoch + 1)
            writer.add_scalar('Train/UnsupFlowLoss', epoch_unsup_flow_loss.avg, epoch + 1)
            writer.add_scalar('Train/FlowConsistencyLoss', epoch_flow_consistency_loss.avg, epoch + 1)


    return current_iter_num

def init_network(args):
    net = get_model(args)
    ema_net = None

    if args.resume:
        resume_load_model_checkpoint(net, ema_net, args)

    if args.torch_compile:
        net = torch.compile(net)

    return net, ema_net 


def main_worker(proc_idx, ngpus_per_node, fold_idx, args, result_dict=None, trainset=None, testset=None):
    # seed each process
    if args.reproduce_seed is not None:
        random.seed(args.reproduce_seed)
        np.random.seed(args.reproduce_seed)
        torch.manual_seed(args.reproduce_seed)

        if hasattr(torch, "set_deterministic"):
            torch.set_deterministic(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # set process specific info
    args.proc_idx = proc_idx
    args.ngpus_per_node = ngpus_per_node

    # suppress printing if not master
    if args.multiprocessing_distributed and args.proc_idx != 0:
        def print_pass(*args, **kwargs):
            pass

        #builtins.print = print_pass
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + proc_idx
        
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=f"{args.dist_url}",
            world_size=args.world_size,
            rank=args.rank,
        )
        torch.cuda.set_device(args.proc_idx)

        # adjust data settings according to multi-processing
        args.batch_size = int(args.batch_size / args.ngpus_per_node)
        args.workers = int((args.num_workers + args.ngpus_per_node - 1) / args.ngpus_per_node)


    args.cp_dir = f"{args.cp_path}/{args.dataset}/{args.unique_name}"
    os.makedirs(args.cp_dir, exist_ok=True)
    configure_logger(args.rank, args.cp_dir+f"/fold_{fold_idx}.txt")
    save_configure(args)

    logging.info(
        f"\nDataset: {args.dataset},\n"
        + f"Model: {args.model},\n"
        + f"Dimension: {args.dimension}"
    )

    net, ema_net = init_network(args)
    
    net.to('cuda')

    if args.distributed:
        net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
        net = DistributedDataParallel(net, device_ids=[args.proc_idx], find_unused_parameters=True)

    logging.info(f"Created Model")

    best_Dice, best_HD, best_ASD = train_net(net, trainset, testset, args, ema_net, fold_idx=fold_idx)
    
    logging.info(f"Training and evaluation on Fold {fold_idx} is done")
    
    if args.distributed:
        if is_master(args):
            # collect results from the master process
            result_dict['best_Dice'] = best_Dice
            result_dict['best_HD'] = best_HD
            result_dict['best_ASD'] = best_ASD
    else:
        return best_Dice, best_HD, best_ASD
        

        



if __name__ == '__main__':
    # parse the arguments
    args = get_parser()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    torch.multiprocessing.set_start_method('spawn')
    args.log_path = args.log_path + '%s/' % args.dataset

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()

    print("-------------------------------------------")
    print("exp setting")
    print("Sup_K: {}".format(args.sup_K))
    print("Sup_B: {}".format(args.sup_B))
    print("Unsup_K: {}".format(args.unsup_K))
    print("Unsup_B: {}".format(args.unsup_B))
    print("Unsup_Flow: {}".format(args.unsup_Flow))
    print("-------------------------------------------")



    Dice_list, HD_list, ASD_list = [], [], []
    
    for fold_idx in range(args.k_fold):
        if args.multiprocessing_distributed:
            with mp.Manager() as manager:
            # use the Manager to gather results from the processes
                result_dict = manager.dict()
                    
                # Since we have ngpus_per_node processes per node, the total world_size
                # needs to be adjusted accordingly
                args.world_size = ngpus_per_node * args.world_size

                trainset = SDTDataset(args, mode='train', fold_idx=fold_idx, debug_ratio=args.debug_ratio)
                testset = SDTDataset(args, mode='test', fold_idx=fold_idx)

                # Use torch.multiprocessing.spawn to launch distributed processes:
                # the main_worker process function
                mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, fold_idx, args, result_dict, trainset, testset))
                best_Dice = result_dict['best_Dice']
                best_HD = result_dict['best_HD']
                best_ASD = result_dict['best_ASD']
            args.world_size = 1
        else:
            trainset = SDTDataset(args, mode='train', fold_idx=fold_idx, debug_ratio=args.debug_ratio)
            testset = SDTDataset(args, mode='test', fold_idx=fold_idx)

            # Simply call main_worker function
            best_Dice, best_HD, best_ASD = main_worker(0, ngpus_per_node, fold_idx, args, trainset=trainset, testset=testset)


        Dice_list.append(best_Dice)
        HD_list.append(best_HD)
        ASD_list.append(best_ASD)

        # break

    #############################################################################################
    # Save the cross validation results
    total_Dice = np.vstack(Dice_list)
    total_HD = np.vstack(HD_list)
    total_ASD = np.vstack(ASD_list)

    with open(f"{args.cp_path}/{args.dataset}/{args.unique_name}/cross_validation.txt",  'w') as f:
        np.set_printoptions(precision=4, suppress=True) 
        f.write('Dice\n')
        for i in range(args.k_fold):
            f.write(f"Fold {i}: {Dice_list[i]}\n")
        f.write(f"Each Class Dice Avg: {np.mean(total_Dice, axis=0)}\n")
        f.write(f"Each Class Dice Std: {np.std(total_Dice, axis=0)}\n")
        f.write(f"All classes Dice Avg: {total_Dice.mean()}\n")
        f.write(f"All classes Dice Std: {np.mean(total_Dice, axis=1).std()}\n")

        f.write("\n")

        f.write("HD\n")
        for i in range(args.k_fold):
            f.write(f"Fold {i}: {HD_list[i]}\n")
        f.write(f"Each Class HD Avg: {np.mean(total_HD, axis=0)}\n")
        f.write(f"Each Class HD Std: {np.std(total_HD, axis=0)}\n")
        f.write(f"All classes HD Avg: {total_HD.mean()}\n")
        f.write(f"All classes HD Std: {np.mean(total_HD, axis=1).std()}\n")

        f.write("\n")

        f.write("ASD\n")
        for i in range(args.k_fold):
            f.write(f"Fold {i}: {ASD_list[i]}\n")
        f.write(f"Each Class ASD Avg: {np.mean(total_ASD, axis=0)}\n")
        f.write(f"Each Class ASD Std: {np.std(total_ASD, axis=0)}\n")
        f.write(f"All classes ASD Avg: {total_ASD.mean()}\n")
        f.write(f"All classes ASD Std: {np.mean(total_ASD, axis=1).std()}\n")

    print(f'All {args.k_fold} folds done.')

    sys.exit(0)

