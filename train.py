import builtins
import logging
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np
from model.utils import get_model
from training.dataset.utils import get_dataset
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter

from training.utils import update_ema_variables
from training.losses import DiceLoss, SurfaceLoss, dynamic_weight_average
from training.validation import validation
from training.utils import (
    cosine_lr_scheduler_with_warmup,
    log_evaluation_result, 
    get_optimizer, 
    filter_validation_results
)
import yaml
from config.args import get_parser
import time
import math
import sys
import pdb
import warnings
from training.dataset.dim2.dataset_acdc import CMRDataset

import matplotlib.pyplot as plt

from utils import (
    configure_logger,
    save_configure,
    AverageMeter,
    ProgressMeter,
    resume_load_optimizer_checkpoint,
    resume_load_model_checkpoint,
)

import types
import collections
from random import shuffle

warnings.filterwarnings("ignore", category=UserWarning)



def train_net(net, args, ema_net=None, fold_idx=0):

    ################################################################################

    # Split Data
    img_name_list = os.listdir(os.path.join(args.data_root, 'imagesTr'))
    img_name_list = ["_".join(p.split('_')[:3]) for p in img_name_list]
    random.Random(args.split_seed).shuffle(img_name_list)

    img_name_list = img_name_list[:2000]

    length = len(img_name_list)
    test_name_list = img_name_list[fold_idx * (length // args.k_fold):(fold_idx + 1) * (length // args.k_fold)]
    train_name_list = list(set(img_name_list) - set(test_name_list))

    print('{} for training, {} for a {}-fold validation'.format(len(train_name_list), len(test_name_list), args.k_fold))

    # Dataset Creation
    trainset = CMRDataset(args, img_name_list=train_name_list, mode='train', load_edge=args.use_edge)
    trainLoader = data.DataLoader(
        trainset, 
        batch_size=args.batch_size,
        shuffle=True, 
        pin_memory=(args.aug_device != 'gpu'), 
        num_workers=args.num_workers, 
        persistent_workers=(args.num_workers>0)
    )

    testset = CMRDataset(args, img_name_list=test_name_list, mode='test', load_edge=False)
    testLoader = data.DataLoader(testset, batch_size=1, pin_memory=True, shuffle=False, num_workers=args.num_workers)
    
    logging.info(f"Created Dataset and DataLoader")

    ################################################################################
    # Initialize tensorboard, optimizer and etc
    writer = SummaryWriter(f"{args.log_path}{args.unique_name}/fold_{fold_idx}")

    optimizer = get_optimizer(args, net)

    if args.resume:
        resume_load_optimizer_checkpoint(optimizer, args)

    criterion = nn.CrossEntropyLoss(weight=torch.tensor(args.weight).cuda(args.proc_idx).float())
    criterion_dl = DiceLoss()
    criterion_bdl = DiceLoss(one_hot=True)
    criterion_b = SurfaceLoss(weight=torch.tensor(args.weight).cuda(args.proc_idx).float())

    ################################################################################
    # Start training
    best_Dice = np.zeros(args.classes)
    best_HD = np.ones(args.classes) * 1000
    best_ASD = np.ones(args.classes) * 1000

    trigger = 0
    LOSS_T_1 = []
    LOSS_T_2 = []

    for epoch in range(args.start_epoch, args.epochs):
        logging.info(f"Starting epoch {epoch+1}/{args.epochs}")
        logging.info(f"Current lr: {optimizer.state_dict()['param_groups'][0]['lr']:.4e}")

        cosine_lr_scheduler_with_warmup(optimizer, args.base_lr, epoch, 10, args.epochs)

        LOSS_T_1, LOSS_T_2 = train_epoch(trainLoader, net, ema_net, optimizer, epoch, writer, criterion, criterion_dl, criterion_b, criterion_bdl, LOSS_T_1, LOSS_T_2, args)

        ########################################################################################
        # Evaluation, save checkpoint and log training info
        net_for_eval = ema_net if args.ema else net 
        
        # save the latest checkpoint, including net, ema_net, and optimizer
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': net.state_dict() if not args.torch_compile else net._orig_mod.state_dict(),
            'ema_model_state_dict': ema_net.state_dict() if args.ema else None,
            'optimizer_state_dict': optimizer.state_dict(),
        }, f"{args.cp_path}{args.dataset}/{args.unique_name}/fold_{fold_idx}_latest.pth")
   
        if (epoch+1) % args.val_freq == 0:
            trigger += 1

            dice_list_test, ASD_list_test, HD_list_test = validation(net, testLoader, args)
            # dice_list_test, ASD_list_test, HD_list_test = filter_validation_results(dice_list_test, ASD_list_test, HD_list_test, args) # filter results for some dataset, e.g. amos_mr
            log_evaluation_result(writer, dice_list_test, ASD_list_test, HD_list_test, 'test', epoch, args)
            
            if dice_list_test.mean() >= best_Dice.mean():
                best_Dice = dice_list_test
                best_HD = HD_list_test
                best_ASD = ASD_list_test

                # Save the checkpoint with best performance
                torch.save({
                    'epoch': epoch+1,
                    'model_state_dict': net.state_dict() if not args.torch_compile else net._orig_mod.state_dict(),
                    'ema_model_state_dict': ema_net.state_dict() if args.ema else None,
                    'optimizer_state_dict': optimizer.state_dict(),
                }, f"{args.cp_path}{args.dataset}/{args.unique_name}/fold_{fold_idx}_best.pth")

                trigger = 0
            
            logging.info("Evaluation Done")
            logging.info(f"Dice: {dice_list_test.mean():.4f}/Best Dice: {best_Dice.mean():.4f}")
    
        writer.add_scalar('LR', optimizer.state_dict()['param_groups'][0]['lr'], epoch+1)

        # early stopping
        if args.es >= 0 and trigger >= args.es:
            print("=> early stopping")
            break

    return best_Dice, best_HD, best_ASD


def train_epoch(trainLoader, net, ema_net, optimizer, epoch, writer, criterion, criterion_dl, criterion_b, criterion_bdl, LOSS_T_1, LOSS_T_2, args):

    batch_time = AverageMeter("Time", ":6.2f")
    epoch_loss = AverageMeter("Loss", ":.2f")
    b_epoch_loss = AverageMeter("BLoss", ":.2f")
    progress = ProgressMeter(
        len(trainLoader) if args.dimension=='2d' else args.iter_per_epoch, 
        [batch_time, epoch_loss], 
        prefix="Epoch: [{}]".format(epoch+1),
    )

    net.train()

    tic = time.time()
    iter_num_per_epoch = 0

    for i, inputs in enumerate(trainLoader):
        loss = 0
        full_bce_loss = 0
        full_dice_loss = 0
        boundary_dice_loss = 0
        boundary_loss = 0

        if args.use_edge:
            img, label, edge_label = inputs[0], inputs[1].long(), inputs[-1].long()
            if args.aug_device != 'gpu':
                img = img.cuda(non_blocking=True)
                label = label.cuda(non_blocking=True)
                edge_label = edge_label.cuda(non_blocking=True)

            # step = i + epoch * len(trainLoader)  # global steps
            optimizer.zero_grad()
            edge_de_out, full_de_out = net(img)

            full_bce_loss = criterion(full_de_out, label.squeeze(1))
            full_dice_loss = criterion_dl(full_de_out, label)
            boundary_dice_loss = criterion_bdl(edge_de_out, edge_label)
            boundary_loss = criterion_b(edge_de_out, edge_label)

            dwa_weights = dynamic_weight_average(LOSS_T_1, LOSS_T_2, args)
            loss = dwa_weights[0] * full_bce_loss \
                   + dwa_weights[1] * full_dice_loss \
                   + dwa_weights[2] * boundary_dice_loss \
                   + dwa_weights[-1] * boundary_loss

            LOSS_T_2 = LOSS_T_1
            LOSS_T_1 = [full_bce_loss.item(), full_dice_loss.item(), boundary_dice_loss.item(), boundary_loss.item()]
        else:
            img, label = inputs[0].cuda(args.proc_idx), inputs[1].cuda(args.proc_idx)

            step = i + epoch * len(trainLoader)  #  global steps
            optimizer.zero_grad()
            result = net(img)


            full_bce_loss = criterion(result, label.squeeze(1).long())
            full_dice_loss = criterion_dl(result, label)

            # dwa_weights = dynamic_weight_average(LOSS_T_1, LOSS_T_2, args)
            loss = 0.5 * full_bce_loss + 0.5 * full_dice_loss

            LOSS_T_2 = LOSS_T_1
            LOSS_T_1 = [full_bce_loss.item(), full_dice_loss.item()]


        loss.backward()
        optimizer.step()
        # if args.ema:
        #     update_ema_variables(net, ema_net, args.ema_alpha, step)

        epoch_loss.update(loss.item(), img.shape[0])
        # b_epoch_loss.update(boundary_loss.item(), img.shape[0])
        batch_time.update(time.time() - tic)
        tic = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
            try:
                print('Epoch: {} [{}/{}], loss: {:.3f}, full bce loss: {:.3f}, full dice loss: {:.3f}, boundary loss: {:.3f}, boundary dice loss: {:.3f}'.format(
                    epoch+1, i, len(trainLoader), loss.item(), full_bce_loss.item(), full_dice_loss.item(),
                    boundary_loss.item(), boundary_dice_loss.item()))
            except:
                print('Epoch: {} [{}/{}], loss: {:.3f}, full bce loss: {:.3f}, full dice loss: {:.3f}, boundary loss: {:.3f}, boundary dice loss: {:.3f}'.format(
                    epoch + 1, i, len(trainLoader), loss.item(), full_bce_loss.item(), full_dice_loss.item(),
                    boundary_loss, boundary_dice_loss))

        if args.dimension == '3d':
            iter_num_per_epoch += 1
            if iter_num_per_epoch > args.iter_per_epoch:
                break


        writer.add_scalar('Train/Loss', epoch_loss.avg, epoch+1)
        writer.add_scalar('Train/BLoss', b_epoch_loss.avg, epoch + 1)



    return LOSS_T_1, LOSS_T_2


def init_network(args):
    net = get_model(args, pretrain=args.pretrain)

    if args.ema:
        ema_net = get_model(args, pretrain=args.pretrain)
        for p in ema_net.parameters():
            p.requires_grad_(False)
        logging.info("Use EMA model for evaluation")
    else:
        ema_net = None
    
    if args.resume:
        resume_load_model_checkpoint(net, ema_net, args)
    
    

    if args.torch_compile:
        net = torch.compile(net)
    return net, ema_net 


if __name__ == '__main__':
    args = get_parser()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    torch.multiprocessing.set_start_method('spawn')
    torch.multiprocessing.set_sharing_strategy('file_system')
    
    args.log_path = args.log_path + '%s/' % args.dataset
    

    if args.reproduce_seed is not None:
        random.seed(args.reproduce_seed)
        np.random.seed(args.reproduce_seed)
        torch.manual_seed(args.reproduce_seed)

        if hasattr(torch, 'set_deterministic'):
            torch.set_deterministic(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
   
    Dice_list, HD_list, ASD_list = [], [], []

    for fold_idx in range(args.k_fold):
        
        args.cp_dir = f"{args.cp_path}/{args.dataset}/{args.unique_name}"
        os.makedirs(args.cp_dir, exist_ok=True)
        configure_logger(0, args.cp_dir+f"/fold_{fold_idx}.txt")
        save_configure(args)
        logging.info(
            f"\nDataset: {args.dataset},\n"
            + f"Model: {args.model},\n"
            + f"Dimension: {args.dimension}"
        )

        net, ema_net = init_network(args)

        net.cuda(args.proc_idx)
        if args.ema:
            ema_net.cuda(args.proc_idx)
        logging.info(f"Created Model")
        best_Dice, best_HD, best_ASD = train_net(net, args, ema_net, fold_idx=fold_idx)

        logging.info(f"Training and evaluation on Fold {fold_idx} is done")

        Dice_list.append(best_Dice)
        HD_list.append(best_HD)
        ASD_list.append(best_ASD)
        break
    

    ############################################################################################3
    # Save the cross validation results
    total_Dice = np.vstack(Dice_list)
    total_HD = np.vstack(HD_list)
    total_ASD = np.vstack(ASD_list)
    

    with open(f"{args.cp_path}/{args.dataset}/{args.unique_name}/cross_validationvalidation.txt",  'w') as f:
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
