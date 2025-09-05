from config.args import get_parser
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import numpy as np
from tqdm import tqdm
import torch.distributed as dist
import cv2
import os
import warnings
from dataset.dim2.dataset_acdc import SDTDataset_ForTest
from model.utils import get_model
import medpy.metric as Metric
from skimage import restoration

warnings.filterwarnings("ignore", category=UserWarning)



def prediction(args, net, testLoader):
    # init metric list
    dice_list = []
    ASD_list = []
    HD_list = []
    for i in range(args.classes - 1):  # background is not including in validation
        dice_list.append([])
        ASD_list.append([])
        HD_list.append([])


    net.eval()

    vis_num = 64
    n_s = 0

    # img and GT vis
    vis_img = []

    # --------------- LABEL VIS ----------------
    vis_GT = []

    # sdt gt vis
    vis_sdt_cls1 = []
    vis_sdt_cls2 = []

    # edge gt vis
    vis_edge_cls1 = []
    vis_edge_cls2 = []

    # --------------- RESULT VIS ----------------
    # mask pred vis
    vis_pred = []

    # sdt pred vis
    vis_pred_sdt_cls1 = []
    vis_pred_sdt_cls2 = []

    # edge pred vis
    vis_pred_edge_cls1 = []
    vis_pred_edge_cls2 = []

    # Metric
    dice_list = []
    asd_list = []
    hd95_list = []
    jac_list = []
    with torch.no_grad():
        iterator = tqdm(testLoader)
        for b, inputs in enumerate(iterator):
            imgs, labels = inputs['img'].cuda(args.proc_idx), inputs['lab'].cuda(args.proc_idx)
            # if labels.max() == .0:
            #     continue
            
            if args.model_mode == 'our':
                sdt_lab = inputs['sdt_lab'].cuda(args.proc_idx)

                res_dict = net(imgs)
                # get the inference res
                # multi-class act
                pred = res_dict['out_full'][0]
                sdt_out = res_dict['out_sdt'][0]

                # just use mid frame
                _, label_pred = torch.max(F.softmax(pred, dim=2), dim=2)
                label_pred = label_pred[:, 0, ...].to(torch.int8)

                imgs = imgs[:, 0, ...].squeeze(0)
                labels = labels[:, 0, ...].squeeze(0)
                sdt_lab = sdt_lab[:, 0, ...].squeeze(0)

                sdt_out_pred = sdt_out[:, 0, ...]
                sdt_out_pred = F.tanh(sdt_out_pred)
            else:
                pred = net(imgs)
                try:
                    _, label_pred = torch.max(F.softmax(pred, dim=1), dim=1)
                except:
                    _, label_pred = torch.max(F.softmax(pred[0], dim=1), dim=1)
                labels = labels.squeeze(0)
            
            label_unique_cls = torch.unique(labels)
            pred_unique_cls = torch.unique(label_pred)
            labels_onehot = torch.nn.functional.one_hot(labels.squeeze(0).to(torch.int64), num_classes=3).permute(2, 0, 1)
            label_pred_onehot = torch.nn.functional.one_hot(label_pred.squeeze(0).to(torch.int64), num_classes=3).permute(2, 0, 1)

            # calc Dice, calc ASD and HD 95
            tmp_dice = Metric.binary.dc(label_pred.cpu().numpy(), labels.cpu().numpy())

            try:
                tmp_jac = Metric.binary.jc(label_pred.cpu().numpy(), labels.cpu().numpy())
            except:
                if labels.max() == label_pred.max():
                    tmp_jac = 1
                else:
                    tmp_jac = 0

    

            try:
                tmp_hd95 = Metric.binary.hd95(label_pred_onehot.cpu().numpy()[1:, :, :], labels_onehot.cpu().numpy()[1:, :, :])
            except:
                if labels.max() == label_pred.max():
                    tmp_hd95 = 0
                else:
                    tmp_hd95 = 100



            try:
                tmp_asd = Metric.binary.asd(label_pred_onehot.cpu().numpy()[1:, :, :], labels_onehot.cpu().numpy()[1:, :, :])
            except:
                if labels.max() == label_pred.max():
                    tmp_asd = 0
                else:
                    tmp_asd = 100
 

            # print('Dice: {}, ASD: {}, HD: {}'.format(tmp_dice, tmp_asd, tmp_hd95))

            dice_list.append(tmp_dice)
            asd_list.append(tmp_asd)
            hd95_list.append(tmp_hd95)
            jac_list.append(tmp_jac)

            if args.model_mode == 'our':
                # save inference out
                if (b % 10 == 0) and (2 in label_unique_cls) and (n_s < vis_num):
                    n_s = n_s + 1
                    # img vis
                    raw_frame = np.zeros((imgs.shape[1], imgs.shape[-1], 3))
                    for c in range(3):
                        raw_frame[:, :, c] = imgs.squeeze(0).cpu().numpy()
                    vis_img.append(raw_frame)

                    # --------------- LABEL VIS ----------------
                    # GT vis
                    vis_GT.append(vis_slice(labels.squeeze(0).cpu().numpy()))

                    # sdt GT vis
                    sdt_cls1_frame = sdt_lab.squeeze(0)[0].cpu().numpy() * 255
                    vis_sdt_cls1.append(sdt_cls1_frame)

                    sdt_cls2_frame = sdt_lab.squeeze(0)[-1].cpu().numpy() * 255
                    vis_sdt_cls2.append(sdt_cls2_frame)

                    # --------------- RESULT VIS ----------------

                    # pred vis
                    vis_pred.append(vis_slice(label_pred.squeeze(0).cpu().numpy()))

                    # sdt pred vis
                    pred_sdt_cls1_frame = sdt_out_pred.squeeze(0)[0].cpu().numpy() * 255
                    vis_pred_sdt_cls1.append(pred_sdt_cls1_frame)

                    pred_sdt_cls2_frame = sdt_out_pred.squeeze(0)[-1].cpu().numpy() * 255
                    vis_pred_sdt_cls2.append(pred_sdt_cls2_frame)


                # if n_s >= vis_num:
                #      break
            else:
                # if (b % 100 == 0) and (n_s < vis_num):
                if (b % 10 == 0) and (2 in label_unique_cls) and (n_s < vis_num):
                    n_s = n_s + 1
                    # img vis
                    raw_frame = np.zeros((imgs.shape[-2], imgs.shape[-1], 3))
                    for c in range(3):
                        raw_frame[:, :, c] = imgs.squeeze(0).cpu().numpy()
                    vis_img.append(raw_frame)

                    # GT vis
                    vis_GT.append(vis_slice(labels.squeeze(0).cpu().numpy()))

                    # pred vis
                    vis_pred.append(vis_slice(label_pred.squeeze(0).cpu().numpy()))
                # if n_s >= vis_num:
                #     break

    # summarize the metric
    out_dice = np.mean(dice_list)
    out_ASD = np.mean(asd_list)
    out_HD = np.mean(hd95_list)
    out_jac = np.mean(jac_list)


    print('Validation Over. Dice: {:.3f}, Jac: {:.3f}, HD: {:.3f}, ASD: {:.3f}'.format(out_dice, out_jac,
                                                                                           out_HD, out_ASD))

    map_show = np.concatenate([np.concatenate(vis_img, axis=1),
                               np.concatenate(vis_GT, axis=1),
                               np.concatenate(vis_pred, axis=1)])
    cv2.imwrite(os.path.join('./EvalOut', '{}_seg_Dice_{:.3f}_ JAC_{:.3f}_HD_{:.3f}_ASD_{:.3f}.png'.format(args.model, out_dice, out_jac, out_HD, out_ASD)), map_show)

    map_show = np.concatenate([np.concatenate(vis_sdt_cls1, axis=1),
                               np.concatenate(vis_sdt_cls2, axis=1),
                               np.concatenate(vis_pred_sdt_cls1, axis=1),
                               np.concatenate(vis_pred_sdt_cls2, axis=1)])
    cv2.imwrite(os.path.join('./EvalOut', '{}_reg.png'.format(args.model)), map_show)


def vis_slice(array):
    H, W = array.shape

    colored_array = np.zeros((H, W, 3))

    for i in range(3):
        colored_array[:, :, i] = array

    # array_1 = np.where(colored_array == [1, 1, 1], [9, 87, 229], [0, 0, 0])
    # array_254 = np.where(colored_array == [2, 2, 2], [138, 29, 80], [0, 0, 0])
    array_1 = np.where(colored_array == [1, 1, 1], [48, 45, 142], [0, 0, 0])
    array_254 = np.where(colored_array == [2, 2, 2], [226, 240, 254], [0, 0, 0])
    colored_array = array_1 + array_254
    colored_array = colored_array.astype(np.float64)

    return colored_array
    

if __name__ == '__main__':
    import os
    data_index = 'DATA1'
    
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '6789'

    args = get_parser()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    img_name_list = os.listdir(os.path.join(args.data_root, 'imagesTs'))
    img_name_list = ["_".join(p.split('_')[:3]) for p in img_name_list]
    
    length = len(img_name_list)
    img_name_list = img_name_list[: length]
    img_name_list = np.load('{}_test_list.npy'.format(data_index)).tolist()
    
    if args.model == 'USEN_UFLOW':
        print('model: {}, {} for testing'.format(args.model, length))
        net = get_model(args)
        testSet = SDTDataset_ForTest(args, img_name_list=img_name_list, mode=args.model_mode)
        testLoader = data.DataLoader(
            testSet,
            batch_size=1,
            shuffle=False,
            # num_workers=args.num_workers
            num_workers=0
        )
        # model_path_name = ['SEN_fold_0_best_full_model.pth', 
        #                   'SEN_Flow_Consistency_fold_0_best_full_model.pth',
        #                   'SEN_Structure_Consistency_fold_0_best_full_model.pth',
        #                   'SEN_Structure_and_Flow_Consistency_fold_0_best_full_model.pth']
        model_path_name = [#'10%SEN_two_branch_fold_0_best_full_model.pth', 
                          #'10%SEN_Flow_Consistency_fold_0_best_full_model.pth',
                          'FLOW_LF_fold_0_best_full_model.pth',
                            'FLOW_K_fold_0_best_full_model.pth',
                          'SEN_K_fold_0_best_full_model.pth',
                          'SEN_KB_fold_0_best_full_model.pth',
                          'SEN_KBK_fold_0_best_full_model.pth']
        
        dist.init_process_group(
            backend='nccl',
            world_size=1,
            rank=0
        )
        net = nn.DataParallel(net)

        for m in model_path_name:
            print(m)
            load_path = os.path.join(
                os.path.join(args.cp_path, '{}/{}_{}_{}'.format(args.dataset, args.dataset, args.dimension, args.arch)),
                m)
            net = torch.load(load_path)['model_state_dict']
            net.cuda(args.proc_idx)

            prediction(args, net, testLoader)
    else:
        net = get_model(args)
        testSet = SDTDataset_ForTest(args, img_name_list=img_name_list, mode=args.model_mode)
        testLoader = data.DataLoader(
            testSet,
            batch_size=1,
            shuffle=False,
            # num_workers=args.num_workers
            num_workers=0
        )
        load_path = os.path.join(os.path.join(args.cp_path, '{}/{}_{}_{}'.format(args.dataset, args.dataset, args.dimension, args.arch)), 'fold_0_best.pth')
        checkpoint = torch.load(load_path)
        ck_model_dict = checkpoint['model_state_dict']
        model_dict = net.state_dict()
        assert model_dict.keys() == ck_model_dict.keys()
        new_dict = {k: v for k, v in ck_model_dict.items() if k in model_dict}
        assert len(new_dict) != 0
        model_dict.update(new_dict)
        net.load_state_dict(new_dict)

        net.cuda(args.proc_idx)

        prediction(args, net, testLoader)


