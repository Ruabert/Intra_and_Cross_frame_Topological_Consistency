import torch
import torch.nn as nn
import torch.nn.functional as F
from MyMetric.utils import calculate_distance, calculate_dice, calculate_dice_split
from dataset.dim2.utils import flow_to_image
import numpy as np
import logging
import pdb
from utils import is_master
from tqdm import tqdm
import SimpleITK as sitk
import cv2
import os

def vis_slice(array):
    H, W = array.shape

    colored_array = np.zeros((H, W, 3))

    for i in range(3):
        colored_array[:, :, i] = array

    array_1 = np.where(colored_array == [1, 1, 1], [9, 87, 229], [0, 0, 0])
    array_254 = np.where(colored_array == [2, 2, 2], [138, 29, 80], [0, 0, 0])
    colored_array = array_1 + array_254
    colored_array = colored_array.astype(np.float64)

    return colored_array


def validation(net, dataloader, args):
    dice_list = []
    ASD_list = []
    HD_list = []
    for i in range(args.classes-1):  # background is not including in validation
        dice_list.append([])
        ASD_list.append([])
        HD_list.append([])

    logging.info("Evaluating")

    net.eval()

    with torch.no_grad():
        iterator = tqdm(dataloader)
        for items in iterator:
            # spacing here is used for distance metrics calculation
            inputs, labels = items[0].float().cuda(), items[1].cuda().to(torch.int8)
            res_dict = net(inputs)
            pred = res_dict['out_full'][0]

            _, label_pred = torch.max(pred, dim=2)

            # just use mid frame
            label_pred = label_pred[:, 1, ...].to(torch.int8)
            labels = labels[:, :, 1, ...].squeeze(0)

            tmp_ASD_list, tmp_HD_list = calculate_distance(label_pred, labels, args.classes)

            tmp_ASD_list = np.clip(np.nan_to_num(tmp_ASD_list, nan=500), 0, 500)
            tmp_HD_list = np.clip(np.nan_to_num(tmp_HD_list, nan=500), 0, 500)

            dice, _, _ = calculate_dice(label_pred.view(-1, 1), labels.view(-1, 1), args.classes)

            # exclude background
            dice = dice.cpu().numpy()[1:]

            unique_cls = torch.unique(labels)

            for cls in range(0, args.classes-1):
                if cls+1 in unique_cls: 
                    # in case some classes are missing in the GT
                    # only classes appear in the GT are used for evaluation
                    ASD_list[cls].append(tmp_ASD_list[cls])
                    HD_list[cls].append(tmp_HD_list[cls])
                    dice_list[cls].append(dice[cls])

    out_dice = []
    out_ASD = []
    out_HD = []
    for cls in range(0, args.classes-1):
        out_dice.append(np.array(dice_list[cls]).mean())
        out_ASD.append(np.array(ASD_list[cls]).mean())
        out_HD.append(np.array(HD_list[cls]).mean())

    print('Validation Over. Dice: {}, ASD: {}, HD: {}'.format(out_dice, out_ASD, out_HD))
    return np.array(out_dice), np.array(out_ASD), np.array(out_HD)



def validation_ddp(net, dataloader, args):
    dice_list = []
    ASD_list = []
    HD_list = []
    for i in range(args.classes - 1):  # background is not including in validation
        dice_list.append([])
        ASD_list.append([])
        HD_list.append([])

    logging.info("Evaluating")
    unique_labels_list = []

    logging.info(f"Evaluating")

    net.eval()

    vis_num = 32
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

    # --------------- FlOW VIS ----------------
#     # edge pred vis
#     vis_pred_edge_cls1 = []
#     vis_pred_edge_cls2 = []

#     vis_pred_edge_t_1_cls1 = []
#     vis_pred_edge_t_1_cls2 = []
#     vis_pred_edge_t_cls1 = []
#     vis_pred_edge_t_cls2 = []
    vis_flow_cls1 = []
    vis_flow_cls2 = []

    with torch.no_grad():
        iterator = tqdm(dataloader) if is_master(args) else dataloader
        for b, inputs in enumerate(iterator):
            # spacing here is used for distance metrics calculation
            imgs, labels, sdt_lab = inputs['img'].cuda(args.proc_idx), inputs['lab'].cuda(args.proc_idx), inputs['sdt_lab'].cuda(args.proc_idx)
            # edge_lab = inputs['edge_lab'].cuda(args.proc_idx)

            res_dict = net(imgs)

            pred = res_dict['out_full'][-1]
            sdt_out = res_dict['out_sdt'][-1]
            # edge_out = res_dict['mask_to_edge'][-1]
            try:
                flows = res_dict['optical_flow']  # there is a list
                flow_cls1 = flows[0][0].squeeze(0).cpu().numpy()
                flow_cls2 = flows[-1][0].squeeze(0).cpu().numpy()
                # flow = res_dict['optical_flow'][0].squeeze(0).cpu().numpy()
            except:
                print('flows load fail, plz check its type {}'.format(type(flows)))
            


            # just use mid frame
            _, label_pred = torch.max(F.softmax(pred, dim=2), dim=2)
            label_pred = label_pred[:, 1, ...].to(torch.int8)

            imgs = imgs[:, 1, ...].squeeze(0)
            labels = labels[:, 1, ...].squeeze(0)
            sdt_lab = sdt_lab[:, 1, ...].squeeze(0)
            # edge_lab = edge_lab[:, 1, ...].squeeze(0)


            sdt_out_pred = sdt_out[:, 1, ...]
            sdt_out_pred = F.tanh(sdt_out_pred)
            # edge_out_pred = edge_out[:, 1, ...]

            tmp_ASD_list, tmp_HD_list = calculate_distance(label_pred, labels, args.classes)
            tmp_ASD_list = np.clip(np.nan_to_num(tmp_ASD_list, nan=500), 0, 500)
            tmp_HD_list = np.clip(np.nan_to_num(tmp_HD_list, nan=500), 0, 500)

            dice, _, _ = calculate_dice(label_pred.view(-1, 1), labels.view(-1, 1), args.classes)
            # exclude background
            dice = dice.cpu().numpy()[1:]
            unique_cls = torch.unique(labels)

            for cls in range(0, args.classes - 1):
                if cls + 1 in unique_cls:
                    # in case some classes are missing in the GT
                    # only classes appear in the GT are used for evaluation
                    ASD_list[cls].append(tmp_ASD_list[cls])
                    HD_list[cls].append(tmp_HD_list[cls])
                    dice_list[cls].append(dice[cls])

            # save inference out
            if (b % 10 == 0) and (n_s < vis_num):
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
                sdt_cls1_frame = sdt_lab.squeeze(0)[0].cpu().numpy() * 256
                vis_sdt_cls1.append(sdt_cls1_frame)

                sdt_cls2_frame = sdt_lab.squeeze(0)[-1].cpu().numpy() * 256
                vis_sdt_cls2.append(sdt_cls2_frame)

                # # edge GT vis
                # edge_cls1_frame = edge_lab.squeeze(0)[1].cpu().numpy() * 256
                # vis_edge_cls1.append(edge_cls1_frame)
                #
                # edge_cls2_frame = edge_lab.squeeze(0)[-1].cpu().numpy() * 256
                # vis_edge_cls2.append(edge_cls2_frame)

                # --------------- RESULT VIS ----------------

                # pred vis
                vis_pred.append(vis_slice(label_pred.squeeze(0).cpu().numpy()))

                # sdt pred vis
                pred_sdt_cls1_frame = sdt_out_pred.squeeze(0)[0].cpu().numpy() * 256
                vis_pred_sdt_cls1.append(pred_sdt_cls1_frame)

                pred_sdt_cls2_frame = sdt_out_pred.squeeze(0)[-1].cpu().numpy() * 256
                vis_pred_sdt_cls2.append(pred_sdt_cls2_frame)

#                 # edge pred vis
#                 pred_edge_cls1_frame = edge_out_pred.squeeze(0)[1].cpu().numpy() * 256
#                 vis_pred_edge_cls1.append(pred_edge_cls1_frame)

#                 pred_edge_cls2_frame = edge_out_pred.squeeze(0)[-1].cpu().numpy() * 256
#                 vis_pred_edge_cls2.append(pred_edge_cls2_frame)

#                 # --------------- FlOW VIS ----------------
#                 # edge pred vis
#                 edge_t_1, edge_t = edge_out[:, 0, ...], edge_out[:, 1, ...]
#                 pred_edge_t_1_cls1_frame = edge_t_1.squeeze(0)[0].cpu().numpy() * 256
#                 pred_edge_t_1_cls2_frame = edge_t_1.squeeze(0)[-1].cpu().numpy() * 256
#                 vis_pred_edge_t_1_cls1.append(pred_edge_t_1_cls1_frame)
#                 vis_pred_edge_t_1_cls2.append(pred_edge_t_1_cls2_frame)

#                 pred_edge_t_cls1_frame = edge_t.squeeze(0)[0].cpu().numpy() * 256
#                 pred_edge_t_cls2_frame = edge_t.squeeze(0)[-1].cpu().numpy() * 256
#                 vis_pred_edge_t_cls1.append(pred_edge_t_cls1_frame)
#                 vis_pred_edge_t_cls2.append(pred_edge_t_cls2_frame)

                # flow vis
                try:
                    flow_cls1 = np.transpose(flow_cls1, (1, 2, 0))
                    vis_flow_cls1.append(flow_to_image(flow_cls1))
                    flow_cls2 = np.transpose(flow_cls2, (1, 2, 0))
                    vis_flow_cls2.append(flow_to_image(flow_cls2))
                except:
                    print('flow vis fail, plz check the type')
                    pass


    out_dice = []
    out_ASD = []
    out_HD = []
    for cls in range(0, args.classes - 1):
        out_dice.append(np.array(dice_list[cls]).mean())
        out_ASD.append(np.array(ASD_list[cls]).mean())
        out_HD.append(np.array(HD_list[cls]).mean())

    print('Validation Over. Dice: {}, ASD: {}, HD: {}'.format(out_dice, out_ASD, out_HD))



    pic_save_path = os.path.join(args.cp_path, args.dataset, args.unique_name)
    map_show = np.concatenate([np.concatenate(vis_img, axis=1),
                               np.concatenate(vis_GT, axis=1),
                               np.concatenate(vis_pred, axis=1)])
    cv2.imwrite(os.path.join('{}/EvalOut'.format(pic_save_path), '{}_seg.png'.format(args.model)), map_show)

    map_show = np.concatenate([np.concatenate(vis_sdt_cls1, axis=1),
                               np.concatenate(vis_sdt_cls2, axis=1),
                               np.concatenate(vis_pred_sdt_cls1, axis=1),
                               np.concatenate(vis_pred_sdt_cls2, axis=1)])
    cv2.imwrite(os.path.join('{}/EvalOut'.format(pic_save_path), '{}_reg.png'.format(args.model)), map_show)

#     map_show = np.concatenate([np.concatenate(vis_edge_cls1, axis=1),
#                                np.concatenate(vis_edge_cls2, axis=1),
#                                np.concatenate(vis_pred_edge_cls1, axis=1),
#                                np.concatenate(vis_pred_edge_cls2, axis=1)])
#     cv2.imwrite(os.path.join('{}/EvalOut'.format(pic_save_path), '{}_edge_est.png'.format(args.model)), map_show)

#     map_show = np.concatenate([np.concatenate(vis_pred_edge_t_1_cls1, axis=1),
#                                np.concatenate(vis_pred_edge_t_1_cls2, axis=1),
#                                np.concatenate(vis_pred_edge_t_cls1, axis=1),
#                                np.concatenate(vis_pred_edge_t_cls2, axis=1)])

#     cv2.imwrite(os.path.join('{}/EvalOut'.format(pic_save_path), '{}_nearby_edge_est.png'.format(args.model)), map_show)

    try:
        map_show = np.concatenate([np.concatenate(vis_flow_cls1, axis=1), np.concatenate(vis_flow_cls2, axis=1)])
        cv2.imwrite(os.path.join('{}/EvalOut'.format(pic_save_path), '{}_forward_flow.png'.format(args.model)), map_show)
    except:
        pass


    return np.array(out_dice), np.array(out_ASD), np.array(out_HD)

