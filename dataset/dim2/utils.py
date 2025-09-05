import torch
import numpy as np
import cv2
from torch import Tensor
from torchvision import transforms
from operator import itemgetter, mul
from functools import partial, reduce
from typing import Callable, BinaryIO, Match, Pattern, Tuple, Union, Optional
from dataset.utils import class2one_hot, one_hot2dist_Neg
from PIL import Image, ImageOps
from scipy.ndimage.morphology import distance_transform_edt
from skimage import morphology
import math
import torch.nn.functional as F

# from scipy.ndimage import distance_transform_edt as distance

# cv2.cuda.setDevice(0)

D = Union[Image.Image, np.ndarray, Tensor]

def rotate_2d(image, b, crop):
    b = -math.radians(b % 360)  # 将角度化为弧度
    n = np.size(image, 0)
    m = np.size(image, 1)
    center = (n/2.0, m/2.0)
    img = np.zeros((n, m))

    # 不裁剪
    if not crop:
        # 计算图幅
        xx = []
        yy = []
        for x0, y0 in ((0, 0), (n, 0), (n, m), (0, m)):
            x = (x0 - center[0]) * math.cos(b) + (y0 - center[1]) * math.sin(b)
            y = -1*(x0 - center[0]) * math.sin(b) + (y0 - center[1]) * math.cos(b)
            xx.append(x)
            yy.append(y)
        nn = int(math.ceil(max(xx)) - math.floor(min(xx)))
        nm = int(math.ceil(max(yy)) - math.floor(min(yy)))
        img = np.zeros((nn, nm))
        center = (nn/2, nm/2)

    # 裁剪
    if crop:
        nn = n
        nm = m

    def inmap(x, y):
       return True if x >= 0 and x < n and y >= 0 and y < m else False

    # 反向推
    for x in range(nn):
        for y in range(nm):
            x0 = (x-center[0])*math.cos(-b)+(y-center[1])*math.sin(-b)+center[0]
            y0 = -1*(x-center[0])*math.sin(-b)+(y-center[1])*math.cos(-b)+center[1]
            # 将坐标对齐
            x0 = x0-(nn-n)/2
            y0 = y0-(nm-m)/2
            # 双线性内插值
            i = int(x0)
            j = int(y0)
            u = x0 - i
            v = y0 - j
            if inmap(i, j):
                f1 = (1-u)*(1-v)*image[i, j]
                img[x, y] += f1
                if inmap(i, j+1):
                    f2 = (1-u)*v*image[i, j+1]
                    img[x, y] += f2
                if inmap(i+1, j):
                    f3 = u*(1-v)*image[i+1, j]
                    img[x, y] += f3
                if inmap(i+1, j+1):
                    f4 = u*v*image[i+1, j+1]
                    img[x, y] += f4
    return img

def random_rotate_cl(data, angle, mode = 'img'):
    data_r = np.zeros((data.shape))

    if mode == 'img':
        data_r = rotate_2d(data, b=angle, crop=True)
    if mode == 'gt':
        try:
            label_I = Image.fromarray(data)
        except:
            label_I = Image.fromarray(data.astype(np.uint8))

        label_I = label_I.rotate(angle)
        data_r = np.asarray(label_I)

    return data_r


def Lesion_Mix(prev_img, prev_mask, warped_img, warped_mask, num_classes=3, alpha = 0.3):
    warped_mask = warped_mask.long()
    prev_mask_oh = class2one_hot(prev_mask.unsqueeze(0), num_classes).squeeze(0)
    warped_mask_oh = class2one_hot(warped_mask.unsqueeze(0), num_classes).squeeze(0)

    _LM_mask = torch.zeros_like(prev_mask_oh)
    LM_img = torch.zeros_like(prev_img)
    for i in range(prev_mask_oh.shape[0]):
        if i == 0:
            continue

        # _LM_mask[i] = np.multiply((1 - warped_mask_oh[i]), prev_mask_oh[i]) + ((1 - alpha) * warped_mask_oh[i] + alpha * prev_mask_oh[i])
        # LM_img += np.multiply((1 - warped_mask_oh[i]), prev_img) + ((1 - alpha) * np.multiply(warped_img, warped_mask_oh[i]) + alpha * np.multiply(prev_img, prev_mask_oh[i]))
        _LM_mask[i] = np.multiply((1 - warped_mask_oh[i]), prev_mask_oh[i]) + warped_mask_oh[i]
        LM_img += np.multiply((1 - warped_mask_oh[i]), prev_img) + np.multiply(warped_img, warped_mask_oh[i])

    LM_img = np.divide(LM_img, prev_mask_oh.shape[0] - 1)

    LM_mask = torch.zeros((_LM_mask.shape[1], _LM_mask.shape[-1]))
    for i in range(prev_mask_oh.shape[0]):
        if i == 0:
            continue
        LM_mask += np.where(_LM_mask[i] == 1, i, 0)


    return LM_img, LM_mask



def to_skeleton_distance_transform(mask, alpha=0.8, num_classes=3):
    o_mask = mask

    mask = torch.tensor(mask).long()
    mask_onehot = class2one_hot(mask.unsqueeze(0), num_classes).squeeze(0)

    sdt_lab = torch.zeros((num_classes - 1, 64, 64))
    edge_lab = torch.zeros((num_classes - 1, 64, 64))

    tmp_edge_lab = to_multiclass_edges(mask, num_classes).numpy().astype(np.float32)
    tmp_skt_lab = to_multiclass_skeletonize(mask, num_classes).numpy().astype(np.float32)
    o_skt_lab = to_multiclass_skeletonize(o_mask, num_classes).numpy().astype(np.float32)

    edge_lab_dist = one_hot2dist_Neg(tmp_edge_lab)
    skt_lab_dist = one_hot2dist_Neg(tmp_skt_lab)

    total_dist = edge_lab_dist + skt_lab_dist
    tmp_sdt_lab = np.divide(edge_lab_dist, total_dist, where=total_dist != 0,
                            out=np.zeros_like(total_dist).astype(np.float))
    tmp_sdt_lab = np.power(tmp_sdt_lab, alpha)
    tmp_sdt_lab = torch.tensor(tmp_sdt_lab).float()

    for i in range(1, num_classes):
        if len(torch.where(mask == i)[0]) == 0:
            sdt_lab[i - 1, ...] = torch.zeros((64, 64))
        else:
            negdis = distance_transform_edt(~mask_onehot[i, ...].to(torch.bool))
            pos_dist_mask = torch.tensor(negdis <= 1.0).to(torch.int8)
            neg_dist_mask = 0.1 * torch.tensor(negdis > 1.0).to(torch.int8)

            tmp = tmp_sdt_lab[i, ...] + o_skt_lab[i, ...]
            tmp = tmp * pos_dist_mask + tmp * neg_dist_mask

            sdt_lab[i - 1, ...] = torch.where(tmp.double() > 1.0, 1.0, tmp.double())

    return sdt_lab

def to_multiclass_edges(mask, radius=1, num_classes=3):
    if radius < 0:
        return mask

    # We need to pad the borders for boundary conditions

    mask_onehot = class2one_hot(mask.unsqueeze(0), num_classes)
    mask_onehot = mask_onehot.squeeze(0)
    mask_pad = np.pad(mask_onehot, ((0, 0), (1, 1), (1, 1)), mode='constant', constant_values=0)
    bg_mask = mask_pad[0, :, :].astype(np.uint8)
    channels = []
    for i in range(num_classes):
        if i == 0:
            channels.append(bg_mask[1:-1, 1:-1])
            continue
        dist = distance_transform_edt(mask_pad[i, :, :]) + distance_transform_edt(1.0 - mask_pad[i, :, :])
        dist = dist[1:-1, 1:-1]
        dist[dist > radius] = 0
        dist = (dist == 1).astype(np.uint8)
        # dist = morphology.skeletonize(dist) * 1
        channels.append(dist)

    return torch.tensor(np.array(channels)).long()

def to_multiclass_skeletonize(mask, num_classes=3):
    mask_onehot = class2one_hot(mask.unsqueeze(0), num_classes)
    mask_onehot = mask_onehot.squeeze(0)
    bg_mask = mask_onehot[0, :, :].numpy()

    skeleton_mask = np.zeros_like(mask_onehot)
    for i in range(num_classes):
        if i == 0:
            skeleton_mask[i] = bg_mask
            continue
        skeleton_mask[i] = morphology.skeletonize(mask_onehot[i].numpy())*1

    skeleton_mask = skeleton_mask
    return torch.tensor(skeleton_mask)

def make_color_wheel():
    """
    Generate color wheel according Middlebury color code
    :return: Color wheel
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3])

    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255*np.arange(0, RY) / RY))
    col += RY

    # YG
    colorwheel[col:col+YG, 0] = 255 - np.transpose(np.floor(255*np.arange(0, YG) / YG))
    colorwheel[col:col+YG, 1] = 255
    col += YG

    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.transpose(np.floor(255*np.arange(0, GC) / GC))
    col += GC

    # CB
    colorwheel[col:col+CB, 1] = 255 - np.transpose(np.floor(255*np.arange(0, CB) / CB))
    colorwheel[col:col+CB, 2] = 255
    col += CB

    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.transpose(np.floor(255*np.arange(0, BM) / BM))
    col += + BM

    # MR
    colorwheel[col:col+MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col+MR, 0] = 255

    return colorwheel

def compute_color(u, v):
    """
    compute optical flow color map
    :param u: optical flow horizontal map
    :param v: optical flow vertical map
    :return: optical flow in color code
    """
    [h, w] = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0

    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)

    rad = np.sqrt(u**2+v**2)

    a = np.arctan2(-v, -u) / np.pi

    fk = (a+1) / 2 * (ncols - 1) + 1

    k0 = np.floor(fk).astype(int)

    k1 = k0 + 1
    k1[k1 == ncols+1] = 1
    f = fk - k0

    for i in range(0, np.size(colorwheel,1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0-1] / 255
        col1 = tmp[k1-1] / 255
        col = (1-f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1-rad[idx]*(1-col[idx])
        notidx = np.logical_not(idx)

        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col*(1-nanIdx)))

    return img

def flow_to_image(flow):
    """
    Convert flow into middlebury color code image
    :param flow: optical flow map
    :return: optical flow image in middlebury color
    """
    u = flow[:, :, 0]
    v = flow[:, :, 1]

    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.
    UNKNOWN_FLOW_THRESH = 1e7
    SMALLFLOW = 0.0
    LARGEFLOW = 1e8

    idxUnknow = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH)
    u[idxUnknow] = 0
    v[idxUnknow] = 0

    maxu = max(maxu, np.max(u))
    minu = min(minu, np.min(u))

    maxv = max(maxv, np.max(v))
    minv = min(minv, np.min(v))

    rad = np.sqrt(u ** 2 + v ** 2)
    maxrad = max(-1, np.max(rad))

    u = u/(maxrad + np.finfo(float).eps)
    v = v/(maxrad + np.finfo(float).eps)

    img = compute_color(u, v)

    idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0

    return np.uint8(img)







