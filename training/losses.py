import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import pdb
from torch import Tensor, einsum
from dataset.utils import one_hot, simplex, class2one_hot, one_hot2dist
from typing import Any, Callable, Iterable, List, Set, Tuple, TypeVar, Union, cast
from model.MyModel import uflow_utils

class DiceLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, size_average=True, reduce=True, one_hot=False, soft_label=False, t=10):
        super(DiceLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

        self.size_average = size_average
        self.reduce = reduce

        self.one_hot = one_hot

        self.soft_label = soft_label
        self.t = t

    def forward(self, preds, targets):
        N = preds.size(0)
        C = preds.size(1)

        if self.one_hot == False:
            class_mask = torch.zeros(preds.shape).to(preds.device)
            class_mask.scatter_(1, targets, 1.)
        else:
            class_mask = targets

        if self.soft_label == True:
            P = F.log_softmax(preds / self.t, dim=1)
            class_mask = F.softmax(class_mask / self.t, dim=1)
        else:
            P = F.softmax(preds, dim=1)

        smooth = torch.zeros(C, dtype=torch.float32).fill_(0.00001)
        ones = torch.ones(preds.shape).to(preds.device)

        P_ = ones - P
        class_mask_ = ones - class_mask

        TP = P * class_mask
        FP = P * class_mask_
        FN = P_ * class_mask

        smooth = smooth.to(preds.device)
        self.alpha = FP.transpose(0, 1).reshape(C, -1).sum(dim=(1)) / ((FP.transpose(0, 1).reshape(C, -1).sum(dim=(1)) + FN.transpose(0, 1).reshape(C, -1).sum(dim=(1))) + smooth)
    
        self.alpha = torch.clamp(self.alpha, min=0.2, max=0.8) 
        #print('alpha:', self.alpha)
        self.beta = 1 - self.alpha
        num = torch.sum(TP.transpose(0, 1).reshape(C, -1), dim=(1)).float()
        den = num + self.alpha * torch.sum(FP.transpose(0, 1).reshape(C, -1), dim=(1)).float() + self.beta * torch.sum(FN.transpose(0, 1).reshape(C, -1), dim=(1)).float()

        dice = num / (den + smooth)

        if not self.reduce:
            loss = torch.ones(C).to(dice.device) - dice
            return loss

        loss = 1 - dice
        loss = loss.sum()

        if self.size_average:
            loss /= C

        return loss

class FocalLoss(nn.Module):
    def __init__(self, class_num, alpha=None, gamma=10, size_average=True, one_hot=False, soft_label=False, t=10):
        super(FocalLoss, self).__init__()

        if alpha is None:
            self.alpha = torch.ones(class_num)
        else:
            self.alpha = alpha

        self.gamma = gamma
        self.size_average = size_average
        self.one_hot = one_hot

        self.soft_label = soft_label
        self.t = t

    def forward(self, preds, targets):
        N = preds.size(0)
        C = preds.size(1)

        if self.one_hot == False:
            class_mask = torch.zeros(preds.shape).to(preds.device)
            class_mask.scatter_(1, targets, 1.)
        else:
            class_mask = targets

        if targets.size(1) == 1:
            # squeeze the chaneel for target
            targets = targets.squeeze(1)

        if self.soft_label == True:
            P = F.softmax(preds / self.t, dim=1)
            log_P = F.log_softmax(preds / self.t, dim=1)
            class_mask = F.softmax(class_mask / self.t, dim=1)
            _, pl = torch.max(class_mask, dim=1)
            alpha = self.alpha[pl]
        else:
            P = F.softmax(preds, dim=1)
            log_P = F.log_softmax(preds, dim=1)
            alpha = self.alpha[targets.data]

        alpha = alpha.to(preds.device)

        probs = (P * class_mask).sum(1)
        log_probs = (log_P * class_mask).sum(1)


        batch_loss = -alpha * (1-probs).pow(self.gamma)*log_probs

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()

        return loss

class SmoothL1(nn.Module):
    def __init__(self):
        super(SmoothL1, self).__init__()
        pass
    def forward(self):
        return

class FlowLoss(nn.Module):
    def __init__(self):
        super(FlowLoss, self).__init__()

    def forward(self, img1, img2, features1, features2, flows):
        """
        :param img1: [BCHW] image 1 batch
        :param img2: [BCHW] image 2 batch
        :param features1: feature pyramid list for img1
        :param features2: feature pyramid list for img2
        :param flows: [B2HW] flow tensor for each image pair in batch
        """
        f1 = [img1] + features1
        f2 = [img2] + features2

        warps = [uflow_utils.flow_to_warp(f) for f in flows]
        warped_f2 = [uflow_utils.resample(f, w) for (f, w) in zip(f2, warps)]

        loss = uflow_utils.compute_all_loss(f1, warped_f2, flows)
        return loss

class FlowConsistency(nn.Module):
    def __init__(self):
        super(FlowConsistency, self).__init__()

    def forward(self, img1, img2, flows):
        """
        :param img1: [BCHW] image 1 batch
        :param img2: [BCHW] image 2 batch
        :param flows: [B2HW] flow tensor for each image pair in batch
        """
        warp = uflow_utils.flow_to_warp(flows[0])
        warped_f2 = uflow_utils.resample(img2, warp)

        loss = uflow_utils.compute_loss(img1, warped_f2, flows, use_mag_loss=False)
        return loss



def dynamic_weight_average(loss_t_1, loss_t_2, args):
    T = args.epochs
    if len(loss_t_1) == 0 or len(loss_t_2) == 0:
        return [1.0, 1.0, 1.0, 1.0]

    assert len(loss_t_1) == len(loss_t_2)
    task_n = len(loss_t_1)

    w = [l_1 / l_2 for l_1, l_2 in zip(loss_t_1, loss_t_2)]

    lamb = [math.exp(v / T) for v in w]

    lamb_sum = sum(lamb)

    return [task_n * l / lamb_sum for l in lamb]



def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))  # e^(-5*(1-t)^2)

def get_current_consistency_weight(epoch, consistency=0.1, consistency_rampup=100.0):
    return consistency * sigmoid_rampup(epoch, consistency_rampup)





# if __name__ == '__main__':
#     att = torch.randn(32, 3, 64, 64).cuda()
#     gt = torch.randn(32, 1, 64, 64).cuda()
#
#     loss = FocalLoss(class_num=3)
#
#     l = loss(att, gt.long())
#     print(l)


