import os
import torch
from torch.utils.data import Dataset
import numpy as np
from scipy.ndimage import distance_transform_edt as distance
from skimage import segmentation as skimage_seg
import torch.nn.functional as F
import random
import logging
import math
import nibabel as nib
from dataset.dim2.utils import to_multiclass_edges, random_rotate_cl, class2one_hot, to_skeleton_distance_transform
from torch.utils.data.distributed import DistributedSampler
from typing import TypeVar, Optional, Iterator
import cv2
import torch.distributed as dist
import itertools


def array_scale(array, t_hw=224, mode='img'):
    if len(array.shape) == 3:
        array = array.transpose(1, 2, 0)

    if mode == 'img':
        array = cv2.resize(array, [t_hw, t_hw], interpolation=cv2.INTER_CUBIC)
        if len(array.shape) == 3:
            array = array.transpose(2, 0, 1)

        return array

    elif mode == 'gt':
        array = cv2.resize(array, [t_hw, t_hw], interpolation=cv2.INTER_NEAREST)
        if len(array.shape) == 3:
            array = array.transpose(2, 0, 1)

        return array
    else:
        return

def cal_SDT(img, lab, t_hw=64):
    # bulid nearby img list
    nearby_img_tensor = torch.zeros([2, 1, 64, 64])  # frames, c, h, w
    nearby_lab_tensor = torch.zeros([2, 1, 64, 64])
    
    nearby_img_tensor[0, ...] = torch.from_numpy(img[0]).unsqueeze(0)
    nearby_lab_tensor[0, ...] = torch.from_numpy(lab[0]).unsqueeze(0)
 
    nearby_img_tensor[1, ...] = torch.from_numpy(img[1]).unsqueeze(0)
    nearby_lab_tensor[1, ...] = torch.from_numpy(lab[1]).unsqueeze(0)

    # bulid nearby SDT
    sdt_tensor = torch.zeros([2, 2, t_hw, t_hw])  # t, c-1, h, w
    for k in range(2):
        sdt_tensor[k] = to_skeleton_distance_transform(mask=nearby_lab_tensor[k].squeeze())


    return nearby_img_tensor.to(torch.float32), \
           nearby_lab_tensor.to(torch.int8), \
           sdt_tensor.to(torch.float32)

def get_test(img, lab, num_class=3, t_hw=64):
    # tensorring the scaled-array
    img = array_scale(img.astype(np.float32), t_hw, mode='img')
    lab = array_scale(lab.astype(np.float32), t_hw, mode='gt')

    # bulid nearby img list
    nearby_img_tensor = torch.zeros([2, 1, 64, 64])
    nearby_lab_tensor = torch.zeros([2, 1, 64, 64])


    if len(img.shape) == 3:
        for i in range(2):
            nearby_img_tensor[i] = torch.tensor(img[i]).unsqueeze(0)
            nearby_lab_tensor[i] = torch.tensor(lab[i]).unsqueeze(0)
    elif len(img.shape) == 2:
        for i in range(2):
            nearby_img_tensor[i] = torch.tensor(img).unsqueeze(0)
            nearby_lab_tensor[i] = torch.tensor(lab).unsqueeze(0)
    else:
        raise ValueError('img shape error')

    sdt_tensor = torch.zeros([2, 2, t_hw, t_hw])  # t, c-1, h, w
    edge_tensor = torch.zeros([2, 3, t_hw, t_hw])  # t, c, h, w

    for k in range(2):
        sdt_tensor[k] = to_skeleton_distance_transform(mask=nearby_lab_tensor[k].squeeze())

    return nearby_img_tensor.to(torch.float32), \
           nearby_lab_tensor.to(torch.int8), \
           sdt_tensor.to(torch.float32)

class SDTDataset(Dataset):
    def __init__(self, args, mode='train', fold_idx=0, debug_ratio=1):
        assert mode in ['train', 'test']
        logging.info(f"Start loading {mode} data")

        self.mode = mode
        self.args = args
        self.path = args.data_root

        # load labeled data length
        self.labeled_path = os.path.join(self.path, 'labeled', 'imagesTr')
        self.labeled_data_length = len(os.listdir(self.labeled_path)) // debug_ratio

        # load unlabeled data length
        self.unlabeled_path = os.path.join(self.path, 'unlabeled', 'imagesTr')
        self.unlabeled_data_length = len(os.listdir(self.unlabeled_path)) // debug_ratio

        self.labeled_path_list = [(os.path.join(self.path, 'labeled', 'imagesTr', os.listdir(self.labeled_path)[i]),
                                   os.path.join(self.path, 'labeled', 'labelsTr', ''.join(os.listdir(self.labeled_path)[i].split('_0000'))))
                                  for i in range(self.labeled_data_length)]
        self.labeled_train_name_list = self.labeled_path_list
        self.unlabeled_path_list = [(os.path.join(self.path, 'unlabeled', 'imagesTr', os.listdir(self.unlabeled_path)[i]),
                                   os.path.join(self.path, 'unlabeled', 'labelsTr', ''.join(os.listdir(self.unlabeled_path)[i].split('_0000'))))
                                  for i in range(self.unlabeled_data_length)]
        self.unlabeled_train_name_list = self.unlabeled_path_list
        
        # cut val
        self.labeled_test_name_list = self.labeled_path_list[
                                      fold_idx * (self.labeled_data_length // args.k_fold):(fold_idx + 1) * (
                                              self.labeled_data_length // args.k_fold)]
        self.labeled_train_name_list = list(set(self.labeled_path_list) - set(self.labeled_test_name_list))
        self.unlabeled_test_name_list = self.unlabeled_path_list[
                                      fold_idx * (self.unlabeled_data_length // args.k_fold):(fold_idx + 1) * (
                                                  self.unlabeled_data_length // args.k_fold)]
        self.unlabeled_train_name_list = list(set(self.unlabeled_path_list) - set(self.unlabeled_test_name_list))


        # train list
        # SSL
        self.train_name_list = self.labeled_train_name_list + self.unlabeled_train_name_list
        # SL
        # self.train_name_list = self.labeled_train_name_list
        
        # SSL
        self.test_name_list = self.labeled_test_name_list + self.unlabeled_test_name_list
        # SL
        # self.test_name_list = self.labeled_test_name_list

        # get the u/l rate
        self.u_l_rate = len(self.unlabeled_train_name_list) // len(self.labeled_train_name_list)
        self.labeled_idx = np.arange(0, len(self.labeled_train_name_list))
        self.unlabeled_idx = np.arange(len(self.labeled_train_name_list), len(self.train_name_list))

        logging.info(f"Load done, length of train dataset: {len(self.train_name_list)}")
        logging.info(f"length of labeled dataset in train: {len(self.labeled_train_name_list)}")
        logging.info(f"length of unlabeled dataset in train: {len(self.unlabeled_train_name_list)}")
        
        # logging.info(f"length of validation dataset: {len(self.test_img_path)}")
        logging.info(f"length of validation dataset: {len(self.test_name_list)}")

        print(f"Load done, length of train dataset: {len(self.train_name_list)}")
        print(f"length of labeled dataset in train: {len(self.labeled_train_name_list)}")
        print(f"length of unlabeled dataset in train: {len(self.unlabeled_train_name_list)}")
        print("-------------------------------------------")
        # print(f"length of validation dataset: {len(self.test_img_path)}")
        print(f"length of validation dataset: {len(self.test_name_list)}")

    def __len__(self):
        if self.mode == 'train':
            return len(self.train_name_list)
        elif self.mode == 'test':
            # return len(self.test_img_path)
            return len(self.test_name_list)

    def __getitem__(self, idx):
        data_dict = {'img': None,
                     'lab': None,
                     'sdt_lab': None}


        if self.mode == 'train':
            img_fp, lab_fp = self.train_name_list[idx]
            itk_img, itk_lab = nib.load(img_fp).get_fdata(), nib.load(lab_fp).get_fdata()  # h ,w
            img, lab, sdt_lab = cal_SDT(itk_img, itk_lab)

        elif self.mode == 'test':
            # img_fp, lab_fp = self.test_img_path[idx], self.test_lab_list[idx]
            img_fp, lab_fp = self.test_name_list[idx]
            itk_img, itk_lab = nib.load(img_fp).get_fdata(), nib.load(lab_fp).get_fdata()  # h ,w
            img, lab, sdt_lab = get_test(itk_img, itk_lab)

        data_dict['img'] = img
        data_dict['lab'] = lab
        data_dict['sdt_lab'] = sdt_lab
        # data_dict['flag'] = flag


        return data_dict

class SDTDataset_ForTest(Dataset):
    def __init__(self, args, img_name_list, mode='baseline'):
        assert mode in ['baseline', 'our']

        self.mode = mode
        self.args = args

        self.img_list = []
        self.lab_list = []

        logging.info(f"Start loading data")

        self.path = args.data_root

        for name in img_name_list:
            self.img_list.append(os.path.join(self.path, 'imagesTs', name + '_0000.nii.gz'))
            self.lab_list.append(os.path.join(self.path, 'labelsTs', name + '.nii.gz'))

        logging.info(f"Load done, length of dataset: {len(self.img_list)}")

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        idx = idx % len(self.img_list)
        img_fp, lab_fp = self.img_list[idx], self.lab_list[idx]
        itk_img, itk_lab = nib.load(img_fp).get_fdata(), nib.load(lab_fp).get_fdata()  # h ,w
        if self.mode == 'our':
            # shape: frames, c, h, w
            img, lab, sdt_lab = get_test(itk_img, itk_lab)
            data_dict = {'img': img, 'lab': lab, 'sdt_lab': sdt_lab}
        elif self.mode == 'baseline':
            # shape: c, h, w
            img = torch.tensor(array_scale(itk_img.astype(np.float32), 64, mode='img')).unsqueeze(0).float()
            lab = torch.tensor(array_scale(itk_lab.astype(np.float32), 64, mode='gt')).unsqueeze(0).long()
            data_dict = {'img': img, 'lab': lab}

        return data_dict

class SDTDataset_copy(Dataset):
    def __init__(self, args, img_name_list, mode='train'):
        assert mode in ['train', 'test']
        logging.info(f"Start loading {mode} data")

        self.args = args
        self.path = args.data_root
        self.mode = mode

        self.img_list = []
        self.lab_list = []

        if self.mode == 'train':
            for name in img_name_list:
                self.img_list.append(os.path.join(self.path, 'imagesTr', name + '_0000.nii.gz'))
                self.lab_list.append(os.path.join(self.path, 'labelsTr', name + '.nii.gz'))
        elif self.mode == 'test':
            for name in img_name_list:
                self.img_list.append(os.path.join(self.path, 'imagesTr', name + '_0000.nii.gz'))
                self.lab_list.append(os.path.join(self.path, 'labelsTr', name + '.nii.gz'))

        logging.info(f"Load done, length of dataset: {len(self.img_list)}")

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        data_dict = {'img': None,
                     'lab': None,
                     'sdt_lab': None,
                     'edge_lab': None}

        img_fp, lab_fp = self.img_list[idx], self.lab_list[idx]
        itk_img, itk_lab = nib.load(img_fp).get_fdata(), nib.load(lab_fp).get_fdata()  # h ,w

        if self.mode == 'train':
            img, lab, sdt_lab, edge_lab = cal_SDT(itk_img, itk_lab)
        elif self.mode == 'test':
            img, lab, sdt_lab, edge_lab = get_test(itk_img, itk_lab)

        data_dict['img'] = img
        data_dict['lab'] = lab
        data_dict['sdt_lab'] = sdt_lab
        data_dict['edge_lab'] = edge_lab

        return data_dict

class DistributedTwoStreamBatchSampler(DistributedSampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, dataset: Dataset, bs: Optional[int] = None, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (len(self.dataset) - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed

        self.bs = bs


    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        primary_indices = list(set(indices) & set(self.dataset.labeled_idx))
        secondary_indices = list(set(indices) & set(self.dataset.unlabeled_idx))

        # assert indices == primary_indices + secondary_indices

        primary_iter = iterate_once(primary_indices)
        secondary_iter = iterate_eternally(secondary_indices)

        primary_batch_size = self.bs // self.dataset.u_l_rate
        secondary_batch_size = self.bs - primary_batch_size

        self.labeled_batch_size = primary_batch_size
        self.unlabeled_batch_size = secondary_batch_size

        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch) in zip(
                grouper(primary_iter, primary_batch_size),
                grouper(secondary_iter, secondary_batch_size),
            )
        )

    def __len__(self):
        return self.num_samples // self.bs

def iterate_once(iterable):
    return np.random.permutation(iterable)

def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)

    return itertools.chain.from_iterable(infinite_shuffles())

def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)
