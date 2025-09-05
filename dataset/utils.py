import numpy as np
import torch
from torch import Tensor
from scipy.ndimage import distance_transform_edt as eucl_distance
from torch import Tensor
from typing import Any, Callable, Iterable, List, Set, Tuple, TypeVar, Union, cast

# Assert utils
def uniq(a: Tensor) -> Set:
    return set(torch.unique(a.cpu()).numpy())

def sset(a: Tensor, sub: Iterable) -> bool:
    return uniq(a).issubset(sub)

def simplex(t: Tensor, axis=1) -> bool:
    _sum = cast(Tensor, t.sum(axis).type(torch.float32))
    _ones = torch.ones_like(_sum, dtype=torch.float32)
    return torch.allclose(_sum, _ones)

def one_hot(t: Tensor, axis=1) -> bool:
    return simplex(t, axis) and sset(t, [0, 1])

def one_hot2dist(seg: np.ndarray, resolution: Tuple[float, float, float] = None,
                 dtype=None) -> np.ndarray:
    # assert one_hot(torch.tensor(seg), axis=0)
    K: int = len(seg)

    res = np.zeros_like(seg, dtype=dtype)
    for k in range(K):
        posmask = seg[k].astype(np.bool)

        if posmask.any():
            negmask = ~posmask
            res[k] = eucl_distance(negmask, sampling=resolution) * negmask \
                - (eucl_distance(posmask, sampling=resolution) - 1) * posmask
        # The idea is to leave blank the negative classes
        # since this is one-hot encoded, another class will supervise that pixel

    return res

def one_hot2dist_Neg(seg: np.ndarray, resolution: Tuple[float, float, float] = None,
                 dtype=None) -> np.ndarray:
    # assert one_hot(torch.tensor(seg), axis=0)
    K: int = len(seg)

    res = np.zeros_like(seg, dtype=dtype)
    for k in range(K):
        posmask = seg[k].astype(np.bool)
        negmask = ~posmask
        if posmask.any():
            res[k] = eucl_distance(negmask, sampling=resolution) * negmask
        # The idea is to leave blank the negative classes
        # since this is one-hot encoded, another class will supervise that pixel

    return res

def class2one_hot(seg: Tensor, K: int) -> Tensor:
    # Breaking change but otherwise can't deal with both 2d and 3d
    # if len(seg.shape) == 3:  # Only w, h, d, used by the dataloader
    #     return class2one_hot(seg.unsqueeze(dim=0), K)[0]
    # assert sset(seg, list(range(K))), (uniq(seg), K)

    b, *img_shape = seg.shape  # type: Tuple[int, ...]

    device = seg.device

    res = torch.zeros((b, K, *img_shape), dtype=torch.int32, device=device)
    for i in range(K):
        res[:, i, ...] = (seg == i).to(torch.int32)

    # res = torch.zeros((b, K, *img_shape), dtype=torch.int32, device=device).scatter_(1, seg[:, None, ...], 1)

    # assert res.shape == (b, K, *img_shape)
    # assert one_hot(res)

    return res

def one_hot_to_class(one_hot):
    _mask = np.zeros((one_hot.shape[1], one_hot.shape[-1]))
    for k in range(one_hot.shape[0]):
        if k == 0:
            continue
        _mask += np.where(one_hot[k] == 1, k, 0)
    _mask = np.where(_mask >= one_hot.shape[0] - 1, one_hot.shape[0] - 1, _mask)
    return _mask
