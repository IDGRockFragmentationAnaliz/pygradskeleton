import numpy as np
import numba as nb
from numba import njit, prange
import matplotlib.pyplot as plt
import time

from numba.np.ufunc import parallel

from .matcher import Matcher
from .alpha_builder import AlphaBuilder
from .jitflatversion.mctopo.pdestr4 import pdestr4
from .jitflatversion.mctopo.alpha8m import alpha8m
from .jitflatversion.match_crutial import match_c, match_c_asymmetric
from .jitversion.mctopo.pdestr4 import pdestr4_all

from .jitflatversion.utils import expand_sparce, flatones_idx, set_edge_zeros, set_edge_pad1

def lhthinpar(image, copy=True):
    if copy:
        image = image.copy()

    h, w = image.shape
    destructible = np.ones((h + 2, w + 2), dtype=np.uint8)
    set_edge_zeros(destructible)
    alpha = np.empty((h + 2, w + 2), dtype=np.uint8)
    bitmask = np.empty((h + 2, w + 2), dtype=np.uint8)
    image_padded = np.pad(image, 1, mode='edge')
    image_padded_flat = image_padded.ravel()
    destructible_flat = destructible.ravel()
    bitmask_flat = bitmask.ravel()
    alpha_flat = alpha.ravel()

    idx = np.flatnonzero(destructible)
    w_pad = w + 2
    h_pad = h + 2
    t_end = 0.0
    for i in range(10000):
        #Обновление паддинга обновленного фото
        set_edge_pad1(image_padded)
        # Вычисление разрушаемых точек в актуальных точках
        destructible = _pdestr4_center(image_padded, destructible, bitmask, idx)
        # Вычисление актуальных точек
        destidx = flatones_idx(destructible, idx)
        # Вычисление возможного понижения
        alpha = _alpha8m_center(image_padded, alpha, destidx, w_pad)
        # Вычисление критических точек
        _match_c_center(image_padded_flat, destructible_flat, alpha_flat, bitmask_flat, destidx, w_pad)
        # Исключение критиеских точек критических точек
        idx = flatones_idx(destructible, idx)

        print("count:", len(idx))
        if idx.size == 0:
            print("total loop", i)
            break
        #Обновление изображения
        image_padded.flat[idx] = alpha.flat[idx]

        # Расширение диапазона актуальности
        t0 = time.perf_counter()
        #destructible_kernel_update(destructible_flat, idx, h_pad, w_pad)
        #idx = np.flatnonzero(destructible == 1)
        idx = expand_sparce(destructible, idx, h_pad, w_pad)
        t_end += time.perf_counter() - t0
    print(f"t_end total: {t_end:.6f} sec")
    return image_padded[1:-1,1:-1]

def lhthinpar_asymmetric_new(image, copy=True):
    if copy:
        image = image.copy()

    h, w = image.shape
    destructible = np.ones((h + 2, w + 2), dtype=np.uint8)
    set_edge_zeros(destructible)
    alpha = np.empty((h + 2, w + 2), dtype=np.uint8)
    bitmask = np.empty((h + 2, w + 2), dtype=np.uint8)
    image_padded = np.pad(image, 1, mode='edge')
    image_padded_flat = image_padded.ravel()
    destructible_flat = destructible.ravel()
    bitmask_flat = bitmask.ravel()
    alpha_flat = alpha.ravel()

    idx = np.flatnonzero(destructible)
    w_pad = w + 2
    h_pad = h + 2
    t_end = 0.0
    for i in range(10000):
        #Обновление паддинга обновленного фото
        set_edge_pad1(image_padded)
        # Вычисление разрушаемых точек в актуальных точках
        destructible = _pdestr4_center(image_padded, destructible, bitmask, idx)
        # Вычисление актуальных точек
        destidx = flatones_idx(destructible, idx)
        # Вычисление возможного понижения
        alpha = _alpha8m_center(image_padded, alpha, destidx, w_pad)
        # Вычисление критических точек
        _match_c_asymmetric_center(image_padded_flat, destructible_flat, alpha_flat, bitmask_flat, destidx, w_pad)
        # Исключение критиеских точек критических точек
        idx = flatones_idx(destructible, idx)

        print("count:", len(idx))
        if idx.size == 0:
            print("total loop", i)
            break
        #Обновление изображения
        image_padded.flat[idx] = alpha.flat[idx]

        # Расширение диапазона актуальности
        t0 = time.perf_counter()
        #destructible_kernel_update(destructible_flat, idx, h_pad, w_pad)
        #idx = np.flatnonzero(destructible == 1)
        idx = expand_sparce(destructible, idx, h_pad, w_pad)
        t_end += time.perf_counter() - t0
    print(f"t_end total: {t_end:.6f} sec")
    return image_padded[1:-1,1:-1]

@njit(cache=True, parallel=True)
def _pdestr4_center(image, destructible, bitmask, idx):
    h, w = image.shape
    k = 0
    for i in prange(idx.size):
        p = idx[i]
        destructible.flat[p] = pdestr4(image.flat, bitmask.flat, p, w)
    return destructible

@njit(cache=True, parallel=True)
def _alpha8m_center(image, alpha, destridx, w):
    for i in prange(destridx.size):
        idx = destridx[i]
        alpha.flat[idx] = alpha8m(image.flat, idx, w)
    return alpha

@njit(cache=True)
def _match_c_center(image, destructible, alpha, bitmask, destridx, w):
    image = image.flat
    destructible = destructible.flat
    alpha = alpha.flat
    bitmask = bitmask.flat

    for i in range(destridx.size):
        p = destridx[i]
        match_c(image, destructible, alpha, bitmask, p, w)

@njit(cache=True)
def _match_c_asymmetric_center(image, destructible, alpha, bitmask, destridx, w):
    image = image.flat
    destructible = destructible.flat
    alpha = alpha.flat
    bitmask = bitmask.flat

    for i in range(destridx.size):
        p = destridx[i]
        match_c_asymmetric(image, destructible, alpha, bitmask, p, w)

def lhthinpar_asymmetric(image, copy=True):
    if copy:
        image = image.copy()
    for i in range(1000):
        alpha = AlphaBuilder(image).alpha8m()
        destructible = pdestr4_all(image)
        matcher = Matcher(image, destructible, alpha)
        matcher.match_c_asymmetric()

        mask = destructible == 1
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            # print(i)
            break
        image.flat[idx] = alpha.flat[idx]
    return image