import numpy as np
import numba as nb
from numba import njit, prange
import matplotlib.pyplot as plt
import time

from numba.np.ufunc import parallel

from .matcher import Matcher
from .alpha_builder import AlphaBuilder
from .jitversion.mctopo.pdestr4 import pdestr4_all, pdestr4_flat
from .jitversion.mctopo.alpha8m import alpha8m_flat
from .jitflatversion.match_crutial.match_crutial import match_c
from .jitversion.voisin import voisin_flat

def lhthinpar(image, copy=True):
    if copy:
        image = image.copy()

    h, w = image.shape
    destructible = np.ones((h + 2, w + 2), dtype=np.uint8)
    set_edge_border_zeros(destructible)
    alpha = np.empty((h + 2, w + 2), dtype=np.uint8)
    bitmask = np.empty((h + 2, w + 2), dtype=np.uint8)
    image_padded = np.pad(image, 1, mode='edge')
    image_padded_flat = image_padded.ravel()
    destructible_flat = destructible.ravel()
    bitmask_flat = bitmask.ravel()

    idx = np.flatnonzero(destructible)

    alpha_flat = alpha.ravel()
    w_pad = w + 2
    h_pad = h + 2
    t_pad_total = 0.0
    t_pdestr4 = 0.0
    t_flatnonzero = 0.0
    t_alpha8m = 0.0
    t_match_c = 0.0
    t_end = 0.0
    t_flatnonzero2 = 0.0
    t_mask = 0.0
    for i in range(10000):
        #print(i)
        t0 = time.perf_counter()
        update_edge_border_pad1(image_padded)
        t_pad_total += time.perf_counter() - t0

        t0 = time.perf_counter()
        destructible = pdestr4_center(image_padded, destructible, bitmask, idx)
        t_pdestr4 += time.perf_counter() - t0

        t0 = time.perf_counter()
        destidx = flatones_idx(destructible, idx)
        t_flatnonzero += time.perf_counter() - t0

        t0 = time.perf_counter()
        alpha = alpha8m_center(image_padded, alpha, destidx, w_pad)
        t_alpha8m += time.perf_counter() - t0

        t0 = time.perf_counter()
        match_c_center(image_padded_flat, destructible_flat, alpha_flat, bitmask_flat, destidx, w_pad)
        t_match_c += time.perf_counter() - t0

        t0 = time.perf_counter()
        idx = flatones_idx(destructible, idx)
        t_flatnonzero2 += time.perf_counter() - t0

        t0 = time.perf_counter()
        destructible_kernel_update(destructible_flat, idx, h_pad, w_pad)
        t_mask += time.perf_counter() - t0

        print("count:", len(idx))
        if idx.size == 0:
            print("total loop", i)
            break

        t0 = time.perf_counter()
        image_padded.flat[idx] = alpha.flat[idx]
        idx = np.flatnonzero(destructible == 1)
        t_end += time.perf_counter() - t0


    print(f"padding total: {t_pad_total:.6f} sec")
    print(f"pdestr4 total: {t_pdestr4:.6f} sec")
    print(f"flatnonzero total: {t_flatnonzero:.6f} sec")
    print(f"alpha8m total: {t_alpha8m:.6f} sec")
    print(f"match_c total: {t_match_c:.6f} sec")
    print(f"flatnonzero2 total: {t_flatnonzero2:.6f} sec")
    print(f"t_mask total: {t_mask:.6f} sec")
    print(f"t_end total: {t_end:.6f} sec")
    return image_padded[1:-1,1:-1]

@njit(cache=True)
def flatones_idx(destructible, idx):
    out = np.empty(idx.size, dtype=idx.dtype)
    n = 0
    for i in range(idx.size):
        p = idx[i]
        if destructible.flat[p] == 1:
            out[n] = p
            n += 1
    return out[:n]

@njit(cache=True)
def destructible_kernel_update(destructible, idx, h, w):
    d = destructible.flat

    hm1 = h - 1
    wm1 = w - 1
    hm2 = h - 2
    wm2 = w - 2

    for i in range(idx.size):
        p = idx[i]

        x = p // w
        y = p - x * w

        # Быстрый путь:
        # все 8 соседей гарантированно не попадают во внешнюю границу.
        if 1 < x < hm2 and 1 < y < wm2:
            d[np.uintp(p + 1)]     = 1
            d[np.uintp(p - w + 1)] = 1
            d[np.uintp(p - w)]     = 1
            d[np.uintp(p - w - 1)] = 1
            d[np.uintp(p - 1)]     = 1
            d[np.uintp(p + w - 1)] = 1
            d[np.uintp(p + w)]     = 1
            d[np.uintp(p + w + 1)] = 1
        else:
            # Медленный путь только для точек около границы.

            # right
            if 0 < x < hm1 and y < wm2:
                d[np.uintp(p + 1)] = 1

            # top-right
            if x > 1 and y < wm2:
                d[np.uintp(p - w + 1)] = 1

            # top
            if x > 1 and 0 < y < wm1:
                d[np.uintp(p - w)] = 1


@njit(cache=True, parallel=True)
def pdestr4_center(image, destructible, bitmask, idx):
    h, w = image.shape
    sparse = np.empty(idx.size, dtype=idx.dtype)
    k = 0
    for i in prange(idx.size):
        p = idx[i]
        destructible.flat[p] = pdestr4_flat(image.flat, bitmask.flat, p, w)
    return destructible

@njit(cache=True, parallel=True)
def alpha8m_center(image, alpha, destridx, w):
    for i in prange(destridx.size):
        idx = destridx[i]
        alpha.flat[idx] = alpha8m_flat(image.flat, idx, w)
    return alpha

@njit(cache=True)
def match_c_center(image, destructible, alpha, bitmask, destridx, w):
    image = image.flat
    destructible = destructible.flat
    alpha = alpha.flat
    bitmask = bitmask.flat

    for i in range(destridx.size):
        p = destridx[i]
        match_c(image, destructible, alpha, bitmask, p, w)


def set_edge_border_zeros(p):
    # сначала боковые стороны, только центральная часть
    p[1:-1, 0] = 0
    p[1:-1, -1] = 0

    # потом верх/низ целиком, включая углы
    p[0, :] = 0
    p[-1, :] = 0

def update_edge_border_pad1(p):
    # сначала боковые стороны, только центральная часть
    p[1:-1, 0] = p[1:-1, 1]
    p[1:-1, -1] = p[1:-1, -2]

    # потом верх/низ целиком, включая углы
    p[0, :] = p[1, :]
    p[-1, :] = p[-2, :]


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