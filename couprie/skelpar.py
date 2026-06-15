import numpy as np
import numba as nb
from numba import njit, prange
import matplotlib.pyplot as plt
import time

from numba.np.ufunc import parallel

from .matcher import Matcher
from .alpha_builder import AlphaBuilder
from .jitversion.mctopo.pdestr4 import pdestr4_all, pdestr4, pdestr4_flat
from .jitversion.mctopo.alpha8m import alpha8m, alpha8m_flat
from .jitversion.match_crutial.match_crutical import match_c
from .jitversion.match_crutial.match_crutial_flat import match_c_flat
from .jitversion.voisin import voisin_flat

def lhthinpar(image, copy=True):
    if copy:
        image = image.copy()

    h, w = image.shape
    destructible = np.ones((h + 2, w + 2), dtype=np.uint8)
    update_edge_border_zeros(destructible)
    alpha = np.empty((h + 2, w + 2), dtype=np.uint8)
    bitmask = np.empty((h + 2, w + 2), dtype=np.uint8)
    image_padded = np.pad(image, 1, mode='edge')
    image_padded_flat = image_padded.ravel()
    destructible_flat = destructible.ravel()

    idx = np.flatnonzero(destructible)

    alpha_flat = alpha.ravel()
    w_pad = w + 2
    h_pad = h + 2
    t_pad_total = 0.0
    t_pdestr4 = 0.0
    t_alpha8m = 0.0
    t_match_c = 0.0
    t_end = 0.0
    t_mask = 0.0
    for i in range(10000):
        #print(i)
        t0 = time.perf_counter()
        update_edge_border_pad1(image_padded)
        t_pad_total += time.perf_counter() - t0

        t0 = time.perf_counter()
        destructible = pdestr4_center(image_padded, destructible, bitmask, idx)
        destidx = np.flatnonzero(destructible)
        t_pdestr4 += time.perf_counter() - t0

        t0 = time.perf_counter()
        alpha = alpha8m_center(image_padded, alpha, destidx, w_pad)
        t_alpha8m += time.perf_counter() - t0

        t0 = time.perf_counter()
        match_c_center(image_padded_flat, destructible_flat, alpha_flat, destidx, w_pad)
        t_match_c += time.perf_counter() - t0

        t0 = time.perf_counter()
        idx = np.flatnonzero(destructible == 1)
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
    print(f"alpha8m total: {t_alpha8m:.6f} sec")
    print(f"match_c total: {t_match_c:.6f} sec")
    print(f"t_mask total: {t_mask:.6f} sec")
    print(f"t_end total: {t_end:.6f} sec")
    return image_padded[1:-1,1:-1]


from numba import njit

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


@njit(cache=True)
def pdestr4_center(image, destructible, bitmask, idx):
    h, w = image.shape
    for i in prange(idx.size):
        p = idx[i]
        destructible.flat[p] = pdestr4_flat(image.flat, p, w)
    return destructible

@njit(cache=True, parallel=True)
def alpha8m_center(image, alpha, destridx, w):
    for i in prange(destridx.size):
        idx = destridx[i]
        alpha.flat[idx] = alpha8m_flat(image.flat, idx, w)
    return alpha

@njit(cache=True)
def match_c_center(image, destructible, alpha, destridx, w):
    image = image.flat
    destructible = destructible.flat
    alpha = alpha.flat

    for i in range(destridx.size):
        flat_idx = destridx[i]
        match_c_flat(image, destructible, alpha, flat_idx, w)


def update_edge_border_zeros(p):
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