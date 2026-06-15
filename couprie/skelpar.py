import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import time
from .matcher import Matcher
from .alpha_builder import AlphaBuilder
from .jitversion.mctopo.pdestr4 import pdestr4_all, pdestr4, pdestr4_flat
from .jitversion.mctopo.alpha8m import alpha8m, alpha8m_flat
from .jitversion.match_crutial.match_crutical import match_c
from .jitversion.match_crutial.match_crutial_flat import match_c_flat


def lhthinpar(image, copy=True):
    if copy:
        image = image.copy()

    h, w = image.shape
    destructible = np.zeros((h + 2, w + 2), dtype=np.uint8)
    alpha = np.empty((h + 2, w + 2), dtype=np.uint8)
    image_padded = np.pad(image, 1, mode='edge')
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
        destructible = pdestr4_center(image_padded, destructible)
        destidx = np.flatnonzero(destructible)
        t_pdestr4 += time.perf_counter() - t0

        t0 = time.perf_counter()
        alpha = alpha8m_center(image_padded, alpha, destidx, w + 2)
        t_alpha8m += time.perf_counter() - t0

        t0 = time.perf_counter()
        match_c_center(image_padded, destructible, alpha, destidx)
        t_match_c += time.perf_counter() - t0

        t0 = time.perf_counter()
        mask = destructible == 1
        idx = np.flatnonzero(mask)
        t_mask += time.perf_counter() - t0

        print("count:", len(idx))
        if idx.size == 0:
            print("total loop", i)
            break

        t0 = time.perf_counter()
        image_padded.flat[idx] = alpha.flat[idx]
        t_end += time.perf_counter() - t0


    print(f"padding total: {t_pad_total:.6f} sec")
    print(f"pdestr4 total: {t_pdestr4:.6f} sec")
    print(f"alpha8m total: {t_alpha8m:.6f} sec")
    print(f"match_c total: {t_match_c:.6f} sec")
    print(f"t_mask total: {t_mask:.6f} sec")
    print(f"t_end total: {t_end:.6f} sec")
    return image_padded[1:-1,1:-1]


@njit(cache=True)
def pdestr4_center(image, destructible):
    h, w = image.shape
    for y in range(1, h - 1):
        row = y * w
        for x in range(1, w - 1):
            idx = row + x
            destructible.flat[idx] = pdestr4_flat(image.flat, idx, w)
    return destructible


@njit(cache=True)
def alpha8m_center(image, alpha, destridx, w):
    for i in range(destridx.size):
        idx = destridx[i]
        alpha.flat[idx] = alpha8m_flat(image.flat, idx, w)
    return alpha

@njit(cache=True)
def match_c_center(image, destructible, alpha, destridx):
    w = image.shape[1]
    for i in range(destridx.size):
        flat_idx = destridx[i]
        match_c_flat(image.flat, destructible.flat, alpha.flat, flat_idx, w)


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