import numpy as np
from numba import njit
from .nbtopo import nbtopo, T4_ZEROS, T8_ONES
from .bitmask import bitmask_p_flat

@njit(cache=True)
def pdestr4_all(image):
    h, w = image.shape
    destructible = np.zeros_like(image, dtype=np.uint8)

    for y in range(1, h - 1):
        for x in range(1, w - 1):
            t4m, t4mm, t8p, t8pp = nbtopo(image, y, x)
            if t4mm == 1 and t8p == 1:
                destructible[y, x] = 1

    return destructible

@njit(cache=True)
def pdestr4(image, y, x):
    t4m, t4mm, t8p, t8pp = nbtopo(image, y, x)
    if t4mm == 1 and t8p == 1:
        return True
    return False

@njit(cache=True, inline="always")
def pdestr4_flat(image, p, w):
    bitmask = bitmask_p_flat(image, p, w)
    t4mm = T4_ZEROS[bitmask]
    t8p = T8_ONES[bitmask]
    if t4mm == 1 and t8p == 1:
        return True
    return False
