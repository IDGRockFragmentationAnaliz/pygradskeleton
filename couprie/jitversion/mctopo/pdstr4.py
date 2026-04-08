import numpy as np
from numba import njit
from .nbtopo import nbtopo

@njit
def pdestr4_all(image):
    h, w = image.shape
    destructible = np.zeros_like(image, dtype=np.uint8)

    for y in range(1, h - 1):
        for x in range(1, w - 1):
            t4m, t4mm, t8p, t8pp = nbtopo(image, y, x)
            if t4mm == 1 and t8p == 1:
                destructible[y, x] = 1

    return destructible
