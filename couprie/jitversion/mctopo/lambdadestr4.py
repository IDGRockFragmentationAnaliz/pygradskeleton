from numba import njit
import numpy as np
from .nbtopo import nbtopo

@njit
def lambdadestr4(image, y, x, lam):
    t4m, t4mm, t8p, t8pp = nbtopo(image, y, x)
    if t4mm == 1 and t8p == 1:
        return True
    return False