import numpy as np
from numba import njit
from ...jitversion import T4_ZEROS, T8_ONES
from .bitmasks import bitmask_p_flat

@njit(cache=True, inline="always")
def pdestr4(image, bitmasks, p, w):
    bitmask = bitmask_p_flat(image, p, w)
    t4mm = T4_ZEROS[bitmask]
    t8p = T8_ONES[bitmask]
    bitmasks[p] = bitmask
    if t4mm == 1 and t8p == 1:
        return True
    return False