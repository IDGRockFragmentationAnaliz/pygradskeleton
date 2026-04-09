from numba import njit
import numpy as np
from .nbtopo import nbtopo


@njit
def pconstr4(image, y, x):
    t4m, t4mm, t8p, t8pp = nbtopo(image, y, x)

    if t4m == 1 and t8pp == 1:
        return 1
    else:
        return 0