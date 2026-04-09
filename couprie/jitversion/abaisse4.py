from numba import njit
import numpy as np
from .mctopo.alpha8m import alpha8m
from .mctopo.lambdadestr4 import lambdadestr4

@njit
def abaisse4(image, y, x, lam):
    mod = False
    while lambdadestr4(image, y, x, lam):
        mod = True
        image[y, x] = alpha8m(image, y, x)
    return mod