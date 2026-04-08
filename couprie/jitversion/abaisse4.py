from numba import njit
import numpy as np
from .mctopo.hseparant4 import hseparant4, separant4
from .voisin import voisin
from .mctopo.alpha8m import alpha8m
from .mctopo.lambdadestr4 import lambdadestr4

@njit
def abaisse4(image, y, x, lam):
    while lambdadestr4(image, y, x, lam):
        image[y, x] = alpha8m(image, y, x)