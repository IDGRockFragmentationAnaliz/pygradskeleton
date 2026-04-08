from numba import njit
import numpy as np
from .hseparant4 import hseparant4, separant4
from .voisin import voisin

@njit
def extensible4(image, y, x):
    if not separant4(image, y, x):
        return 0

    nivext = 0
    for k in range(8):
        x2, y2 = voisin(y, x, k)
        if image[y2, x2] > image[y, x]:
            if image[y2, x2] > nivext:
                nivext = image[y2, x2]
    return nivext

