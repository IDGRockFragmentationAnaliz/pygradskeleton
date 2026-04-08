from numba import njit
import numpy as np
from .mctopo.hseparant4 import hseparant4, separant4
from .voisin import voisin

@njit
def extensible4(image, y, x):
    if not separant4(image, y, x):
        return 1
    nivext = 0
    for k in range(8):
        x2, y2 = voisin(y, x, k)
        if image[y2, x2] > image[y, x]:
            if image[y2, x2] > nivext:
                nivext = image[y2, x2]
    return 0

@njit
def extensible4_all(image):
    height, width = image.shape
    vals = np.zeros_like(image, dtype=np.uint8)

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            vals[y, x] = extensible4(image, y, x)
    return vals