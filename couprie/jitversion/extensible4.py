from numba import njit
import numpy as np
from .mctopo.hseparant4 import hseparant4, separant4
from .voisin import voisin

@njit
def extensible4(image, p_y, p_x):
    if not separant4(image, p_y, p_x):
        return 0
    nivext = 0
    for k in range(8):
        q_x, q_y = voisin(p_y, p_x, k)
        if image[q_y, q_x] > image[p_y, p_x]:
            if image[q_y, q_x] > nivext:
                nivext = image[q_y, q_x]
    return nivext

@njit
def extensible4_all(image):
    height, width = image.shape
    vals = np.zeros_like(image, dtype=np.uint8)

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            vals[y, x] = extensible4(image, y, x)
    return vals