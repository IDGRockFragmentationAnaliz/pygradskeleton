import numpy as np
from numba import njit
from .mctopo.hseparant4 import separant4

@njit
def thin_segment(image):
    height, width = image.shape
    borders = np.zeros_like(image, dtype=np.uint8)

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            if separant4(image, y, x):
                borders[y, x] = 255

    return borders