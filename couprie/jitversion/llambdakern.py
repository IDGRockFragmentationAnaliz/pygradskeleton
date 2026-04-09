import numpy as np
from numba import njit
from .abaisse4 import abaisse4

@njit
def llambdakern(image, lam):
    height, width = image.shape
    marks = np.ones_like(image)

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            abaisse4(image, y, x, lam)
    return image