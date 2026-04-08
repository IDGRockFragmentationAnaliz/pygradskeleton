from numba import njit
import numpy as np
from .nbtopoh import nbtopoh
from .nbtopo import nbtopo

@njit
def separant4(image: np.ndarray, y, x):
    t4m, t4mm, t8p, t8pp = nbtopo(image, y, x)
    if t4mm >= 2:
        return image[y, x]

    return 0

@njit
def hseparant4(image: np.ndarray, y, x, h):
    if image[y, x] <= h:
        return 0

    t4m, t4mm, t8p, t8pp = nbtopo(image, y, x)
    if t4mm >= 2:
        return 1

    return 0