from numba import njit
import numpy as np
from .nbtopoh import nbtopoh
from .nbtopo import nbtopo
from ..voisin import voisin

@njit
def separant4(image: np.ndarray, y, x):
    t4m, t4mm, t8p, t8pp = nbtopo(image, y, x)
    if t4mm >= 2:
        return True
    for k in range(8):
        y2, x2 = voisin(y, x, k)
        if image[y2, x2] < image[y, x]:
            t4m, t4mm, t8p, t8pp = nbtopoh(image, y, x, image[y2, x2])
            if t4mm >= 2:
                return True
    return False


@njit
def hseparant4(image: np.ndarray, y, x, h):
    if image[y, x] <= h:
        return False

    t4m, t4mm, t8p, t8pp = nbtopo(image, y, x)
    if t4mm >= 2:
        return True

    return False