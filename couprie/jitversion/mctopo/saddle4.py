from numba import njit
from .nbtopo import nbtopo

@njit
def saddle4(image, y, x):
    t4m, t4mm, t8p, t8pp = nbtopo(image, y, x)

    if t8pp > 1 and t4mm > 1:
        return True
    else:
        return False