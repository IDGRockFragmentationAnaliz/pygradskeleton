import numpy as np
from numba import njit
from .nbtopo import nbtopo


@njit
def alpha8m(image, y, x):
    h, w = image.shape
    val = image[y, x]
    alpha = -1

    v = image[y, x + 1] if x != w - 1 else val
    if val > v > alpha:
        alpha = v

    v = image[y - 1, x + 1] if (x != w - 1 and y != 0) else val
    if val > v > alpha:
        alpha = v

    v = image[y - 1, x] if y != 0 else val
    if val > v > alpha:
        alpha = v

    v = image[y - 1, x - 1] if (y != 0 and x != 0) else val
    if val > v > alpha:
        alpha = v

    v = image[y, x - 1] if x != 0 else val
    if val > v > alpha:
        alpha = v

    v = image[y + 1, x - 1] if (x != 0 and y != h - 1) else val
    if val > v > alpha:
        alpha = v

    v = image[y + 1, x] if y != h - 1 else val
    if val > v > alpha:
        alpha = v

    v = image[y + 1, x + 1] if (y != h - 1 and x != w - 1) else val
    if val > v > alpha:
        alpha = v

    if alpha == -1:
        return image[y, x]
    else:
        return np.uint8(alpha)