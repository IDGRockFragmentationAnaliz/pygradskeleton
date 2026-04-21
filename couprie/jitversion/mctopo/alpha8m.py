from numba import njit
import numpy as np


@njit
def alpha8m(image, y, x):
    val = image[y, x]
    found = False
    alpha = np.uint8(0)

    v = image[y, x + 1]
    if v < val:
        alpha = v
        found = True

    v = image[y - 1, x + 1]
    if v < val and (not found or v > alpha):
        alpha = v
        found = True

    v = image[y - 1, x]
    if v < val and (not found or v > alpha):
        alpha = v
        found = True

    v = image[y - 1, x - 1]
    if v < val and (not found or v > alpha):
        alpha = v
        found = True

    v = image[y, x - 1]
    if v < val and (not found or v > alpha):
        alpha = v
        found = True

    v = image[y + 1, x - 1]
    if v < val and (not found or v > alpha):
        alpha = v
        found = True

    v = image[y + 1, x]
    if v < val and (not found or v > alpha):
        alpha = v
        found = True

    v = image[y + 1, x + 1]
    if v < val and (not found or v > alpha):
        alpha = v
        found = True

    return alpha if found else val