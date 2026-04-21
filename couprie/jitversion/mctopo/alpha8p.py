from numba import njit
import numpy as np


@njit
def alpha8p(img, p, rs):
    val = img[p]
    found = False
    alpha = np.uint8(0)

    v = img[p + 1]
    if v > val:
        alpha = v
        found = True

    v = img[p + 1 - rs]
    if v > val and (not found or v < alpha):
        alpha = v
        found = True

    v = img[p - rs]
    if v > val and (not found or v < alpha):
        alpha = v
        found = True

    v = img[p - rs - 1]
    if v > val and (not found or v < alpha):
        alpha = v
        found = True

    v = img[p - 1]
    if v > val and (not found or v < alpha):
        alpha = v
        found = True

    v = img[p - 1 + rs]
    if v > val and (not found or v < alpha):
        alpha = v
        found = True

    v = img[p + rs]
    if v > val and (not found or v < alpha):
        alpha = v
        found = True

    v = img[p + rs + 1]
    if v > val and (not found or v < alpha):
        alpha = v
        found = True

    return alpha if found else val