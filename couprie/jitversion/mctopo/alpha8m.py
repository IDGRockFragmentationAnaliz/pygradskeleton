from numba import njit
import numpy as np


@njit(cache=True)
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

@njit(cache=True, inline='always')
def alpha8m_flat(image_flat, idx, w):
    val = image_flat[idx]
    best = val

    v = image_flat[idx + 1]
    if v < val:
        best = v

    v = image_flat[idx - w + 1]
    if v < val and (best == val or v > best):
        best = v

    v = image_flat[idx - w]
    if v < val and (best == val or v > best):
        best = v

    v = image_flat[idx - w - 1]
    if v < val and (best == val or v > best):
        best = v

    v = image_flat[idx - 1]
    if v < val and (best == val or v > best):
        best = v

    v = image_flat[idx + w - 1]
    if v < val and (best == val or v > best):
        best = v

    v = image_flat[idx + w]
    if v < val and (best == val or v > best):
        best = v

    v = image_flat[idx + w + 1]
    if v < val and (best == val or v > best):
        best = v

    return best