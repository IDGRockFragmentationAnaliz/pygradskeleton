from numba import njit
import numpy as np


@njit(cache=True, inline='always')
def alpha8m(image_flat, idx, w):
    val = image_flat[idx]
    best = val

    v = image_flat[idx + 1]
    if v < val: best = v

    v = image_flat[idx - w + 1]
    if v < val and (best == val or v > best): best = v

    v = image_flat[idx - w]
    if v < val and (best == val or v > best): best = v

    v = image_flat[idx - w - 1]
    if v < val and (best == val or v > best): best = v

    v = image_flat[idx - 1]
    if v < val and (best == val or v > best): best = v

    v = image_flat[idx + w - 1]
    if v < val and (best == val or v > best): best = v

    v = image_flat[idx + w]
    if v < val and (best == val or v > best): best = v

    v = image_flat[idx + w + 1]
    if v < val and (best == val or v > best): best = v

    return best