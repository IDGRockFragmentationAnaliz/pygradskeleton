import numpy as np
from numba import njit

@njit(cache=True)
def bitmask_p(image, y, x):
    c = image[y, x]
    mask_p = 0
    if image[y + 1, x    ] >= c: mask_p |= 1 << 0
    if image[y + 1, x + 1] >= c: mask_p |= 1 << 1
    if image[y    , x + 1] >= c: mask_p |= 1 << 2
    if image[y - 1, x + 1] >= c: mask_p |= 1 << 3
    if image[y - 1, x    ] >= c: mask_p |= 1 << 4
    if image[y - 1, x - 1] >= c: mask_p |= 1 << 5
    if image[y    , x - 1] >= c: mask_p |= 1 << 6
    if image[y + 1, x - 1] >= c: mask_p |= 1 << 7
    return mask_p

@njit(cache=True)
def bitmask_pp(image, y, x):
    c = image[y, x]
    mask_p = 0
    if image[y + 1, x    ] > c: mask_p |= 1 << 0
    if image[y + 1, x + 1] > c: mask_p |= 1 << 1
    if image[y    , x + 1] > c: mask_p |= 1 << 2
    if image[y - 1, x + 1] > c: mask_p |= 1 << 3
    if image[y - 1, x    ] > c: mask_p |= 1 << 4
    if image[y - 1, x - 1] > c: mask_p |= 1 << 5
    if image[y    , x - 1] > c: mask_p |= 1 << 6
    if image[y + 1, x - 1] > c: mask_p |= 1 << 7
    return mask_p