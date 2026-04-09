import numpy as np
from numba import njit
from .topology import get_t4_zeros, get_t8_ones

T4_ZEROS = get_t4_zeros().astype(np.uint8)
T8_ONES = get_t8_ones().astype(np.uint8)

@njit
def maskmm(image, y, x):
    val = image[y, x]
    mask_mm = 0
    if image[y + 1, x    ] < val: mask_mm |= 1 << 0
    if image[y + 1, x + 1] < val: mask_mm |= 1 << 1
    if image[y    , x + 1] < val: mask_mm |= 1 << 2
    if image[y - 1, x + 1] < val: mask_mm |= 1 << 3
    if image[y - 1, x    ] < val: mask_mm |= 1 << 4
    if image[y - 1, x - 1] < val: mask_mm |= 1 << 5
    if image[y    , x - 1] < val: mask_mm |= 1 << 6
    if image[y + 1, x - 1] < val: mask_mm |= 1 << 7
    return mask_mm

@njit
def nbtopo(image, y, x):
    bitmask = get_bitmask_p(image, y, x)
    t4mm = T4_ZEROS[bitmask]
    t8p = T8_ONES[bitmask]

    bitmask = get_bitmask_pp(image, y, x)
    t4m = T4_ZEROS[bitmask]
    t8pp = T8_ONES[bitmask]

    return t4m, t4mm, t8p, t8pp

@njit
def get_bitmask_p(image, y, x):
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

@njit
def get_bitmask_pp(image, y, x):
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