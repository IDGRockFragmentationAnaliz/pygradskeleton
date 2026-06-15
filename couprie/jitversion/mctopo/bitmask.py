import numpy as np
from numba import njit

M1 = np.uint8(1 << 0)
M2 = np.uint8(1 << 1)
M3 = np.uint8(1 << 2)
M4 = np.uint8(1 << 3)
M5 = np.uint8(1 << 4)
M6 = np.uint8(1 << 5)
M7 = np.uint8(1 << 6)
M8 = np.uint8(1 << 7)

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

@njit(cache=True, inline="always")
def bitmask_p_flat(image_flat, idx, w):
    c = image_flat[idx]
    mask_p = np.uint8(0)
    if image_flat[np.uintp(idx + w)] >= c: mask_p |= M1
    if image_flat[np.uintp(idx + w + 1)] >= c: mask_p |= M2
    if image_flat[np.uintp(idx + 1)] >= c: mask_p |= M3
    if image_flat[np.uintp(idx - w + 1)] >= c: mask_p |= M4
    if image_flat[np.uintp(idx - w)] >= c: mask_p |= M5
    if image_flat[np.uintp(idx - w - 1)] >= c: mask_p |= M6
    if image_flat[np.uintp(idx - 1)] >= c: mask_p |= M7
    if image_flat[np.uintp(idx + w - 1)] >= c: mask_p |= M8
    return mask_p