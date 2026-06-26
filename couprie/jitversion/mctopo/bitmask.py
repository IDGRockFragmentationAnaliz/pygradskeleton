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
    center = image_flat[idx]
    mask = np.uint8(0)

    # m1: right
    if image_flat[np.uintp(idx + 1)] >= center:
        mask |= M1

    # m2: up-right
    if image_flat[np.uintp(idx - w + 1)] >= center:
        mask |= M2

    # m3: up
    if image_flat[np.uintp(idx - w)] >= center:
        mask |= M3

    # m4: up-left
    if image_flat[np.uintp(idx - w - 1)] >= center:
        mask |= M4

    # m5: left
    if image_flat[np.uintp(idx - 1)] >= center:
        mask |= M5

    # m6: down-left
    if image_flat[np.uintp(idx + w - 1)] >= center:
        mask |= M6

    # m7: down
    if image_flat[np.uintp(idx + w)] >= center:
        mask |= M7

    # m8: down-right
    if image_flat[idx + w + 1] >= center: mask |= M8

    return mask