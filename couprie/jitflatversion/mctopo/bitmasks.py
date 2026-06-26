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