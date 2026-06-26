import numpy as np
import numba as nb
from numba import njit

M1 = np.uint8(1 << 0)
M2 = np.uint8(1 << 1)
M3 = np.uint8(1 << 2)
M4 = np.uint8(1 << 3)
M5 = np.uint8(1 << 4)
M6 = np.uint8(1 << 5)
M7 = np.uint8(1 << 6)
M8 = np.uint8(1 << 7)

NON_DESTRUCTIBLE = np.uint8(0)
CRUCIAL_C = np.uint8(9)


@njit(cache=True, inline="always")
def match_c_flat(image_flat, destructible_flat, alpha_flat, bitmasks, p, w):
    bitmask = bitmasks[p]
    match_c_right_flat(bitmask, image_flat, destructible_flat, alpha_flat, p)
    match_c_left_flat(bitmask, image_flat, destructible_flat, alpha_flat, p)
    match_c_up_flat(bitmask, image_flat, destructible_flat, alpha_flat, p, w)
    match_c_down_flat(bitmask, image_flat, destructible_flat, alpha_flat, p, w)

@njit(cache=True, inline="always")
def match_c_right_flat(bitmask, image_flat, destructible_flat, alpha_flat, idx):
    right = np.uintp(idx + 1)

    if destructible_flat[right] == NON_DESTRUCTIBLE or destructible_flat[idx] == NON_DESTRUCTIBLE:
        return False

    if not (alpha_flat[right] < image_flat[idx]):
        return False

    if (bitmask & M1) == 0:
        return False

    if (bitmask & (M2 | M3)) == 0:
        return False

    if (bitmask & (M7 | M8)) == 0:
        return False

    destructible_flat[idx] = CRUCIAL_C
    destructible_flat[right] = CRUCIAL_C

    return True


@njit(cache=True, inline="always")
def match_c_left_flat(bitmask, image_flat, destructible_flat, alpha_flat, idx):
    left = np.uintp(idx - 1)

    if destructible_flat[left] == NON_DESTRUCTIBLE or destructible_flat[idx] == NON_DESTRUCTIBLE:
        return False

    if not (alpha_flat[left] < image_flat[idx]):
        return False

    if (bitmask & M5) == 0:
        return False

    if (bitmask & (M3 | M4)) == 0:
        return False

    if (bitmask & (M6 | M7)) == 0:
        return False

    destructible_flat[idx] = CRUCIAL_C
    destructible_flat[left] = CRUCIAL_C

    return True


@njit(cache=True, inline="always")
def match_c_up_flat(bitmask, image_flat, destructible_flat, alpha_flat, idx, w):
    up = np.uintp(idx - w)

    if destructible_flat[up] == NON_DESTRUCTIBLE or destructible_flat[idx] == NON_DESTRUCTIBLE:
        return False

    if not (alpha_flat[up] < image_flat[idx]):
        return False

    if (bitmask & M3) == 0:
        return False

    if (bitmask & (M1 | M2)) == 0:
        return False

    if (bitmask & (M4 | M5)) == 0:
        return False

    destructible_flat[idx] = CRUCIAL_C
    destructible_flat[up] = CRUCIAL_C

    return True


@njit(cache=True, inline="always")
def match_c_down_flat(bitmask, image_flat, destructible_flat, alpha_flat, idx, w):
    down = np.uintp(idx + w)

    if destructible_flat[down] == NON_DESTRUCTIBLE or destructible_flat[idx] == NON_DESTRUCTIBLE:
        return False

    if not (alpha_flat[down] < image_flat[idx]):
        return False

    if (bitmask & M7) == 0:
        return False

    if (bitmask & (M8 | M1)) == 0:
        return False

    if (bitmask & (M5 | M6)) == 0:
        return False

    destructible_flat[idx] = CRUCIAL_C
    destructible_flat[down] = CRUCIAL_C

    return True
