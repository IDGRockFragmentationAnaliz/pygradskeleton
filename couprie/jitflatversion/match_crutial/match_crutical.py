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

NON_DESTRUCTIBLE = 0
CRUCIAL_C = 9

@njit(cache=True)
def match_c(image, destructible, alpha, y, x):
    bitmask = bitmask_b(image, y, x)
    match_c_right(bitmask, image, destructible, alpha, y, x)
    match_c_left(bitmask, image, destructible, alpha, y, x)
    match_c_up(bitmask, image, destructible, alpha, y, x)
    match_c_down(bitmask, image, destructible, alpha, y, x)


@njit(cache=True)
def match_c_right(bitmask, image, destructible, alpha, y, x):
    """
        at least one of the numbered ones must be
        [-1, 1, 1]
        [-1, 2, 2]
        [-1, 3, 3]
        for forth directions
    """
    if destructible[y, x + 1] == NON_DESTRUCTIBLE or destructible[y, x] == NON_DESTRUCTIBLE:
        return False

    if not (alpha[y, x + 1] < image[y, x]):
        return False

    if (bitmask & M1) == 0:
        return False

    if (bitmask & (M2 | M3)) == 0:
        return False

    if (bitmask & (M7 | M8)) == 0:
        return False

    destructible[y, x] = CRUCIAL_C
    destructible[y, x + 1] = CRUCIAL_C

    return True

@njit(cache=True)
def match_c_up(bitmask, image, destructible, alpha, y, x):
    if not (destructible[y - 1, x] >= 1 and destructible[y, x] >= 1):
        return False

    if not (alpha[y - 1, x] < image[y, x]):
        return False

    # direction: up
    if (bitmask & M3) == 0:
        return False

    # right side of the pair: right or up-right
    if (bitmask & (M1 | M2)) == 0:
        return False

    # left side of the pair: up-left or left
    if (bitmask & (M4 | M5)) == 0:
        return False

    destructible[y, x] = CRUCIAL_C
    destructible[y - 1, x] = CRUCIAL_C

    return True

@njit(cache=True)
def match_c_down(bitmask, image, destructible, alpha, y, x):
    if not (destructible[y + 1, x] >= 1 and destructible[y, x] >= 1):
        return False

    if not (alpha[y + 1, x] < image[y, x]):
        return False

    # direction: down
    if (bitmask & M7) == 0:
        return False

    # right side of the pair: down-right or right
    if (bitmask & (M8 | M1)) == 0:
        return False

    # left side of the pair: left or down-left
    if (bitmask & (M5 | M6)) == 0:
        return False

    destructible[y, x] = CRUCIAL_C
    destructible[y + 1, x] = CRUCIAL_C

    return True

@njit(cache=True)
def match_c_left(bitmask, image, destructible, alpha, y, x):
    if not (destructible[y, x - 1] >= 1 and destructible[y, x] >= 1):
        return False

    if not (alpha[y, x - 1] < image[y, x]):
        return False

    # direction: left
    if (bitmask & M5) == 0:
        return False

    # upper side of the pair: up or up-left
    if (bitmask & (M3 | M4)) == 0:
        return False

    # lower side of the pair: down-left or down
    if (bitmask & (M6 | M7)) == 0:
        return False

    destructible[y, x] = CRUCIAL_C
    destructible[y, x - 1] = CRUCIAL_C

    return True

@njit(cache=True)
def bitmask_b(image, y, x):
    center = image[y, x]
    mask = np.uint8(0)

    # m1 -> bit 0
    if image[y, x + 1] >= center:
        mask |= np.uint8(1 << 0)

    # m2 -> bit 1
    if image[y - 1, x + 1] >= center:
        mask |= np.uint8(1 << 1)

    # m3 -> bit 2
    if image[y - 1, x] >= center:
        mask |= np.uint8(1 << 2)

    # m4 -> bit 3
    if image[y - 1, x - 1] >= center:
        mask |= np.uint8(1 << 3)

    # m5 -> bit 4
    if image[y, x - 1] >= center:
        mask |= np.uint8(1 << 4)

    # m6 -> bit 5
    if image[y + 1, x - 1] >= center:
        mask |= np.uint8(1 << 5)

    # m7 -> bit 6
    if image[y + 1, x] >= center:
        mask |= np.uint8(1 << 6)

    # m8 -> bit 7
    if image[y + 1, x + 1] >= center:
        mask |= np.uint8(1 << 7)

    return mask