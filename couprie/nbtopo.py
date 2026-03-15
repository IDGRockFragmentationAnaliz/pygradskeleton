import numpy as np
from .topology import get_t8_ones, get_t4_zeros

"""
    3 2 1
    4 X 0
    5 6 7
    
    mm - строго меньше уровня
    m  - меньше или равны уровню
    p  - больше или равны уровню
    pp - строго больше уровню
"""

offsets = [
    ((1, 0), 0), ((1, 1), 1), ((0, 1), 2),
    ((-1, 1), 3), ((-1, 0), 4), ((-1, -1), 5),
    ((0, -1), 6), ((1, -1), 7)
]

def pdestr4(image):
    image_view = build_views3x3(image)
    t4m, t4mm, t8p, t8pp = nbtopo(image)
    destructible = np.zeros_like(image, dtype=np.uint8)
    destructible_center = destructible[1:-1, 1:-1]
    destructible_center[(t4mm == 1) & (t8p == 1)] = 1
    return destructible


def nbtopo(image):
    t4_zeros = get_t4_zeros()
    t8_ones = get_t8_ones()

    bitmask_p = get_bitmask_p(image)
    t4mm = t4_zeros[bitmask_p]
    t8p = t8_ones[bitmask_p]

    bitmask_pp = get_bitmask_pp(image)
    t4m = t4_zeros[bitmask_pp]
    t8pp = t8_ones[bitmask_pp]

    return t4m, t4mm, t8p, t8pp


def get_bitmask_pp(image):
    image_view = build_views3x3(image)
    center = image_view[(0, 0)]
    # pp mask
    packed = np.zeros(center.shape, dtype=np.uint8)
    for offset, bit in offsets:
        result = image_view[offset] > center
        packed |= result.astype(np.uint8) << bit
    return packed

def get_bitmask_p(image):
    image_view = build_views3x3(image)
    center = image_view[(0, 0)]
    # pp mask
    packed = np.zeros(center.shape, dtype=np.uint8)
    for offset, bit in offsets:
        result = image_view[offset] >= center
        packed |= result.astype(np.uint8) << bit
    return packed

def build_views3x3(image):
    offsets_3x3 = tuple(
        (dy, dx)
        for dy in range(-1, 2)
        for dx in range(-1, 2)
    )
    view = {}
    for dy, dx in offsets_3x3:
        y_start = 1 + dy
        y_stop = image.shape[0] - 1 + dy
        x_start = 1 + dx
        x_stop = image.shape[1] - 1 + dx

        view[(dy, dx)] = image[y_start:y_stop, x_start:x_stop]
    return view