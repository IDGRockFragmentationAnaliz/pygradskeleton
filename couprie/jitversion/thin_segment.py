import numpy as np
from numba import njit
from .mctopo.hseparant4 import separant4


def thin_segment(image, threshold=None):
    if threshold is None:
        return separants_raw(image)
    else:
        return separants_threshold(image, threshold)


@njit(cache=True)
def separants_raw(image):
    height, width = image.shape
    borders = np.zeros_like(image, dtype=np.uint8)

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            if separant4(image, y, x):
                borders[y, x] = image[y, x]
    return borders

@njit(cache=True)
def separants_threshold(image, threshold):
    height, width = image.shape
    borders = np.zeros_like(image, dtype=np.uint8)
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            if separant4(image, y, x) and image[y, x] >= threshold:
                borders[y, x] = 255
    return borders