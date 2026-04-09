import numpy as np
from numba import njit
from .abaisse4 import abaisse4
from .voisin import voisin

@njit
def llambdakern(image, lam, copy = True):
    if copy:
        image = image.copy()

    height, width = image.shape
    n = height * width
    seen = np.zeros(n, np.uint8)

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            seen[y * width + x] = 1
    head = (height - 2) * (width - 2)

    while head > 0:
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                p = y * width + x
                if seen[p] == 1:
                    seen[p] = 0
                    head = head - 1
                    if abaisse4(image, y, x, lam): #main weigh function
                        for k in range(8):
                            qy, qx = voisin(y, x, k)
                            q = qy * width + qx
                            if qy <= 0 or qy >= height - 1 or qx <= 0 or qx >= width - 1:
                                seen[q] = 0
                            elif seen[q] == 0:
                                seen[q] = 1
                                head = head + 1


    return image

