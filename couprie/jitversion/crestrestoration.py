from numba import njit
import numpy as np
from .extensible4 import extensible4
from .mctopo.pconstr4 import pconstr4
from .mctopo.delta4p import delta4p

@njit
def crestrestore(image, copy=True):
    image = image.copy() if copy else image
    height, width = image.shape

    a = np.zeros_like(image)


    for y in range(1, height - 1):
        for x in range(1, width - 1):
            _a = extensible4(image, y, x)
            if _a > 0:
                a[y, x] = _a

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            if a[y, x] > 0:
                if pconstr4(image, y, x) == 1:
                    _d = delta4p(image, y, x)
                    #_a = a[y, x]
                    #image[y, x] = _d
                    # if _d < _a:
                    #     image[y, x] = _d
                    # else:
                    #     image[y, x] = _a
    return image