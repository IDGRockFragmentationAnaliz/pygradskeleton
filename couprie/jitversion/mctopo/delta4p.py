from numba import njit
import numpy as np
from .pconstr4 import pconstr4
from .alpha8m import alpha8m

@njit
def delta4p(image, y, x):
    _saved = image[y, x]
    while pconstr4(image, y, x):
        image[y, x] = alpha8m(image, y, x)
    res = image[y, x]
    #image[y, x] = _saved
    return res