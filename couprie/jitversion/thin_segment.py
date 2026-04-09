import numpy as np
from numba import njit
from .extensible4 import extensible4_all

def thin_segment(image):
    return extensible4_all(image)