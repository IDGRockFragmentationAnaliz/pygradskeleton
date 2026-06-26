import numpy as np
import numba as nb
from numba import njit, prange
import matplotlib.pyplot as plt
import time

def set_edge_zeros(p):
    # сначала боковые стороны, только центральная часть
    p[1:-1, 0] = 0
    p[1:-1, -1] = 0

    # потом верх/низ целиком, включая углы
    p[0, :] = 0
    p[-1, :] = 0


def set_edge_pad1(p):
    # сначала боковые стороны, только центральная часть
    p[1:-1, 0] = p[1:-1, 1]
    p[1:-1, -1] = p[1:-1, -2]

    # потом верх/низ целиком, включая углы
    p[0, :] = p[1, :]
    p[-1, :] = p[-2, :]