from numba import njit

shift_y = (1,  1,  0, -1, -1, -1,  0,  1)
shift_x = (0, -1, -1, -1,  0,  1,  1,  1)

@njit
def voisin(y, x, k):
    return y + shift_y[k], x + shift_x[k]