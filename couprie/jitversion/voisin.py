from numba import njit

#тут немного не верно , надо обратное
shift_y = (1,  1,  0, -1, -1, -1,  0,  1)
shift_x = (0, -1, -1, -1,  0,  1,  1,  1)

@njit
def voisin(y, x, k):
    return y + shift_y[k], x + shift_x[k]


shift_y8 = (0, -1, -1, -1,  0,  1,  1, 1)
shift_x8 = (1,  1,  0, -1, -1, -1,  0, 1)

@njit(cache=True, inline="always")
def voisin_flat(p, k, w):
    return p + shift_y8[k] * w + shift_x8[k]