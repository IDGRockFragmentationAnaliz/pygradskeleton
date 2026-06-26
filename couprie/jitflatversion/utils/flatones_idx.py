import numpy as np
from numba import njit

@njit(cache=True)
def flatones_idx(destructible, idx):
    out = np.empty(idx.size, dtype=idx.dtype)
    n = 0
    for i in range(idx.size):
        p = idx[i]
        if destructible.flat[p] == 1:
            out[n] = p
            n += 1
    return out[:n]