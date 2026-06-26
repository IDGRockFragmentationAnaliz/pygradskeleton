from numba import njit
import numpy as np


@njit(cache=True)
def expand_sparce(destructible, idx, h, w):
    d = destructible.flat
    idx_raw = np.empty(destructible.size, idx.dtype)
    hm2 = h - 2
    wm2 = w - 2

    # сначала копируем старые idx
    n = idx.size
    for i in range(idx.size):
        idx_raw[i] = idx[i]

    # потом дописываем новые
    for i in range(idx.size):
        p = idx[i]

        x = p // w
        y = p - x * w

        # right
        if y < wm2:
            q = p + 1
            if d[q] == 0:
                d[q] = 1
                idx_raw[n] = q
                n += 1

        # top-right
        if x > 1 and y < wm2:
            q = p - w + 1
            if d[q] == 0:
                d[q] = 1
                idx_raw[n] = q
                n += 1

        # top
        if x > 1:
            q = p - w
            if d[q] == 0:
                d[q] = 1
                idx_raw[n] = q
                n += 1

        # top-left
        if x > 1 and y > 1:
            q = p - w - 1
            if d[q] == 0:
                d[q] = 1
                idx_raw[n] = q
                n += 1

        # left
        if y > 1:
            q = p - 1
            if d[q] == 0:
                d[q] = 1
                idx_raw[n] = q
                n += 1

        # bottom-left
        if x < hm2 and y > 1:
            q = p + w - 1
            if d[q] == 0:
                d[q] = 1
                idx_raw[n] = q
                n += 1

        # bottom
        if x < hm2:
            q = p + w
            if d[q] == 0:
                d[q] = 1
                idx_raw[n] = q
                n += 1

        # bottom-right
        if x < hm2 and y < wm2:
            q = p + w + 1
            if d[q] == 0:
                d[q] = 1
                idx_raw[n] = q
                n += 1

    return idx_raw[:n]



@njit(cache=True, inline='always')
def _expand_sparce_fast(d, idx_raw, n, p, w):
    # Без проверок границ.
    # Вызывать только если все 8 соседей гарантированно внутри рабочей области.

    q = p + 1
    if d[q] == 0:
        d[q] = 1
        idx_raw[n] = q
        n += 1

    q = p - w + 1
    if d[q] == 0:
        d[q] = 1
        idx_raw[n] = q
        n += 1

    q = p - w
    if d[q] == 0:
        d[q] = 1
        idx_raw[n] = q
        n += 1

    q = p - w - 1
    if d[q] == 0:
        d[q] = 1
        idx_raw[n] = q
        n += 1

    q = p - 1
    if d[q] == 0:
        d[q] = 1
        idx_raw[n] = q
        n += 1

    q = p + w - 1
    if d[q] == 0:
        d[q] = 1
        idx_raw[n] = q
        n += 1

    q = p + w
    if d[q] == 0:
        d[q] = 1
        idx_raw[n] = q
        n += 1

    q = p + w + 1
    if d[q] == 0:
        d[q] = 1
        idx_raw[n] = q
        n += 1

    return n