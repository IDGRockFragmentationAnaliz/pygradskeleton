from numba import njit
import numpy as np
from .nbtopo import nbtopo, maskmm
from .alpha8m import alpha8m
from .topology import get_comp4tab
from ..voisin import voisin

COMP4TAB = get_comp4tab()

@njit
def lambdadestr4(image, y, x, lam):
    t4m, t4mm, t8p, t8pp = nbtopo(image, y, x)
    if t4mm == 1 and t8p == 1:
        return True

    if t4mm == 1 and t8p == 0:
        return image[y, x] - alpha8m(image, y, x) <= lam

    # k-divergent
    if t4mm >= 2:
        n = 0
        m = maskmm(image, y, x)
        center = int(image[y, x])

        for i in range(t4mm):
            mi = COMP4TAB[m][i]
            if mi == 0:
                break
            # нету
            ok = True
            #
            for k in range(8):
                if (mi & 1) != 0:
                    qy, qx = voisin(y, x, k)
                    if (center - int(image[qy, qx])) > lam:
                        ok = False
                        break
                mi = mi >> 1

            if ok:
                n += 1

        if n >= (t4mm - 1):
            return True

    return False