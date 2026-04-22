from couprie.matcher import Matcher
from couprie.alpha_builder import AlphaBuilder
from couprie.jitversion.mctopo.pdestr4 import pdestr4_all
import numpy as np

def lhthinpar(image, copy=True):
    if copy:
        image = image.copy()
    n = image.size
    for i in range(1000):
        alpha = AlphaBuilder(image).alpha8m()
        destructible = pdestr4_all(image)
        matcher = Matcher(image, destructible, alpha)
        matcher.match_c()
        #matcher.match_c1()

        mask = destructible == 1
        idx = np.flatnonzero(mask)
        # print((1 - idx.size / n) * 100)
        if idx.size == 0:
            # print(i)
            break
        image.flat[idx] = alpha.flat[idx]
    return image

def lhthinpar_asymmetric(image, copy=True):
    if copy:
        image = image.copy()
    for i in range(1000):
        alpha = AlphaBuilder(image).alpha8m()
        destructible = pdestr4_all(image)
        matcher = Matcher(image, destructible, alpha)
        matcher.match_c_asymmetric()

        mask = destructible == 1
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            # print(i)
            break
        image.flat[idx] = alpha.flat[idx]
    return image