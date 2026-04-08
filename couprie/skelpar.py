from couprie.matcher import Matcher
from couprie.alpha_builder import AlphaBuilder
from couprie.jitversion.mctopo.pdstr4 import pdestr4
import numpy as np

def lhthinpar(image):
    image = image.copy()
    for i in range(1000):
        print(i)
        alpha = AlphaBuilder(image).alpha8m()
        destructible = pdestr4(image)
        matcher = Matcher(image, destructible, alpha)
        matcher.match_c()
        #matcher.match_c1()

        mask = destructible == 1
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            print(i)
            break
        image.flat[idx] = alpha.flat[idx]
    return image

def lhthinpar_asymmetric(image):
    image = image.copy()
    for i in range(1000):
        alpha = AlphaBuilder(image).alpha8m()
        destructible = pdestr4(image)
        matcher = Matcher(image, destructible, alpha)
        matcher.match_c_asymmetric()

        mask = destructible == 1
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            print(i)
            break
        image.flat[idx] = alpha.flat[idx]
    return image