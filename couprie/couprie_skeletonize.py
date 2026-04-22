from .skelpar import lhthinpar, lhthinpar_asymmetric
from .jitversion.llambdakern import llambdakern
from .jitversion.thin_segment import thin_segment


def couprie(image, lam=20, copy=True, printout=False):
    if copy:
        image = image.copy()
    if printout:
        print("lhthinpar: started")
    image = lhthinpar(image, copy=False)
    if printout:
        print("lhthinpar_asymmetric: started")
    image = lhthinpar_asymmetric(image, copy=False)
    if printout:
        print("llambdakern: started")
    image = llambdakern(image, lam, copy=False)
    if printout:
        print("thin_segment: started")
    borders = thin_segment(image)
    if printout:
        print("couprie: ended")
    return borders


