from .skelpar import lhthinpar, lhthinpar_asymmetric
from .jitversion.llambdakern import llambdakern
from .jitversion.thin_segment import thin_segment


def couprie(image, lam=20, copy=True):
    if copy:
        image = image.copy()
    image = lhthinpar(image, copy=False)
    image = lhthinpar_asymmetric(image, copy=False)
    image = llambdakern(image, lam, copy=False)
    borders = thin_segment(image)
    return borders


