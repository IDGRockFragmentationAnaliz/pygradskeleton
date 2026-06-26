from .skelpar import lhthinpar, lhthinpar_asymmetric
from .llambdakern import llambdakern
from .jitversion.thin_segment import thin_segment


def couprie(image, lam=20, copy=True, progress=False):
    if copy:
        image = image.copy()
    image = lhthinpar(image, copy=False, progress=progress)
    image = lhthinpar_asymmetric(image, copy=False, progress=progress)
    image = llambdakern(image, lam, copy=False, progress=progress)
    if progress:
        print("thin_segment: started")
    borders = thin_segment(image)
    if progress:
        print("couprie: ended")
    return borders


