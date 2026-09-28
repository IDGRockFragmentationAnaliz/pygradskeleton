from .skelpar import lhthinpar, lhthinpar_asymmetric
from .llambdakern import llambdakern
from .llambdakern_c import llambdakern_c
from .jitversion.crestrestoration import crestrestore
from .jitversion.thin_segment import thin_segment


def couprie(
    image,
    lam=20,
    threshold=128,
    use_crestrestore=False,
    copy=True,
    progress=False,
):
    """Build a grayscale skeleton, optionally restoring crests before lambda filtering."""
    if copy:
        image = image.copy()
    image = lhthinpar(image, copy=False, progress=progress)
    image = lhthinpar_asymmetric(image, copy=False, progress=progress)
    if use_crestrestore:
        image = crestrestore(
            image,
            copy=True,
            n_repeat=-1,
            progress=progress,
        )
    image = llambdakern_c(image, lam, copy=True, progress=progress)
    if progress:
        print("thin_segment: started")
    borders = thin_segment(image, threshold)
    if progress:
        print("couprie: ended")
    return borders


