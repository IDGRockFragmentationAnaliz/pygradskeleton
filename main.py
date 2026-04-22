import numpy as np
from .kim import Kim
from couprie.couprie_skeletonize import couprie


methods = {
    "KIM": Kim,
    "COUPRIE": couprie
}


def grayscale_skeletonize(image: np.array, method="COUPRIE", *args, **kwargs) -> np.array:
    return methods[method](image, *args, **kwargs)