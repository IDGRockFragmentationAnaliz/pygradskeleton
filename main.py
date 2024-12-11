import numpy as np
from .kim import Kim


methods = {
       "KIM": Kim
}


def grayscale_skeletonize(image: np.array, method="KIM", *args, **kwargs) -> np.array:
    return methods[method](image, *args, **kwargs).run()

