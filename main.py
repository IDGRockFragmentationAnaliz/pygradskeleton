import numpy as np
from kim import Kim


methods = {
       "KIM": Kim
}


def grayscaleskelet(image: np.array, method="KIM", *args, **kwargs) -> np.array:
    return methods[method](image.copy(), *args, **kwargs).run()

