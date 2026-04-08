import numpy as np

COND_TRUE = 1

def crestrestore(image):
    mask = np.full_like(image, COND_TRUE,  dtype=np.uint8)
