import cv2
import time

import numpy as np

from .functions import flip
from .functions import makeequalmatrix
from .functions import binmatrix
from .functions import lowneighbour
from .functions import endpointmodified
from .functions import oneobject
from .functions import forbidden
from .functions import simpleafterremove

class Couprie:
    def __init__(self, _img):
        self.img = _img.copy()
        self.img2 = _img.copy()
        self.helper = _img.copy()
        self.lowest = _img.copy()


    def run(self):
        start_time = time.time()

        # Reading in the pictures as a gray picture

        img = self.img
        img2 = self.img2
        helper = self.helper
        lowest = self.lowest

        # Converting values 0-255
        img = 255 - img
        img2 = 255 - img2
        helper = 255 - helper
        lowest = 255 - lowest

        # Initialization
        lepes = 0
        size = img.shape
        n = size[0]
        m = size[1]

        while True:
            print(1)
            border = np.zeros((n, m), dtype=np.uint8)
            lowest = np.zeros((n, m), dtype=np.uint8)
            print(2)

            for row in range(2, size[0] - 2):
                for col in range(2, size[1] - 2):
                    if img[row][col] == 0:
                        continue
                    if borderpoint8(img, row, col):
                        lowest[row][col] = lowneighbour(img, row, col)
                        border[row][col] = 1

            print(3)
            for row in range(2, size[0] - 2):
                for col in range(2, size[1] - 2):
                    binmatrixhelper = binmatrix(img, row, col, size)
                    if border[row][col] == 0:
                        continue
                    if endpointmodified(binmatrixhelper, 2, 2):
                        continue
                    if not oneobject(binmatrixhelper, 2, 2) <= 1:
                        continue
                    if not simpleafterremove(binmatrixhelper, 2, 2):
                        continue
                    if forbidden(binmatrixhelper, 2, 2):
                        continue
                    helper[row][col] = 1
            print(4)
            img[helper == 1] = lowest[helper == 1]
            makeequalmatrix(helper, img, size)
            print(5)
            lepes += 1
            if np.array_equal(img, img2):
                break
            else:
                np.copyto(img2, img)

        flip(img)

        print("My program took", time.time() - start_time, "to run")
        return img



def borderpoint8(img, row, col):
    if (
        img[row + 1][col] == 0
        or img[row + 1][col + 1] == 0
        or img[row][col + 1] == 0
        or img[row - 1][col + 1] == 0
        or img[row - 1][col] == 0
        or img[row - 1][col - 1] == 0
        or img[row][col - 1] == 0
        or img[row + 1][col - 1] == 0
    ):
        return True
    return False