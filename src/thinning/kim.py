import cv2
import numpy as np
from src.common.functions import countf, makeequalmatrix
from src.thinning.interface.algorithm_interface import IAlgorithm
import torch


class Kim(IAlgorithm):
    h = 50

    def __init__(self, image):
        self.img = image.copy()
        self.img2 = image.copy()
        self.compstar = image.copy()
        self.c8 = image.copy()
        self.E = image.copy()
        self.R = image.copy()
        self.O1 = image.copy()
        self.O2 = image.copy()
        self.helper1 = image.copy()
        self.helper2 = image.copy()
        self.kernel = np.ones((3, 3), np.uint8)
        self.kernel2 = np.ones((5, 5), np.uint8)

    def initialize(self):
        for rowIndex in range(0, self.img.shape[0]):
            for colIndex in range(0, self.img.shape[1]):
                self.compstar[rowIndex][colIndex] = 0
                self.c8[rowIndex][colIndex] = 0
                self.E[rowIndex][colIndex] = 0
                self.R[rowIndex][colIndex] = 0
                self.O1[rowIndex][colIndex] = 0
                self.O2[rowIndex][colIndex] = 0
                self.helper1[rowIndex][colIndex] = 0
                self.helper2[rowIndex][colIndex] = 0

        self.kernel[0][0] = 0
        self.kernel[0][2] = 0
        self.kernel[2][0] = 0
        self.kernel[2][2] = 0

        self.kernel2[0][0] = 0
        self.kernel2[0][1] = 0
        self.kernel2[0][3] = 0
        self.kernel2[0][4] = 0
        self.kernel2[1][0] = 0
        self.kernel2[1][4] = 0
        self.kernel2[3][0] = 0
        self.kernel2[3][4] = 0
        self.kernel2[4][0] = 0
        self.kernel2[4][1] = 0
        self.kernel2[4][3] = 0
        self.kernel2[4][4] = 0

    def step(self):  # noqa: C901
        self.E = cv2.erode(self.img, self.kernel, iterations=1)
        self.helper1 = cv2.erode(self.img, self.kernel, iterations=1)
        self.helper2 = cv2.erode(self.img, self.kernel2, iterations=1)
        self.O1 = cv2.dilate(self.helper1, self.kernel, iterations=1)
        self.O2 = cv2.dilate(self.helper2, self.kernel2, iterations=1)

        img_slice = self.img[1:-1, 1:-1].astype(int)
        O1_slice = self.O1[1:-1, 1:-1].astype(int)
        O2_slice = self.O2[1:-1, 1:-1].astype(int)
        E_slice = self.E[1:-1, 1:-1].astype(int)

        mask = (img_slice - O1_slice > 0) & (img_slice - O2_slice > Kim.h)
        self.R[1:-1, 1:-1] = np.where(mask, img_slice, 0)

        self.compstar[1:-1, 1:-1] = np.maximum(E_slice, self.R[1:-1, 1:-1].astype(int))
        self.c8[1:-1, 1:-1] = countf(self.img)
        mask_c8 = self.c8[1:-1, 1:-1] >= 2
        self.compstar[1:-1, 1:-1][mask_c8] = self.img[1:-1, 1:-1][mask_c8]

        self.img = self.compstar.copy()

    def clear_helpers(self):
        pass

    def after_processing(self):
        for rowIndex in range(0, self.img.shape[0]):
            for colIndex in range(0, self.img.shape[1]):
                self.R[rowIndex][colIndex] = 0
                self.O1[rowIndex][colIndex] = 0
                self.O2[rowIndex][colIndex] = 0
                self.helper1[rowIndex][colIndex] = 0
                self.helper2[rowIndex][colIndex] = 0

        helper1 = cv2.erode(self.img, self.kernel, iterations=1)
        helper2 = cv2.erode(self.img, self.kernel2, iterations=1)
        self.O1 = cv2.dilate(helper1, self.kernel, iterations=1)
        self.O2 = cv2.dilate(helper2, self.kernel2, iterations=1)

        for rowIndex in range(1, self.img.shape[0] - 1):
            for colIndex in range(1, self.img.shape[1] - 1):
                if (
                    int(self.img[rowIndex][colIndex]) - int(self.O1[rowIndex][colIndex])
                    > 0
                ) and int(self.img[rowIndex][colIndex]) - int(
                    self.O2[rowIndex][colIndex]
                ) > Kim.h:
                    self.R[rowIndex][colIndex] = self.img[rowIndex][colIndex]
                else:
                    self.R[rowIndex][colIndex] = 0

    def print_algorithm_name(self):
        print(
            bcolors.OK,
            r"""
          _  ___             _                  _____ _           _
         | |/ (_)           | |                / ____| |         (_)
         | ' / _ _ __ ___   | |     ___  ___  | |    | |__   ___  _
         |  < | | '_ ` _ \  | |    / _ \/ _ \ | |    | '_ \ / _ \| |
         | . \| | | | | | | | |___|  __/  __/ | |____| | | | (_) | |
         |_|\_\_|_| |_| |_| |______\___|\___|  \_____|_| |_|\___/|_|
        """,
            bcolors.ENDC,
        )
