import cv2
import numpy as np
from .tools import countf


class Kim:
    def __init__(self, image, steps=10, h=20):
        self.image = image.copy()
        self.h = h
        self.steps = steps
        shape = self.image.shape
        self.compstar = np.zeros(shape)
        self.c8 = np.zeros(shape)
        self.E = np.zeros(shape)
        self.result = np.zeros(shape)

        self.kernel = np.array([
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0]
        ], np.uint8)

        self.kernel2 = np.array([
            [0, 0, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 0, 0]
        ], dtype=np.uint8)

    def step(self):  # noqa: C901
        helper1 = cv2.erode(self.image, self.kernel, iterations=1)
        helper2 = cv2.erode(self.image, self.kernel2, iterations=1)
        e_inner = helper1.copy()[1:-1, 1:-1].astype(int)
        o1_inner = cv2.dilate(helper1, self.kernel, iterations=1)[1:-1, 1:-1].astype(int)
        o2_inner = cv2.dilate(helper2, self.kernel2, iterations=1)[1:-1, 1:-1].astype(int)

        img_slice = self.image[1:-1, 1:-1].astype(int)
        mask = (img_slice - o1_inner > 0) & (img_slice - o2_inner > self.h)
        self.result[1:-1, 1:-1] = np.where(mask, img_slice, 0)

        self.compstar[1:-1, 1:-1] = np.maximum(e_inner, self.result[1:-1, 1:-1].astype(int))
        self.c8[1:-1, 1:-1] = countf(self.image)
        mask_c8 = self.c8[1:-1, 1:-1] >= 2
        self.compstar[1:-1, 1:-1][mask_c8] = self.image[1:-1, 1:-1][mask_c8]

        self.image = self.compstar.copy()

    def after_processing(self):
        helper1 = cv2.erode(self.image, self.kernel, iterations=1)
        helper2 = cv2.erode(self.image, self.kernel2, iterations=1)
        o1 = cv2.dilate(helper1, self.kernel, iterations=1)
        o2 = cv2.dilate(helper2, self.kernel2, iterations=1)

        mask1 = (self.image[1:-1, 1:-1] - o1[1:-1, 1:-1]) > 0
        mask2 = (self.image[1:-1, 1:-1] - o2[1:-1, 1:-1]) > self.h

        self.result[1:-1, 1:-1] = np.where(mask1 & mask2, self.image[1:-1, 1:-1], 0)

    def run(self):
        for i in range(10):
            self.step()
        self.after_processing()
        return self.result
