import cv2
import numpy as np
from time import time
from tqdm import tqdm
from .tools import countf
import torch
import torch.nn.functional as F

class Kim:
    def __init__(self, grayscale_image, steps=10, h=20):
        ndim = (len(grayscale_image.shape))
        if ndim == 2:
            self.image = grayscale_image
        else:
            self.image = grayscale_image[:, :, 0]
        self.h = h
        self.steps = steps
        shape = self.image.shape
        self.compstar = np.zeros(shape)
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
        t = time()
        helper1 = cv2.erode(self.image, self.kernel, iterations=1)
        helper2 = cv2.erode(self.image, self.kernel2, iterations=1)
        e = helper1.copy().astype(int)
        o1 = cv2.dilate(helper1, self.kernel, iterations=1).astype(int)
        o2 = cv2.dilate(helper2, self.kernel2, iterations=1).astype(int)
        img = torch.tensor(self.image, dtype=torch.uint8, device='cuda')
        compstar = self.get_compstar(img, o1, o2, e)
        c8 = countf(img) >= 2
        if torch.equal(compstar[1:-1, 1:-1][c8], img[1:-1, 1:-1][c8]):
            return True

        compstar[1:-1, 1:-1][c8] = img[1:-1, 1:-1][c8]
        self.image = compstar.cpu().numpy().copy()
        return False

    def after_processing(self):
        helper1 = cv2.erode(self.image, self.kernel, iterations=1)
        helper2 = cv2.erode(self.image, self.kernel2, iterations=1)
        o1 = cv2.dilate(helper1, self.kernel, iterations=1)
        o2 = cv2.dilate(helper2, self.kernel2, iterations=1)

        mask1 = (self.image[1:-1, 1:-1] - o1[1:-1, 1:-1]) > 0
        mask2 = (self.image[1:-1, 1:-1] - o2[1:-1, 1:-1]) > self.h

        self.result[1:-1, 1:-1] = np.where(mask1 & mask2, self.image[1:-1, 1:-1], 0)

    def run(self):
        for i in tqdm(range(150)):
            if self.step():
                break
        self.after_processing()
        self.result[self.result > self.h] = 255
        return self.result

    def get_mask(self, img: torch.Tensor, o1: torch.Tensor, o2: torch.Tensor):
        mask_1 = img > o1
        mask_2 = (img >= o2) & ((img - o2) > self.h)
        mask = mask_1 & mask_2
        return mask

    def get_compstar(self, img, o1, o2, e):
        e = torch.tensor(e, dtype=torch.uint8, device='cuda')
        o1 = torch.tensor(o1, dtype=torch.uint8, device='cuda')
        o2 = torch.tensor(o2, dtype=torch.uint8, device='cuda')
        mask = self.get_mask(img, o1, o2)
        result = torch.where(mask, img, 0)
        compstar = torch.maximum(e, result)
        return compstar