from operator import and_

import numpy as np
import cv2

NON_DESTRUCTIBLE = 0
DESTRUCTIBLE = 1
DESTRUCTIBLE_PAR = 2
CRUCIAL = 3
CRUCIAL_C = 9
CRUCIAL_C1 = 5


class Matcher:
    OFFSETS_3X3 = [(dy, dx)
                   for dy in range(-1, 2)
                   for dx in range(-1, 2)
                   if (dy, dx) != (0, 0)]

    OFFSETS_5X5 = [(dy, dx)
                   for dy in range(-2, 3)
                   for dx in range(-2, 3)
                   if (dy, dx) != (0, 0)]

    def get_template_mask(self, template):
        mask = np.ones(self.bordered_size, dtype=bool)
        for offset in self.OFFSETS_3X3:
            c = self.image_view[(0, 0)]
            value = self._template_value(template, offset)
            if value < 0:
                continue
            if value > 0:
                mask &= (self.image_view[offset] == c)
                continue
            if value == 0:
                mask &= (self.image_view[offset] < c)
                continue
        return mask

    def match_c(self):
        """
            at least one of the numbered ones must be
            [-1, 1, 1]
            [-1, 2, 2]
            [-1, 3, 3]
            for forth directions
        """
        buffer = self.buffer

        # (1, 0)
        mask = (buffer[(1, 0)] &
                (buffer[(0, 1)] | buffer[(1, 1)]) &
                (buffer[(0, -1)] | buffer[(1, -1)])
        )
        mask &= self.alpha_view[(1, 0)] < self.image_view[(0, 0)]
        self.destroyable_view[(0, 0)][mask] = CRUCIAL_C
        self.destroyable_view[(1, 0)][mask] = CRUCIAL_C

        # (-1, 0)
        mask = (
                buffer[(-1, 0)] &
                (buffer[(0, 1)] | buffer[(-1, 1)]) &
                (buffer[(0, -1)] | buffer[(-1, -1)])
        )
        mask &= self.alpha_view[(-1, 0)] < self.image_view[(0, 0)]
        self.destroyable_view[(0, 0)][mask] = CRUCIAL_C
        self.destroyable_view[(-1, 0)][mask] = CRUCIAL_C

        # (0, 1)
        mask = (
                buffer[(0, 1)] &
                (buffer[(-1, 0)] | buffer[(-1, 1)]) &
                (buffer[(1, 0)] | buffer[(1, 1)])
        )
        mask &= self.alpha_view[(0, 1)] < self.image_view[(0, 0)]
        self.destroyable_view[(0, 0)][mask] = CRUCIAL_C
        self.destroyable_view[(0, 1)][mask] = CRUCIAL_C

        # (0, -1)
        mask = (
                buffer[(0, -1)] &
                (buffer[(-1, 0)] | buffer[(-1, -1)]) &
                (buffer[(1, 0)] | buffer[(1, -1)])
        )
        mask &= self.alpha_view[(0, -1)] < self.image_view[(0, 0)]
        self.destroyable_view[(0, 0)][mask] = CRUCIAL_C
        self.destroyable_view[(0, -1)][mask] = CRUCIAL_C

    def run(self):
        template = np.array([
            [-1,-1,-1],
            [-1, 1, 1],
            [-1, 1, 1],
        ])
        c = self.image_view[(0, 0)]
        mask = self.get_template_mask(template)
        kernel = (template >= 1).astype(np.uint8)
        kernel = np.array([
            [1, 1, 0],
            [1, 1, 0],
            [0, 0, 0],
        ], dtype=np.uint8)
        mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1) > 0
        c[mask] = 0
        return c

    def __init__(self, image, destroyable, alpha):
        self.image = image
        self.size = self.image.shape
        self.bordered_size = (self.size[0] - 4, self.size[1] - 4)
        self.image_view = self._build_views5x5(image)
        self.destroyable = destroyable
        self.destroyable_view = self._build_views5x5(self.destroyable)
        self.alpha = alpha
        self.alpha_view = self._build_views5x5(alpha)
        self.buffer = self._create_buffer()


    def _create_buffer(self):
        buffer = {}
        c = self.image_view[(0, 0)]
        for offset in self.OFFSETS_5X5:
            buffer[offset] = self.image_view[offset] >= c
        return buffer

    @staticmethod
    def _build_views5x5(image):
        offsets_5x5 = tuple(
            (dy, dx)
            for dy in range(-2, 3)
            for dx in range(-2, 3)
        )
        view = {}
        for dy, dx in offsets_5x5:
            y_start = 2 + dy
            y_stop = image.shape[0] - 2 + dy
            x_start = 2 + dx
            x_stop = image.shape[1] - 2 + dx

            view[(dy, dx)] = image[y_start:y_stop, x_start:x_stop]
        return view

    @staticmethod
    def _template_value(template, offset):
        dy, dx = offset
        radius_y = template.shape[0] // 2
        radius_x = template.shape[1] // 2
        return template[dy + radius_y, dx + radius_x]