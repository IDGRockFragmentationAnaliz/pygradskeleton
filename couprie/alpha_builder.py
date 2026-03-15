import numpy as np


class AlphaBuilder:
    NDG_MAX = 255
    NDG_MIN = 0
    OFFSETS_3X3 = [(dy, dx)
                   for dy in range(-1, 2)
                   for dx in range(-1, 2)
                   if (dy, dx) != (0, 0)]

    def __init__(self, image):
        self.image = image
        self.alpha = np.empty_like(image)

    def alpha8m(self):
        """
            Альфа строго меньше центра,
            но больше или равен любому соседу,
            если таких нет, то центр.
        """
        image_padded = np.pad(self.image, 1, mode='edge')
        image_view = self._build_views3x3(image_padded)
        alpha = self.alpha

        # по умолчанию alpha = image
        np.copyto(alpha, self.image)

        center = image_view[(0, 0)]

        # внутренняя область: полностью векторно
        for offset in self.OFFSETS_3X3:
            neighbor = image_view[offset]
            mask = (neighbor < center) & ((alpha == center) | (neighbor > alpha))
            np.copyto(alpha, neighbor, where=mask)

        return alpha

    @staticmethod
    def _build_views3x3(image):
        offsets_3x3 = tuple(
            (dy, dx)
            for dy in range(-1, 2)
            for dx in range(-1, 2)
        )
        view = {}
        for dy, dx in offsets_3x3:
            y_start = 1 + dy
            y_stop = image.shape[0] - 1 + dy
            x_start = 1 + dx
            x_stop = image.shape[1] - 1 + dx

            view[(dy, dx)] = image[y_start:y_stop, x_start:x_stop]
        return view