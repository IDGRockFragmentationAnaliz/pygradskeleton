from numba import njit
import numpy as np
from .extensible4 import extensible4
from .mctopo.pconstr4 import pconstr4
from .mctopo.delta4p import delta4p
from .mctopo.saddle4 import saddle4
from .mctopo.alpha8p import alpha8p
from .voisin import voisin

@njit
def crestrestore(image, copy=True, n_repeat=50000):
    image = image.copy() if copy else image
    height, width = image.shape

    # максимум внутренних точек
    max_points = (height - 2) * (width - 2)

    ys = np.empty(max_points, dtype=np.int32)
    xs = np.empty(max_points, dtype=np.int32)
    a = np.empty(max_points, dtype=image.dtype)
    mask = np.zeros_like(image, dtype=np.bool)
    n_points = 0

    y2s = np.empty(max_points, dtype=np.int32)
    x2s = np.empty(max_points, dtype=np.int32)

    # 1-й проход: собираем только активные точки
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            _a = extensible4(image, y, x)
            if _a > 0:
                ys[n_points] = y
                xs[n_points] = x
                a[n_points] = _a
                n_points += 1

    image_copy = image.copy()
    for r in range(n_repeat):
        n2_points = 0
        for i in range(n_points):
            y = ys[i]
            x = xs[i]
            _a = a[i]
            mask[y, x] = False
            if pconstr4(image, y, x) is True:
                _d = delta4p(image, y, x)
                image[y, x] = min(_d, _a)

                y2s[n2_points] = y
                x2s[n2_points] = x
                n2_points += 1
            elif saddle4(image, y, x) is True:
                image[y, x] = alpha8p(image, y, x)

                y2s[n2_points] = y
                x2s[n2_points] = x
                n2_points += 1

        n_points = 0
        for i in range(n2_points):
            py = y2s[i]
            px = x2s[i]

            pa = extensible4(image, py, px)
            if pa > 0 and mask[py, px] is False:
                ys[n_points] = py
                xs[n_points] = px
                a[n_points] = pa
                mask[py, px] = True
                n_points += 1

            for k in range(8):
                qy, qx = voisin(py, px, k)
                if qy == 0 or qx == 0 or qy == height or qx == width:
                    continue

                qa = extensible4(image, qy, qx)
                if qa > 0 and mask[qy, qx] is False:
                    ys[n_points] = qy
                    xs[n_points] = qx
                    a[n_points] = qa
                    mask[qy, qx] = True
                    n_points += 1

        if n_points == 0:
            break

        print("test: ", r, "points: ", n_points)
    return image




# @njit(parallel=False)
# def crestrestore(image, copy=True):
#     image = image.copy() if copy else image
#     height, width = image.shape
#
#     a = np.zeros_like(image)
#
#
#     for y in range(1, height - 1):
#         for x in range(1, width - 1):
#             _a = extensible4(image, y, x)
#             if _a > 0:
#                 a[y, x] = _a
#
#     for y in range(1, height - 1):
#         for x in range(1, width - 1):
#             if a[y, x] > 0:
#                 if pconstr4(image, y, x) == 1:
#                     _d = delta4p(image, y, x)
#
#     return image


    #_a = a[y, x]
                    #image[y, x] = _d
                    # if _d < _a:
                    #     image[y, x] = _d
                    # else:
                    #     image[y, x] = _a