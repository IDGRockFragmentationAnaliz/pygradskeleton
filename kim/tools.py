import numpy as np


def countf(img):
    # Создаём центральную область изображения (без краёв)
    center = img[1:-1, 1:-1]
    right = img[1:-1, 2:]
    left = img[1:-1, :-2]
    up = img[:-2, 1:-1]
    down = img[2:, 1:-1]
    up_right = img[:-2, 2:]
    up_left = img[:-2, :-2]
    down_left = img[2:, :-2]
    down_right = img[2:, 2:]

    # Вычисляем f1, f3, f5, f7
    f1 = np.where(
        (right < center) & ((right < up_right) | (right < up)),
        -1,
        np.where((right > up) & (up < center), 1, 0)
    )

    f3 = np.where(
        (up < center) & ((up < up_left) | (up < left)),
        -1,
        np.where((up > left) & (left < center), 1, 0)
    )

    f5 = np.where(
        (left < center) & ((left < down_left) | (left < down)),
        -1,
        np.where((left > down) & (down < center), 1, 0)
    )

    f7 = np.where(
        (down < center) & ((down < down_right) | (down < right)),
        -1,
        np.where((down > right) & (right < center), 1, 0))

    # Вычисляем f2, f4, f6, f8
    f2 = np.where((up_right > up) & (up < center), 1, 0)
    f4 = np.where((up_left > left) & (left < center), 1, 0)
    f6 = np.where((down_left > down) & (down < center), 1, 0)
    f8 = np.where((down_right > right) & (right < center), 1, 0)

    clockwise = [f1, f2, f3, f4, f5, f6, f7, f8, f1]
    sign_changes = np.zeros_like(center, dtype=np.int32)
    last_sign = np.zeros_like(center, dtype=np.int8)

    for x in clockwise:
        mask = x != 0
        sign_changes += ((x == 1) & (last_sign == -1)).astype(np.int32)
        last_sign = np.where(x != 0, x, last_sign)

    return sign_changes
