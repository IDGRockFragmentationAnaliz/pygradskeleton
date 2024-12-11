import numpy as np
from time import time
from numba import njit, prange
import torch


def countf(img: torch.Tensor):
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

    clockwise = [
        lambda: calculate_f_odd(right, up_right, up, center),
        lambda: calculate_f_even(up_right, up, center),
        lambda: calculate_f_odd(up, up_left, left, center),
        lambda: calculate_f_even(up_left, left, center),
        lambda: calculate_f_odd(left, down_left, down, center),
        lambda: calculate_f_even(down_left, down, center),
        lambda: calculate_f_odd(down, down_right, right, center),
        lambda: calculate_f_even(down_right, right, center),
        lambda: calculate_f_odd(right, up_right, up, center),
    ]

    sign_changes = cuda_culc(clockwise, center.shape)

    return sign_changes


def cuda_culc(clockwise, shape):
    t = time()
    sign_changes = torch.zeros(shape, dtype=torch.int8, device='cuda')
    last_sign = torch.zeros(shape, dtype=torch.int8, device='cuda')

    for x in clockwise:
        _x = x()
        sign_changes += ((_x == 1) & (last_sign == -1)).to(torch.int8)
        last_sign = torch.where(_x != 0, _x, last_sign)
    return sign_changes



def calculate_f_odd(right, up_right, up, center):
    f_result = torch.zeros_like(center, dtype=torch.int32)
    condition_neg = (right < center) & ((right < up_right) | (right < up))
    condition_pos = (right > up) & (up < center)

    f_result[condition_neg] = -1
    f_result[condition_pos] = 1
    return f_result


def calculate_f_even(up_right, up, center):
    return torch.where((up_right > up) & (up < center), 1, 0)
