import numpy as np
from numba import njit
from .jitversion.abaisse4 import abaisse4
from .jitversion.voisin import voisin
from .jitflatversion.utils.set_edge import set_edge_zeros


def llambdakern(image, lam, copy = True, progress=False):
    if copy:
        image = image.copy()

    height, width = image.shape
    n = height * width
    seen_rect = np.ones(image.shape, np.uint8)
    set_edge_zeros(seen_rect)
    seen = seen_rect.ravel()

    # for y in range(1, height - 1):
    #     for x in range(1, width - 1):
    #         seen[y * width + x] = 1
    head = (height - 2) * (width - 2)

    pbar = _make_progress(head, progress)
    first_iteration = True

    while head > 0:
        head = _llambdakern_loop(image, seen, head, lam)
        if pbar is not None:
            pbar.n = head
            if first_iteration:
                pbar.total = head
                first_iteration = False
            pbar.refresh()
    return image

@njit(cache=True)
def _llambdakern_loop(image, seen, head, lam):
    height, width = image.shape
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            p = y * width + x

            if seen[p] == 1:
                seen[p] = 0
                head -= 1

                if abaisse4(image, y, x, lam):
                    for k in range(8):
                        qy, qx = voisin(y, x, k)
                        q = qy * width + qx

                        if qy <= 0 or qy >= height - 1 or qx <= 0 or qx >= width - 1:
                            seen[q] = 0
                        elif seen[q] == 0:
                            seen[q] = 1
                            head += 1

    return head

def _make_progress(total, enabled, desc="llambdakern"):
    if not enabled:
        return None

    try:
        from tqdm.auto import tqdm
    except ImportError:
        print("tqdm is not installed; progress display disabled")
        return None

    return tqdm(total=total, desc=desc)