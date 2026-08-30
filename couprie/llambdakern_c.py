from operator import index

import numpy as np
from numba import njit

from .jitversion.mctopo.nbtopo import T4_ZEROS, T8_ONES
from .jitversion.mctopo.topology import get_comp4tab


COMP4TAB = get_comp4tab()
_QUEUE_LEVELS = 256


@njit(cache=True, inline="always")
def _is_nonborder(p, w, n):
    return p >= w and p < n - w and p % w != 0 and p % w != w - 1


@njit(cache=True, inline="always")
def _neighbor_c(p, k, w, n):
    """Equivalent of Pink's voisin(): E, NE, N, NW, W, SW, S, SE."""
    x = p % w

    if k == 0:  # E
        return p + 1 if x != w - 1 else -1
    if k == 1:  # NE
        return p + 1 - w if x != w - 1 and p >= w else -1
    if k == 2:  # N
        return p - w if p >= w else -1
    if k == 3:  # NW
        return p - w - 1 if p >= w and x != 0 else -1
    if k == 4:  # W
        return p - 1 if x != 0 else -1
    if k == 5:  # SW
        return p + w - 1 if x != 0 and p < n - w else -1
    if k == 6:  # S
        return p + w if p < n - w else -1
    # SE
    return p + w + 1 if p < n - w and x != w - 1 else -1


@njit(cache=True, inline="always")
def _mask_ge_c(f, p, w):
    value = f[p]
    mask = 0
    if f[p + 1] >= value:
        mask |= 1
    if f[p + 1 - w] >= value:
        mask |= 2
    if f[p - w] >= value:
        mask |= 4
    if f[p - w - 1] >= value:
        mask |= 8
    if f[p - 1] >= value:
        mask |= 16
    if f[p + w - 1] >= value:
        mask |= 32
    if f[p + w] >= value:
        mask |= 64
    if f[p + w + 1] >= value:
        mask |= 128
    return mask


@njit(cache=True, inline="always")
def _mask_lower_c(f, p, w):
    value = f[p]
    mask = 0
    if f[p + 1] < value:
        mask |= 1
    if f[p + 1 - w] < value:
        mask |= 2
    if f[p - w] < value:
        mask |= 4
    if f[p - w - 1] < value:
        mask |= 8
    if f[p - 1] < value:
        mask |= 16
    if f[p + w - 1] < value:
        mask |= 32
    if f[p + w] < value:
        mask |= 64
    if f[p + w + 1] < value:
        mask |= 128
    return mask


@njit(cache=True, inline="always")
def _alpha8m_c(f, p, w):
    value = f[p]
    best = value

    v = f[p + 1]
    if v < value:
        best = v

    v = f[p + 1 - w]
    if v < value and (best == value or v > best):
        best = v

    v = f[p - w]
    if v < value and (best == value or v > best):
        best = v

    v = f[p - w - 1]
    if v < value and (best == value or v > best):
        best = v

    v = f[p - 1]
    if v < value and (best == value or v > best):
        best = v

    v = f[p + w - 1]
    if v < value and (best == value or v > best):
        best = v

    v = f[p + w]
    if v < value and (best == value or v > best):
        best = v

    v = f[p + w + 1]
    if v < value and (best == value or v > best):
        best = v

    return best


@njit(cache=True, inline="always")
def _lambdadestr4_c(f, p, lam, w, n):
    if not _is_nonborder(p, w, n):
        return False

    mask_ge = _mask_ge_c(f, p, w)
    t4mm = T4_ZEROS[mask_ge]
    t8p = T8_ONES[mask_ge]

    if t4mm == 1 and t8p == 1:
        return True

    if t4mm == 1 and t8p == 0:
        return int(f[p]) - int(_alpha8m_c(f, p, w)) <= lam

    if t4mm >= 2:
        close_components = 0
        mask_lower = _mask_lower_c(f, p, w)
        center = int(f[p])

        for component_index in range(t4mm):
            component = COMP4TAB[mask_lower, component_index]
            if component == 0:
                break

            close = True
            for k in range(8):
                if component & 1:
                    q = _neighbor_c(p, k, w, n)
                    if center - int(f[q]) > lam:
                        close = False
                        break
                component >>= 1

            if close:
                close_components += 1

        if close_components >= t4mm - 1:
            return True

    return False


@njit(cache=True, inline="always")
def _lower_point_c(f, p, lam, w, n):
    modified = False
    while _lambdadestr4_c(f, p, lam, w, n):
        f[p] = _alpha8m_c(f, p, w)
        modified = True
    return modified


@njit(cache=True)
def _regional_minima4(f, w, n):
    visited = np.zeros(n, dtype=np.uint8)
    minima = np.zeros(n, dtype=np.uint8)
    plateau = np.empty(n, dtype=np.intp)

    for start in range(n):
        if visited[start]:
            continue

        value = f[start]
        head = 0
        tail = 1
        plateau[0] = start
        visited[start] = 1
        is_minimum = True

        while head < tail:
            p = plateau[head]
            head += 1

            # Orthogonal directions in the same order as voisin(k = 0, 2, 4, 6).
            for k in range(0, 8, 2):
                q = _neighbor_c(p, k, w, n)
                if q == -1:
                    continue

                if f[q] < value:
                    is_minimum = False
                elif f[q] == value and visited[q] == 0:
                    visited[q] = 1
                    plateau[tail] = q
                    tail += 1

        if is_minimum:
            for i in range(tail):
                minima[plateau[i]] = 1

    return minima


@njit(cache=True, inline="always")
def _queue_push(heads, tails, next_point, p, priority, min_level):
    next_point[p] = -1
    if heads[priority] == -1:
        heads[priority] = p
        tails[priority] = p
    else:
        next_point[tails[priority]] = p
        tails[priority] = p

    if priority < min_level:
        min_level = priority
    return min_level


@njit(cache=True, inline="always")
def _queue_pop(heads, tails, next_point, min_level):
    while min_level < _QUEUE_LEVELS and heads[min_level] == -1:
        min_level += 1

    p = heads[min_level]
    heads[min_level] = next_point[p]
    next_point[p] = -1

    if heads[min_level] == -1:
        tails[min_level] = -1
        min_level += 1
        while min_level < _QUEUE_LEVELS and heads[min_level] == -1:
            min_level += 1

    return p, min_level


@njit(cache=True)
def _llambdakern_c_kernel(image, constraint, lam):
    h, w = image.shape
    n = h * w
    f = image.ravel()
    g = constraint.ravel()

    minima = _regional_minima4(f, w, n)
    queued = np.zeros(n, dtype=np.uint8)
    heads = np.full(_QUEUE_LEVELS, -1, dtype=np.intp)
    tails = np.full(_QUEUE_LEVELS, -1, dtype=np.intp)
    next_point = np.empty(n, dtype=np.intp)

    queue_size = 0
    min_level = _QUEUE_LEVELS

    # C initialisation: enqueue non-minimum neighbours of regional minima.
    for p in range(n):
        if f[p] > g[p] and minima[p]:
            for k in range(8):
                q = _neighbor_c(p, k, w, n)
                if (q != -1 and minima[q] == 0 and queued[q] == 0 and
                        _is_nonborder(q, w, n)):
                    min_level = _queue_push(
                        heads, tails, next_point, q, int(f[q]), min_level
                    )
                    queued[q] = 1
                    queue_size += 1

    popped = 0
    modified = 0

    while queue_size > 0:
        p, min_level = _queue_pop(heads, tails, next_point, min_level)
        queued[p] = 0
        queue_size -= 1
        popped += 1

        if f[p] > g[p] and _lower_point_c(f, p, lam, w, n):
            modified += 1
            for k in range(8):
                q = _neighbor_c(p, k, w, n)
                if q != -1 and queued[q] == 0 and _is_nonborder(q, w, n):
                    min_level = _queue_push(
                        heads, tails, next_point, q, int(f[q]), min_level
                    )
                    queued[q] = 1
                    queue_size += 1

    return popped, modified


def llambdakern_c(image, lam, copy=True, progress=False, imagecond=None, connex=4):
    """C-compatible port of Pink's 2-D ``llambdakern`` for 4-connectivity.

    ``imagecond=None`` corresponds to ``llambdakern_short`` and uses an all-zero
    constraint image. Like the C implementation, the operation only supports
    8-bit grayscale images and never modifies the outer image frame.
    """
    connex = index(connex)
    if connex != 4:
        raise NotImplementedError("Pink llambdakern implements only connex=4")

    lam = index(lam)
    if lam < np.iinfo(np.int32).min or lam > np.iinfo(np.int32).max:
        raise OverflowError("lam must fit into a signed 32-bit integer")

    source = np.asarray(image)
    if source.ndim != 2:
        raise ValueError("llambdakern_c expects a 2-D image")
    if source.dtype != np.uint8:
        raise TypeError("llambdakern_c expects image.dtype == numpy.uint8")

    if copy:
        result = np.array(source, dtype=np.uint8, order="C", copy=True)
    else:
        if not isinstance(image, np.ndarray):
            raise TypeError("copy=False requires a numpy.ndarray")
        if not source.flags.c_contiguous:
            raise ValueError("copy=False requires a C-contiguous image")
        if not source.flags.writeable:
            raise ValueError("copy=False requires a writeable image")
        result = source

    if imagecond is None:
        constraint = np.zeros(result.shape, dtype=np.uint8)
    else:
        constraint_source = np.asarray(imagecond)
        if constraint_source.shape != result.shape:
            raise ValueError("imagecond must have the same shape as image")
        if constraint_source.dtype != np.uint8:
            raise TypeError("imagecond.dtype must be numpy.uint8")
        constraint = np.ascontiguousarray(constraint_source)

    popped, modified = _llambdakern_c_kernel(result, constraint, lam)

    if progress:
        try:
            from tqdm.auto import tqdm
        except ImportError:
            print("tqdm is not installed; progress display disabled")
        else:
            with tqdm(total=popped, desc="llambdakern C") as pbar:
                pbar.update(popped)
                pbar.set_postfix(modified=modified)

    return result
