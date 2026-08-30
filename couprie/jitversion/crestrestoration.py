from operator import index

import numpy as np
from numba import njit

from .mctopo.nbtopo import T4_ZEROS, T8_ONES


@njit(cache=True, inline="always")
def _is_nonborder(p, width, size):
    return (
        p >= width
        and p < size - width
        and p % width != 0
        and p % width != width - 1
    )


@njit(cache=True, inline="always")
def _neighbor_c(p, k, width, size):
    """Pink voisin() order: E, NE, N, NW, W, SW, S, SE."""
    x = p % width

    if k == 0:
        return p + 1 if x != width - 1 else -1
    if k == 1:
        return p + 1 - width if x != width - 1 and p >= width else -1
    if k == 2:
        return p - width if p >= width else -1
    if k == 3:
        return p - width - 1 if p >= width and x != 0 else -1
    if k == 4:
        return p - 1 if x != 0 else -1
    if k == 5:
        return p + width - 1 if x != 0 and p < size - width else -1
    if k == 6:
        return p + width if p < size - width else -1
    return p + width + 1 if p < size - width and x != width - 1 else -1


@njit(cache=True, inline="always")
def _mask_ge_level(f, p, level, width):
    mask = 0
    if int(f[p + 1]) >= level:
        mask |= 1
    if int(f[p + 1 - width]) >= level:
        mask |= 2
    if int(f[p - width]) >= level:
        mask |= 4
    if int(f[p - width - 1]) >= level:
        mask |= 8
    if int(f[p - 1]) >= level:
        mask |= 16
    if int(f[p + width - 1]) >= level:
        mask |= 32
    if int(f[p + width]) >= level:
        mask |= 64
    if int(f[p + width + 1]) >= level:
        mask |= 128
    return mask


@njit(cache=True, inline="always")
def _mask_gt_level(f, p, level, width):
    mask = 0
    if int(f[p + 1]) > level:
        mask |= 1
    if int(f[p + 1 - width]) > level:
        mask |= 2
    if int(f[p - width]) > level:
        mask |= 4
    if int(f[p - width - 1]) > level:
        mask |= 8
    if int(f[p - 1]) > level:
        mask |= 16
    if int(f[p + width - 1]) > level:
        mask |= 32
    if int(f[p + width]) > level:
        mask |= 64
    if int(f[p + width + 1]) > level:
        mask |= 128
    return mask


@njit(cache=True, inline="always")
def _separant4_c(f, p, width, size):
    if not _is_nonborder(p, width, size):
        return False

    center = int(f[p])
    if T4_ZEROS[_mask_ge_level(f, p, center, width)] >= 2:
        return True

    for k in range(8):
        q = _neighbor_c(p, k, width, size)
        level = int(f[q])
        if level < center:
            if T4_ZEROS[_mask_ge_level(f, p, level, width)] >= 2:
                return True
    return False


@njit(cache=True, inline="always")
def _hseparant4_c(f, p, h, width, size):
    if not _is_nonborder(p, width, size):
        return False

    center = int(f[p])
    if center <= h:
        return False
    if T4_ZEROS[_mask_ge_level(f, p, center, width)] >= 2:
        return True

    for k in range(8):
        q = _neighbor_c(p, k, width, size)
        level = int(f[q])
        if h < level < center:
            if T4_ZEROS[_mask_ge_level(f, p, level, width)] >= 2:
                return True
    return False


@njit(cache=True, inline="always")
def _extensible4_c(f, condition, p, width, size):
    if not _separant4_c(f, p, width, size):
        return 0

    center = int(f[p])
    extension_level = 0
    for k in range(8):
        q = _neighbor_c(p, k, width, size)
        q_level = int(f[q])
        if condition[q] != 0 and q_level > center:
            if (
                _hseparant4_c(f, q, center - 1, width, size)
                and not _hseparant4_c(f, q, center, width, size)
                and q_level > extension_level
            ):
                extension_level = q_level
    return extension_level


@njit(cache=True, inline="always")
def _pconstr4_c(f, p, width, size):
    if not _is_nonborder(p, width, size):
        return False

    center = int(f[p])
    mask_gt = _mask_gt_level(f, p, center, width)
    return T4_ZEROS[mask_gt] == 1 and T8_ONES[mask_gt] == 1


@njit(cache=True, inline="always")
def _saddle4_c(f, p, width, size):
    if not _is_nonborder(p, width, size):
        return False

    center = int(f[p])
    t4mm = T4_ZEROS[_mask_ge_level(f, p, center, width)]
    t8pp = T8_ONES[_mask_gt_level(f, p, center, width)]
    return t8pp > 1 and t4mm > 1


@njit(cache=True, inline="always")
def _alpha8p_c(f, p, width):
    center = int(f[p])
    alpha = 256

    value = int(f[p + 1])
    if center < value < alpha:
        alpha = value
    value = int(f[p + 1 - width])
    if center < value < alpha:
        alpha = value
    value = int(f[p - width])
    if center < value < alpha:
        alpha = value
    value = int(f[p - width - 1])
    if center < value < alpha:
        alpha = value
    value = int(f[p - 1])
    if center < value < alpha:
        alpha = value
    value = int(f[p + width - 1])
    if center < value < alpha:
        alpha = value
    value = int(f[p + width])
    if center < value < alpha:
        alpha = value
    value = int(f[p + width + 1])
    if center < value < alpha:
        alpha = value

    return center if alpha == 256 else alpha


@njit(cache=True, inline="always")
def _delta4p_c(f, p, width, size):
    saved = f[p]
    while _pconstr4_c(f, p, width, size):
        f[p] = _alpha8p_c(f, p, width)
    result = f[p]
    f[p] = saved
    return result


@njit(cache=True, inline="always")
def _nbvoiss8_c(f, p, level, width, size):
    count = 0
    for k in range(8):
        q = _neighbor_c(p, k, width, size)
        if q != -1 and int(f[q]) >= level:
            count += 1
    return count


@njit(cache=True, inline="always")
def _colextensible4_c(f, condition, p, width, size):
    """COLMULTI with EXTENSIBLE_TOPO, EXTENSIBLE_GEO and EXTENSIBLE_MARK."""
    center = int(f[p])
    higher_neighbours = 0

    for k in range(8):
        q = _neighbor_c(p, k, width, size)
        if q == -1 or int(f[q]) <= center:
            continue

        if not _hseparant4_c(f, q, center, width, size):
            higher_neighbours += 1
        if _nbvoiss8_c(f, q, center + 1, width, size) <= 1:
            higher_neighbours += 1
        if condition[q] != 0:
            higher_neighbours += 1

        if higher_neighbours >= 2:
            return True

    return False


@njit(cache=True)
def _crestrestore_c_kernel(image, condition, nitermax):
    height, width = image.shape
    size = height * width
    f = image.ravel()

    lifo1_points = np.empty(size, dtype=np.intp)
    lifo1_levels = np.empty(size, dtype=np.uint8)
    lifo2 = np.empty(size, dtype=np.intp)
    in_lifo = np.zeros(size, dtype=np.uint8)

    lifo1_size = 0
    for p in range(size):
        extension_level = _extensible4_c(f, condition, p, width, size)
        if extension_level != 0:
            lifo1_points[lifo1_size] = p
            lifo1_levels[lifo1_size] = extension_level
            lifo1_size += 1

    iteration = 0
    modified = 0

    while lifo1_size != 0 and iteration < nitermax:
        iteration += 1
        lifo2_size = 0

        # First half-iteration: raise constructible and eligible saddle points.
        while lifo1_size != 0:
            lifo1_size -= 1
            p = lifo1_points[lifo1_size]
            extension_level = lifo1_levels[lifo1_size]
            in_lifo[p] = 0

            raised = False
            if _pconstr4_c(f, p, width, size):
                delta = int(_delta4p_c(f, p, width, size))
                f[p] = min(delta, int(extension_level))
                raised = True
            elif _saddle4_c(f, p, width, size) and _colextensible4_c(
                f, condition, p, width, size
            ):
                f[p] = _alpha8p_c(f, p, width)
                raised = True

            if raised:
                condition[p] = 1
                lifo2[lifo2_size] = p
                lifo2_size += 1
                modified += 1

        # Second half-iteration: enqueue newly extensible modified points/neighbours.
        while lifo2_size != 0:
            lifo2_size -= 1
            p = lifo2[lifo2_size]

            if in_lifo[p] == 0:
                extension_level = _extensible4_c(f, condition, p, width, size)
                if extension_level != 0:
                    lifo1_points[lifo1_size] = p
                    lifo1_levels[lifo1_size] = extension_level
                    lifo1_size += 1
                    in_lifo[p] = 1

            for k in range(8):
                q = _neighbor_c(p, k, width, size)
                if q != -1 and in_lifo[q] == 0:
                    extension_level = _extensible4_c(
                        f, condition, q, width, size
                    )
                    if extension_level != 0:
                        lifo1_points[lifo1_size] = q
                        lifo1_levels[lifo1_size] = extension_level
                        lifo1_size += 1
                        in_lifo[q] = 1

    return iteration, modified


def crestrestore(
    image,
    copy=True,
    n_repeat=50000,
    imagecond=None,
    connex=4,
    progress=False,
):
    """Restore crests like Pink's 2-D ``lcrestrestoration``.

    The implementation reproduces the original 4-connected C algorithm,
    including its LIFO scheduling, ``COLMULTI`` saddle-point test and dynamic
    condition set. ``n_repeat=-1`` means no practical iteration limit.

    When ``imagecond`` is supplied, it is updated in place to the final binary
    condition set (0 or 255), as in the original C function.
    """
    connex = index(connex)
    if connex != 4:
        raise NotImplementedError("Pink lcrestrestoration implements only connex=4")

    n_repeat = index(n_repeat)
    int32 = np.iinfo(np.int32)
    if n_repeat < int32.min or n_repeat > int32.max:
        raise OverflowError("n_repeat must fit into a signed 32-bit integer")
    nitermax = int32.max if n_repeat == -1 else n_repeat

    source = np.asarray(image)
    if source.ndim != 2:
        raise ValueError("crestrestore expects a 2-D image")
    if source.dtype != np.uint8:
        raise TypeError("crestrestore expects image.dtype == numpy.uint8")
    if source.size > 1 << 24:
        raise ValueError("crestrestore supports at most 2^24 pixels")

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

    condition_target = None
    if imagecond is None:
        condition = np.ones(result.size, dtype=np.uint8)
    else:
        if not isinstance(imagecond, np.ndarray):
            raise TypeError("imagecond must be a numpy.ndarray")
        condition_source = np.asarray(imagecond)
        if condition_source.shape != result.shape:
            raise ValueError("imagecond must have the same shape as image")
        if condition_source.dtype != np.uint8:
            raise TypeError("imagecond.dtype must be numpy.uint8")
        if not condition_source.flags.writeable:
            raise ValueError("imagecond must be writeable")
        condition_target = condition_source
        condition = np.ascontiguousarray(condition_source != 0, dtype=np.uint8).ravel()

    iterations, modified = _crestrestore_c_kernel(result, condition, nitermax)

    if condition_target is not None:
        condition_target[...] = condition.reshape(result.shape) * np.uint8(255)

    if progress:
        try:
            from tqdm.auto import tqdm
        except ImportError:
            print("tqdm is not installed; progress display disabled")
        else:
            with tqdm(total=iterations, desc="crest restoration C") as pbar:
                pbar.update(iterations)
                pbar.set_postfix(modified=modified)

    return result


# Name used by Pink's public API.
crestrestoration = crestrestore
