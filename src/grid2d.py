"""Utilities for building, measuring, and indexing 2D grids and histograms.

A "2D grid" here means a pair of 1D coordinate arrays ``x`` (shape ``(nx,)``)
and ``y`` (shape ``(ny,)``) together with a value array ``z`` of shape
``(nx, ny)`` such that ``z[i, j]`` is the value at ``(x[i], y[j])``.

For peak extraction, cropping, and Gaussian fitting, see ``peaks.py``.
"""

import multiprocessing as mp

import numpy as np


# --------------------------------------------------------------------------- #
# Grid construction
# --------------------------------------------------------------------------- #

def ranges_data(data, pad=0.07):
    """Per-dimension ``[min, max]`` of ``data`` (shape ``(npoints, ndim)``), padded.

    ``pad`` is the fraction of each dimension's peak-to-peak range to extend
    beyond the data on each side.

    Returns ``[[xmin, xmax], [ymin, ymax], ...]``.
    """
    def axis_range(v):
        dv = pad * np.ptp(v)
        return [np.min(v) - dv, np.max(v) + dv]

    return [axis_range(data[:, n]) for n in range(data.shape[1])]


def _compute_row(x_i, y, func):
    """Evaluate ``func(x_i, y_j)`` for every ``y_j`` in ``y`` (one grid row)."""
    return np.array([func(x_i, y_j) for y_j in y])


def grid_from_arrays(x, y, func, processes=None):
    """Return ``z`` with shape ``(len(x), len(y))``, ``z[i, j] = func(x[i], y[j])``.

    Parameters
    ----------
    x, y : array-like
        The two axes of the grid.
    func : callable
        Scalar function of two scalars. When ``processes`` is not None it must
        be picklable (a module-level function or a ``functools.partial`` of one,
        not a lambda or a nested closure).
    processes : int or None
        None evaluates serially in this process. An int uses a multiprocessing
        pool of that many workers, distributing one grid row per task.
    """
    if processes is None:
        z = np.empty((len(x), len(y)), dtype=float)
        for i, x_ in enumerate(x):
            for j, y_ in enumerate(y):
                z[i, j] = func(x_, y_)
        return z

    with mp.Pool(processes=processes) as pool:
        rows = pool.starmap(_compute_row, [(x_i, y, func) for x_i in x])
    return np.vstack(rows)


# --------------------------------------------------------------------------- #
# Measure (area element and integral)
# --------------------------------------------------------------------------- #

def grid_da(x, y):
    """Area element ``dx * dy`` of a uniform grid."""
    return (x[1] - x[0]) * (y[1] - y[0])


def grid_norm(x, y, z):
    """Integral of ``z = f(x, y)`` over the grid (area-element rule)."""
    return grid_da(x, y) * np.sum(z)


# --------------------------------------------------------------------------- #
# Histograms
# --------------------------------------------------------------------------- #

def _bin_edges(centers):
    """``(lo, hi, nbins)`` for uniform bins whose CENTERS are ``centers``."""
    dx = centers[1] - centers[0]
    return centers[0] - dx / 2, centers[-1] + dx / 2, len(centers)


def histogram2d(x, y, data, weights=None, density=True):
    """Histogram ``data`` onto a grid whose points are bin CENTERS.

    Parameters
    ----------
    x, y : array, shape ``(nx,)``, ``(ny,)``
        Grid points interpreted as bin centers.
    data : array, shape ``(nframes, 2)``
        Rows ``(xi, yi)`` to bin.
    weights : array, shape ``(nframes,)`` or None
    density : bool
        Passed through to ``np.histogram2d``.

    Returns
    -------
    H : array, shape ``(nx, ny)``, with ``H[i, j] ~ p(x_i, y_j)``.
    """
    xlo, xhi, xbins = _bin_edges(x)
    ylo, yhi, ybins = _bin_edges(y)
    H, _, _ = np.histogram2d(
        data[:, 0], data[:, 1],
        bins=[xbins, ybins], range=[[xlo, xhi], [ylo, yhi]],
        density=density, weights=weights,
    )
    return H


# --------------------------------------------------------------------------- #
# Indexing / slicing / argopt
# --------------------------------------------------------------------------- #

def range_indexes(x, xmin, xmax):
    """Inclusive index bounds ``(imin, imax)`` of points with ``xmin < x < xmax``."""
    inside = np.where((x > xmin) & (x < xmax))[0]
    return inside[0], inside[-1]


def slices_from_ranges(x, y, ranges):
    """Slices selecting the sub-grid inside ``((xmin, xmax), (ymin, ymax))``.

    Both slices are inclusive of the last in-range point.
    """
    (xmin, xmax), (ymin, ymax) = ranges
    ix0, ix1 = range_indexes(x, xmin, xmax)
    iy0, iy1 = range_indexes(y, ymin, ymax)
    return slice(ix0, ix1 + 1), slice(iy0, iy1 + 1)


def argopt2d(a, op):
    """Index ``(i, j)`` of the min or max of 2D array ``a``. ``op`` in {'min', 'max'}."""
    if a.ndim != 2:
        raise ValueError(f'argopt2d expects a 2D array, got ndim={a.ndim}')
    try:
        f = {'min': np.argmin, 'max': np.argmax}[op]
    except KeyError:
        raise ValueError(f"op must be 'min' or 'max', got {op!r}")
    return np.unravel_index(f(a), a.shape)


def argmin2d(a):
    return argopt2d(a, 'min')


def argmax2d(a):
    return argopt2d(a, 'max')
