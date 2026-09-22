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


def grid1d_from_range(range, dx=None, num_points=100):
    """Return a grid (over range) based on either:

    dx : float = grid spacing
    num_points: int = length of grid
    """
    if dx is not None:
        return np.arange(*range, dx)
    else:
        return np.linspace(*range, num_points)


def periodic_grid(num_points, start=-np.pi, period=2 * np.pi):
    """Bin CENTERS that tile one period ``[start, start + period)`` exactly.

    Use this, not ``np.linspace(-pi, pi, n)``, for angles.  linspace includes
    both endpoints, which are the same angle, and when its points are used as
    bin centers the outer edges land at ``-pi - dx/2`` and ``pi + dx/2``: the
    first and last bins then cover only half their width inside the domain,
    and show about half the true density.  Here the bin edges fall exactly on
    ``start`` and ``start + period``, every bin has width ``period / n``, and
    no angle is represented twice.

    Paired with :func:`periodic_index` (bin lookup) and :func:`wrap_periodic`
    (map data into the half-open period before calling :func:`hist2d`).
    """
    dx = period / num_points
    return start + dx * (np.arange(num_points) + 0.5)


def wrap_periodic(theta, start=-np.pi, period=2 * np.pi):
    """Map angles into the half-open period ``[start, start + period)``.

    mdtraj returns dihedrals in ``(-pi, pi]``; wrapping sends ``pi`` to ``-pi``
    so it lands in the first bin, consistent with :func:`periodic_index`.
    """
    return np.mod(np.asarray(theta) - start, period) + start


def periodic_index(theta, num_points, start=-np.pi, period=2 * np.pi):
    """Index of the :func:`periodic_grid` bin containing each angle.

    Wraps around: ``pi`` and ``-pi`` both map to bin 0, and values just outside
    the period map to the bin at the opposite end.
    """
    # Wrap first: computing floor((pi + pi) / dx) directly gives
    # floor(99.99999...) = 99 in floating point, not 100 -> 0.  The clip covers
    # np.mod returning exactly `period` for tiny negative offsets.
    dx = period / num_points
    t = wrap_periodic(theta, start=start, period=period)
    idx = np.floor((t - start) / dx).astype(int)
    return np.clip(idx, 0, num_points - 1)


def _compute_row(x_i, y, func):
    """Evaluate ``func(x_i, y_j)`` for every ``y_j`` in ``y`` (one grid row)."""
    return np.array([func(x_i, y_j) for y_j in y])


def compute_grid2d(x, y, func, processes=None):
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
    """``(vmin, vmax, nbins)`` for uniform bins whose CENTERS are ``centers``."""
    dx = centers[1] - centers[0]
    return centers[0] - dx / 2, centers[-1] + dx / 2, len(centers)


def hist2d(x, y, data, weights=None, density=True):
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
    xmin, xmax, xbins = _bin_edges(x)
    ymin, ymax, ybins = _bin_edges(y)
    H, _, _ = np.histogram2d(
        data[:, 0], data[:, 1],
        bins=[xbins, ybins], range=[[xmin, xmax], [ymin, ymax]],
        density=density, weights=weights,
    )
    return H


# --------------------------------------------------------------------------- #
# Indexing / slicing / argopt
# --------------------------------------------------------------------------- #

def range_indexes(x, xmin, xmax):
    """Returns
    -------
    (imin, imax) : tuple of int
        Indices of the first and last in-range elements of ``x``.

    Notes
    -----
    Raises ``IndexError`` if no element of ``x`` falls inside the interval. If
    ``x`` is not sorted, ``imin``/``imax`` still mark the first and last
    in-range positions, but the span ``imin:imax + 1`` may then include
    out-of-range points.
    """
    inside = np.where((x > xmin) & (x < xmax))[0]
    return inside[0], inside[-1]


def slice_from_range(x, range):
    """Slice selecting the elements of grid axis ``x`` inside ``range = (xmin, xmax)``.

    Inclusive of the last in-range point: if ``range_indexes`` reports in-range
    index bounds ``(i0, i1)``, the returned slice is ``slice(i0, i1 + 1)`` so
    that ``x[slice_from_range(x, range)]`` keeps the element at ``i1``.
    """
    xmin, xmax = range
    i0, i1 = range_indexes(x, xmin, xmax)
    return slice(i0, i1 + 1)


def argopt2d(a, op):
    """Index ``(i, j)`` of the minimum or maximum element of a 2D array.
    """
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
