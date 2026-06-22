"""Cutting out, cropping, and fitting peaks in 2D histograms.

A "peak" is a sub-grid ``Peak(x, y, z)`` (see grid2d.py for the grid
convention), usually a region of a larger histogram containing one mode.

For general grid/histogram utilities, see grid2d.py.
"""

from collections import namedtuple

import numpy as np
from scipy.stats import multivariate_normal

from .grid2d import argopt2d, grid_norm, histogram2d, slice_from_range


Peak = namedtuple('Peak', 'x y z')


# --------------------------------------------------------------------------- #
# Extracting peaks by coordinate range
# --------------------------------------------------------------------------- #

def peak_from_range(x, y, z, ranges):
    """Sub-grid ``Peak(x, y, z)`` inside ``((xmin, xmax), (ymin, ymax))``.

    Returns views into ``x``, ``y``, ``z`` (no copy).
    """
    (xrange, yrange) = ranges
    xslice = slice_from_range(x, xrange)
    yslice = slice_from_range(y, yrange)
    return Peak(x[xslice], y[yslice], z[xslice, yslice])


def peaks_from_ranges(x, y, z, ranges):
    """Map ``{label: ((xmin, xmax), (ymin, ymax))}`` to ``{label: Peak}``."""
    return {label: peak_from_range(x, y, z, r) for label, r in ranges.items()}


def argopt2d_in_range(x, y, z, ranges, op):
    """Locate the min/max of ``z`` within a coordinate window, in full-grid indices.

    Restricts the search to the sub-grid inside ``ranges = ((xmin, xmax),
    (ymin, ymax))``, finds the argmin/argmax there with ``argopt2d``, then
    shifts the local indices back into the original full grid by adding each
    slice's start offset. So the returned ``(i, j)`` index ``z`` directly, not
    the cropped sub-grid.

    Parameters
    ----------
    x, y : array, shape ``(nx,)``, ``(ny,)``
    z : array, shape ``(nx, ny)``
    ranges : ((xmin, xmax), (ymin, ymax))
    op : {'min', 'max'}

    Returns
    -------
    (i, j) : tuple of int
        Indices into the full ``z`` of the extremum within ``ranges``.
    """
    xrange, yrange = ranges
    xslice = slice_from_range(x, xrange)
    yslice = slice_from_range(y, yrange)
    i, j = argopt2d(z[xslice, yslice], op)
    return i + xslice.start, j + yslice.start


# --------------------------------------------------------------------------- #
# Cropping empty histogram borders
# --------------------------------------------------------------------------- #

def crop_nonzero(x, y, z, ipad=0, return_slices=False):
    """Crop ``(x, y, z)`` to the bounding box of ``z > 0``.

    ``ipad`` grid points of padding are kept on every side (clamped to the
    array bounds). With ``return_slices=True``, return ``(Peak, xslice, yslice)``.
    """
    xis, yis = np.where(z > 0)
    xslice = slice(max(int(xis.min()) - ipad, 0), int(xis.max()) + 1 + ipad)
    yslice = slice(max(int(yis.min()) - ipad, 0), int(yis.max()) + 1 + ipad)
    peak = Peak(x[xslice], y[yslice], z[xslice, yslice])
    if return_slices:
        return peak, xslice, yslice
    return peak


def crop_histogram2d(x, y, data, ipad=0, return_slices=False, **hist_kwargs):
    """Histogram ``data`` onto grid ``(x, y)``, then crop to the nonzero region.

    Convenience wrapper around ``histogram2d`` + ``crop_nonzero``. Extra keyword
    arguments (``weights``, ``density``) are forwarded to ``histogram2d``.
    """
    z = histogram2d(x, y, data, **hist_kwargs)
    return crop_nonzero(x, y, z, ipad=ipad, return_slices=return_slices)


# --------------------------------------------------------------------------- #
# Norms
# --------------------------------------------------------------------------- #

def norms_from_peaks(peaks):
    """Per-peak integrals and their total, for peaks holding PDF grids.

    Returns ``(norms, total)`` where ``norms`` is ``{label: norm}``.
    """
    norms = {label: grid_norm(*p) for label, p in peaks.items()}
    return norms, sum(norms.values())


# --------------------------------------------------------------------------- #
# Gaussian fitting
# --------------------------------------------------------------------------- #

def fit_gaussian2d(x, y, z):
    """Moment-fit a single 2D Gaussian to a peak ``z`` on grid ``(x, y)``.

    The peak is renormalized to a probability mass ``P`` (``sum(P) == 1``) and
    the mean and covariance are computed as discrete moments::

        mean = sum_s P(s) s
        cov  = sum_s P(s) (s - mean)(s - mean)'

    Returns
    -------
    height : float
        Integral of ``z`` over the grid (scales a unit-norm Gaussian to the peak).
    mean : array, shape ``(2,)``
    cov : array, shape ``(2, 2)``
    """
    height = grid_norm(x, y, z)
    P = z / np.sum(z)

    px = P.sum(axis=1)   # marginal over y -> weights for x
    py = P.sum(axis=0)   # marginal over x -> weights for y
    xmean = px @ x
    ymean = py @ y
    mean = np.array([xmean, ymean])

    dx = x - xmean
    dy = y - ymean
    cxx = px @ dx ** 2
    cyy = py @ dy ** 2
    cxy = dx @ P @ dy
    cov = np.array([[cxx, cxy], [cxy, cyy]])

    return height, mean, cov


def fit_multivariate_normal(x, y, z):
    """Like ``fit_gaussian2d`` but return ``(height, scipy multivariate_normal)``."""
    height, mean, cov = fit_gaussian2d(x, y, z)
    return height, multivariate_normal(mean, cov)
