"""
kernel.py
=========

Single-Gaussian computations.  Every function here evaluates the *unnormalized*
Gaussian kernel

    G(s) = height * exp(-1/2 * ||s - mean||^2),

with the Mahalanobis norm for a diagonal covariance Sigma = diag(width ** 2):

    ||s - mean||^2 = sum_i ((s_i - mean_i) / width_i) ** 2.

There is no normalization prefactor -- ``height`` carries whatever amplitude the
caller wants.  When a *normalized* kernel is needed, the prefactor
1 / ((2*pi)^(d/2) * prod(width)) is folded into ``height`` at construction time
(see gaussians.WeightedGaussians.from_weights); it never appears inside an
evaluation here.

    gaussian2d                  single point / many points
    gaussian2d_grid             full regular grid
    gaussian2d_grid_over_patch  a single Gaussian on an nwidth box of a grid

The generic gridding helpers live in ``grid2d.py``; dataset evaluation lives in
``gaussian2d_dataset.py``.
"""

import numpy as np
from functools import partial

from .grid2d import grid2d_from_arrays


# ----------------------------
# Core math: single 2D Gaussian
# ----------------------------
def gaussian2d(points, height, mean, width, dtype=np.float64):
    """
    Evaluate an unnormalized 2D Gaussian kernel at one or many points:

        G(s) = height * exp(-1/2 [((x - mean0)/width0)^2 + ((y - mean1)/width1)^2]).

    Parameters
    ----------
    points : ndarray, shape (..., 2)
        Points where to evaluate the Gaussian.
    height : float
        Peak value of the kernel (no normalization is applied).
    mean : array-like, shape (2,)
        Gaussian mean.
    width : array-like, shape (2,)
        Per-axis width; the covariance is diag(width ** 2).
    dtype : numpy dtype, optional
        Floating precision of the evaluation. Default np.float64. Pass
        np.float32 to halve memory traffic (and roughly double throughput on
        bandwidth-bound callers) at float32 accuracy. ``height`` is cast to this
        dtype as well, so the returned array always has dtype ``dtype`` rather
        than being silently upcast by numpy scalar promotion.

    Returns
    -------
    vals : ndarray, shape points.shape[:-1], dtype ``dtype``
        Gaussian values at each point.
    """
    points = np.asarray(points, dtype=dtype)
    mean = np.asarray(mean, dtype=dtype).reshape(2,)
    width = np.asarray(width, dtype=dtype).reshape(2,)
    height = np.asarray(height, dtype=dtype)

    d = points - mean
    quad = (d[..., 0] / width[0]) ** 2 + (d[..., 1] / width[1]) ** 2
    return height * np.exp(-0.5 * quad)


def _gaussian2d_point(x, y, height, mean, width):
    """
    Scalar evaluation of the kernel at a single point ``(x, y)``.

    Defined at module level (rather than as a nested closure) so it can be
    pickled and shipped to multiprocessing workers by ``gaussian2d_grid``.
    """
    return gaussian2d(np.array([x, y], dtype=float), height, mean, width).item()


def gaussian2d_grid(x, y, height, mean, width, processes=None, by_row=True):
    """
    Evaluate a single 2D Gaussian over a regular grid.

    Parameters
    ----------
    x : ndarray, shape (numx,)
    y : ndarray, shape (numy,)
    mean, width, height : see gaussian2d
    processes : int or None
        If None: single-process, fully vectorized evaluation (fast).
        If int: multiprocessing evaluation via grid2d.grid2d_from_arrays.
    by_row : bool
        Only relevant if processes is not None.

    Returns
    -------
    grid : ndarray, shape (numx, numy)
        Uses 'ij' indexing (math axes): s1 maps to axis 0, s2 to axis 1.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if processes is None:
        # Fully vectorized grid build.
        xy = np.dstack(np.meshgrid(x, y, indexing="ij"))  # (numx, numy, 2)
        return gaussian2d(xy, height, mean, width)

    # Multiprocessing path (row/point evaluation). A picklable partial of the
    # module-level helper is required here.
    func = partial(_gaussian2d_point, height=height, mean=mean, width=width)
    return grid2d_from_arrays(x, y, func, processes=processes, by_row=by_row)


# ----------------------------------------------------------------------
# Single Gaussian over a *patch* of a regular grid
# ----------------------------------------------------------------------
# Same unnormalized kernel as above, but evaluated only on an nwidth box around
# the center.  Used by gaussians.Gaussians and bias_shift_exact, so
# add_gaussian() can keep adding G_t(s) directly.
def _slice_to_nwidth(grid, center, width, nwidth):
    """Index slice of ``grid`` within +/- nwidth*width of ``center``, clamped.

    ``grid`` is assumed sorted ascending.
    """
    i0 = int(np.searchsorted(grid, center - nwidth * width, side="left"))
    i1 = int(np.searchsorted(grid, center + nwidth * width, side="right"))
    return slice(max(i0, 0), min(i1, len(grid)))


def _gaussian2d_patch(x, y, height, center, width):
    """Unnormalized 2D Gaussian over patch axes ``x``, ``y``.

    Returns shape (len(x), len(y)) with 'ij' orientation:
    out[i, j] = height * exp(-1/2 [((x_i-cx)/wx)^2 + ((y_j-cy)/wy)^2]).
    """
    dx2 = ((x - center[0]) / width[0]) ** 2
    dy2 = ((y - center[1]) / width[1]) ** 2
    return height * np.exp(-0.5 * (dx2[:, None] + dy2[None, :]))


def gaussian2d_grid_over_patch(x, y, height, center, width, nwidth=4):
    """Evaluate a single Gaussian only on an nwidth box around its center.

    Parameters
    ----------
    x, y : array, shape (nx,), (ny,)
        The *full* grid axes (assumed sorted ascending).
    height : float
    center : array, shape (2,)
    width : array, shape (2,)
        Per-axis Gaussian widths.
    nwidth : float, optional
        Patch half-extent in units of ``width``, by default 4.

    Returns
    -------
    xslice, yslice : slice
        Index slices into ``x`` and ``y`` delimiting the patch.
    patch : array, shape (len(x[xslice]), len(y[yslice]))
        The Gaussian on the patch, 'ij' orientation.
    """
    center = np.asarray(center, dtype=float)
    width = np.asarray(width, dtype=float)
    xslice = _slice_to_nwidth(x, center[0], width[0], nwidth)
    yslice = _slice_to_nwidth(y, center[1], width[1], nwidth)
    patch = _gaussian2d_patch(x[xslice], y[yslice], height, center, width)
    return xslice, yslice, patch
