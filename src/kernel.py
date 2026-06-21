"""
Computation involving Gaussians:

    Single Gaussian:  single point        gaussian2d_pdf
                      many points         gaussian2d_pdf
                      grid                gaussian2d_grid
                      grid over patch     gaussian2d_grid_over_patch

    Sum of Gaussians: single point        gaussians.Gaussians.__call__
                      many points         TO DISCUSS
                      grid                new function importing from grid2d.py

The 2D Gaussian kernel itself (``gaussian2d_pdf``) and grid evaluation of a
single Gaussian (``gaussian2d_grid``).

The generic gridding helpers live in ``grid2d.py``; dataset evaluation lives in
``gaussian2d_dataset.py``.
"""

import numpy as np
from functools import partial

from .grid2d import grid_from_arrays


# ----------------------------
# Core math: single 2D Gaussian
# ----------------------------
def gaussian2d_pdf(points, mean, width, height=1.0):
    """
    Evaluate an (optionally scaled) 2D Gaussian PDF at one or many points.

    Parameters
    ----------
    points : ndarray, shape (..., 2)
        Points where to evaluate the Gaussian.
    mean : array-like, shape (2,)
        Gaussian mean.
    width : array-like, shape (2,)
        Diagonal covariance entries (variance in each dimension).
        (This matches the original usage: cov = diag(width).)
    height : float, optional
        Scale factor multiplying the PDF, default 1.0.

    Returns
    -------
    vals : ndarray, shape points.shape[:-1]
        Gaussian values at each point.
    """
    points = np.asarray(points, dtype=float)
    mean = np.asarray(mean, dtype=float).reshape(2,)
    width = np.asarray(width, dtype=float).reshape(2,)

    # Diagonal covariance PDF:
    # pdf(x) = (2π)^(-d/2) |Σ|^(-1/2) exp(-1/2 (x-μ)^T Σ^{-1} (x-μ))
    # Here Σ = diag(width) so |Σ| = width[0]*width[1], Σ^{-1} = diag(1/width)
    d = points - mean
    inv_width = 1.0 / width
    quad = (d[..., 0] ** 2) * inv_width[0] + (d[..., 1] ** 2) * inv_width[1]
    norm = 1.0 / (2.0 * np.pi * np.sqrt(width[0] * width[1]))
    return height * norm * np.exp(-0.5 * quad)


def _gaussian2d_point(a, b, mean, width, height):
    """
    Scalar evaluation of the Gaussian at a single point ``(a, b)``.

    Defined at module level (rather than as a nested closure) so it can be
    pickled and shipped to multiprocessing workers by ``gaussian2d_grid``.
    """
    return gaussian2d_pdf(
        np.array([a, b], dtype=float), mean=mean, width=width, height=height
    ).item()


def gaussian2d_grid(s1, s2, mean, width, height=1.0, processes=None, by_row=True):
    """
    Evaluate a single 2D Gaussian over a regular grid.

    Parameters
    ----------
    s1 : ndarray, shape (numx,)
    s2 : ndarray, shape (numy,)
    mean, width, height : see gaussian2d_pdf
    processes : int or None
        If None: single-process, fully vectorized evaluation (fast).
        If int: multiprocessing evaluation via grid2d.grid_from_arrays.
    by_row : bool
        Only relevant if processes is not None.

    Returns
    -------
    grid : ndarray, shape (numx, numy)
        Uses 'ij' indexing (math axes): s1 maps to axis 0, s2 to axis 1.
    """
    s1 = np.asarray(s1, dtype=float)
    s2 = np.asarray(s2, dtype=float)

    if processes is None:
        # Fully vectorized grid build.
        xy = np.dstack(np.meshgrid(s1, s2, indexing="ij"))  # (numx, numy, 2)
        return gaussian2d_pdf(xy, mean=mean, width=width, height=height)

    # Multiprocessing path (row/point evaluation). A picklable partial of the
    # module-level helper is required here.
    func = partial(_gaussian2d_point, mean=mean, width=width, height=height)
    return grid_from_arrays(s1, s2, func, processes=processes, by_row=by_row)
