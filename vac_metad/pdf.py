"""Probability densities for quick views of a dataset.

Pure functions: arrays in, DataClass out.  Saving, loading and the decision of
whether to recompute are left to the caller, e.g.

    compute_pdf = False
    if compute_pdf:
        pdf = cvs_pdf(sd.cvs)
        sd.save_and_assign_objects({"cvs_pdf": pdf})

Densities are normalized: the integral over the grid is 1.  Grid points are
bin CENTERS throughout, matching ``grid2d.hist2d``.
"""

import numpy as np

from .dataclass import DataClass
from .grid2d import (
    grid1d_from_range,
    hist2d,
    periodic_grid,
    periodic_index,
    ranges_data,
)


# --------------------------------------------------------------------------- #
#  Dihedrals (periodic)
# --------------------------------------------------------------------------- #
def dihedrals_pdf(dihedrals, num_points=100, weights=None):
    """2D density of (phi, psi) on a periodic grid; integrates to 1.

    Binned with :func:`grid2d.periodic_index`, the same rule
    ``data.feature_grid_over_dihedrals`` uses, so a frame lands in the same
    cell in both objects, and phi = pi and phi = -pi share a bin.

    Parameters
    ----------
    dihedrals : array, shape (num_frames, 2)
        (phi, psi) in radians.
    num_points : int
        Bins per axis.  ``x`` / ``y`` are bin centers whose edges fall on +/- pi.
    weights : array, shape (num_frames,), or None
        Frame weights (e.g. reweighting factors for biased simulations).

    Returns
    -------
    DataClass with x, y (bin centers) and p, shape (num_points, num_points),
    where ``p[i, j]`` is the density at ``(x[i], y[j])``.
    """
    theta = periodic_grid(num_points)
    i = periodic_index(dihedrals[:, 0], num_points)
    j = periodic_index(dihedrals[:, 1], num_points)
    counts = np.bincount(
        i * num_points + j, weights=weights, minlength=num_points**2
    ).reshape(num_points, num_points)

    dtheta = 2 * np.pi / num_points
    return DataClass(x=theta, y=theta, p=counts / (counts.sum() * dtheta**2))


# --------------------------------------------------------------------------- #
#  Collective variables (not periodic; grid from the data)
# --------------------------------------------------------------------------- #
def cv_ranges(cvs, pad=0.10, decimals=1):
    """Padded ``[[min, max], ...]`` per CV, rounded to ``decimals`` places.

    Rounding to one decimal is safe for SRV eigenfunctions specifically: they
    are normalized to zero mean and unit variance, so a padded range spans a
    few units and the rounding error is far smaller than the padding.  Pass
    ``decimals=None`` for raw padded ranges, or supply ``ranges`` directly to
    :func:`cvs_pdf` to share one grid across datasets.
    """
    ranges = np.asarray(ranges_data(cvs, pad=pad), dtype=float)
    return ranges if decimals is None else np.round(ranges, decimals)


def _count_outside(cvs, ranges):
    """Frames falling outside the grid, which the histogram silently drops."""
    lo = np.asarray(ranges)[:, 0]
    hi = np.asarray(ranges)[:, 1]
    return int(np.any((cvs < lo) | (cvs > hi), axis=1).sum())


def cvs_pdf(cvs, num_points=100, weights=None, ranges=None, pad=0.10, decimals=1):
    """Density of the collective variables; 1D or 2D, from ``cvs.shape[1]``.

    Parameters
    ----------
    cvs : array, shape (num_frames, 1) or (num_frames, 2)
        The CV trajectory, i.e. ``sd.cvs``.  One column gives a 1D density,
        two give a 2D one.
    num_points : int
        Grid points per axis; these are bin CENTERS.
    weights : array, shape (num_frames,), or None
        Frame weights (reweighting factors for biased simulations).
    ranges : [[min, max], ...] or None
        Grid limits.  None derives them from the data via :func:`cv_ranges`.
        Pass them explicitly to compare densities across datasets bin by bin.
        Frames outside the grid are dropped, and the density is renormalized
        over what is left, so a too-tight range still integrates to 1 -- hence
        the warning.
    pad, decimals
        Passed to :func:`cv_ranges` when ``ranges`` is None.

    Returns
    -------
    DataClass
        2D: ``x``, ``y`` (bin centers) and ``p`` with shape
        ``(num_points, num_points)``.
        1D: ``x`` and ``p`` only -- there is no ``y`` attribute, since
        DataClass cannot store None through a .npz round trip (savez writes an
        object array, and from_npz loads with allow_pickle=False).  Branch on
        ``"y" in pdf.attrs``.
    """
    cvs = np.asarray(cvs)
    if cvs.ndim != 2 or cvs.shape[1] not in (1, 2):
        raise ValueError(
            f"cvs must have shape (num_frames, 1) or (num_frames, 2); got {cvs.shape}"
        )

    if ranges is None:
        ranges = cv_ranges(cvs, pad=pad, decimals=decimals)

    num_outside = _count_outside(cvs, ranges)
    if num_outside:
        print(
            f"[warn] {num_outside} frame(s) fall outside {np.asarray(ranges).tolist()} "
            f"and are dropped from the density."
        )

    x = grid1d_from_range(ranges[0], num_points=num_points)
    if cvs.shape[1] == 1:
        dx = x[1] - x[0]
        p, _ = np.histogram(
            cvs[:, 0],
            bins=num_points,
            range=(x[0] - dx / 2, x[-1] + dx / 2),
            weights=weights,
            density=True,
        )
        return DataClass(x=x, p=p)

    y = grid1d_from_range(ranges[1], num_points=num_points)
    return DataClass(x=x, y=y, p=hist2d(x, y, cvs, weights=weights, density=True))
