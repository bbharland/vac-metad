"""Probability densities for quick views of a dataset.

Pure functions: arrays in, DataClass out.  Saving and loading is left to
SimulationData, which calls these where the input data is written.
"""

import numpy as np

from .dataclass import DataClass
from .grid2d import periodic_grid, periodic_index


def dihedrals_pdf(dihedrals, num_points=100, weights=None):
    """2D density of (phi, psi) on a periodic grid; integrates to 1.

    Binned with :func:`periodic_index`, the same rule
    ``feature_grid_over_dihedrals`` uses, so a frame lands in the same cell in
    both objects, and phi = pi and phi = -pi are the same bin.

    Parameters
    ----------
    dihedrals : array, shape (num_frames, 2)
        (phi, psi) in radians.
    num_points : int
        Bins per axis.  ``x``/``y`` are bin centers whose edges fall on +/- pi.
    weights : array, shape (num_frames,), or None
        Frame weights (e.g. reweighting factors for biased simulations).

    Returns
    -------
    DataClass with x, y (bin centers) and p, shape (num_points, num_points),
    with p[i, j] ~ density at (x[i], y[j]).
    """
    theta = periodic_grid(num_points)
    i = periodic_index(dihedrals[:, 0], num_points)
    j = periodic_index(dihedrals[:, 1], num_points)
    counts = np.bincount(
        i * num_points + j, weights=weights, minlength=num_points**2
    ).reshape(num_points, num_points)

    dtheta = 2 * np.pi / num_points
    p = counts / (counts.sum() * dtheta**2)
    return DataClass(x=theta, y=theta, p=p)
