"""Backwards-compatible facade over WeightedGaussians.

``KDE`` is Parrinello's KDE representation of p(s):
Rethinking Metadynamics: From Bias Potentials to Probability Distributions,
J. Phys. Chem. Lett. 2020, 11, 2731.

All of the machinery -- evaluation, analytic norms, the Z_n estimators, the
OPES compression, wsum-reweighted addition -- now lives in
:class:`gaussians.WeightedGaussians`.  This module keeps only the older
call-site spellings (``savez``, the ``.npz`` suffix check, ``__eq__``) so
existing notebooks keep working.

New code should use WeightedGaussians directly.

NB: p(s) is represented by sums of Gaussians.  Represented as grids over
(s1, s2) these have analytical norms ~ 1, but not numerical -- especially
with Silverman's rule of thumb, which gives large widths for eigenfunction
data.  Be careful with functions that check numerical norms
(hist2d.grid_norm).
"""

import numpy as np

from .gaussians import WeightedGaussians


class KDE(WeightedGaussians):
    """WeightedGaussians under its original name.  See the module docstring."""

    # ---- persistence: the original spelling --------------------------
    def savez(self, file):
        """Alias for :meth:`save_npz`, kept for existing call sites."""
        return self.save_npz(file)

    @classmethod
    def from_npz(cls, file):
        """Reconstruct from a .npz written by ``savez``/``save_npz``.

        Reads files written before the type tag existed: ``_from_loaded``
        pulls heights/centers/widths and an optional wsum, and ignores the
        tag entirely.
        """
        if getattr(file, "suffix", ".npz") != ".npz":
            raise ValueError(f"Incorrect file extension {file.suffix}; must be .npz")
        return super().from_npz(file)

    # ---- comparison ---------------------------------------------------
    def __eq__(self, other):
        if not isinstance(other, WeightedGaussians):
            return NotImplemented
        if (
            self.heights.shape != other.heights.shape
            or self.centers.shape != other.centers.shape
            or self.widths.shape != other.widths.shape
        ):
            return False
        if (self.wsum is None) != (other.wsum is None):
            return False
        return (
            np.allclose(self.heights, other.heights)
            and np.allclose(self.centers, other.centers)
            and np.allclose(self.widths, other.widths)
            and (self.wsum is None or np.isclose(self.wsum, other.wsum))
        )

    __hash__ = None  # mutable (renormalize rebinds heights)

    # ---- Parrinello normalization-factor estimators (Z_n) ------------
    def norm_factor_mc(self):
        """Monte-Carlo Z estimate: mean of self evaluated at its own centers.
        2020-parrinello-opes-si, Eq. (S9).
        """
        return float(np.mean(self.evaluate(self.centers)))

    def norm_factor_quad(self, x, y, hist):
        """Quadrature Z estimate over occupied grid cells (hist > 0)."""
        ix, iy = np.where(hist > 0)
        pts = np.column_stack([np.asarray(x)[ix], np.asarray(y)[iy]])
        return float(np.mean(self.evaluate(pts)))



