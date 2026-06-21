"""Gaussian-sum machinery, numpy only.

See gaussians_math.pdf

Layering
--------
Gaussian          : a single kernel; merge + Mahalanobis distance (used by compress).
Gaussians         : a set of kernels; the evaluation engine + pure geometry
                    (evaluate at point/dataset/grid, norms, compress, ...).
                    Primary data is ``heights`` -- this serves a Metadynamics
                    bias directly and is the substrate everything else builds on.
WeightedGaussians : Gaussians + ``wsum``.  ``heights`` stays primary; the
                    statistical ``weights`` are exposed as a *derived* property
                    (weight_k = wsum * kernel_norm_k).  Addition reweights by the
                    wsum ratio so the combined estimate stays normalized.

The torch/TorchScript bias module is deliberately NOT in this hierarchy -- it
gets built from arrays via a separate bridge.
"""
import numpy as np
from functools import partial

from grid2d import grid_from_arrays

try:                                      # progress bar is optional
    from tqdm.auto import tqdm
except ImportError:                       # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable


def _sum_at_points(points, heights, centers, widths):
    """Sum of unnormalized 2D Gaussians at ``points`` (M, 2) -> (M,).

    Single source of truth for the evaluation math.  Kept at module level (no
    ``self``) so that it -- and the thin ``_gaussians_eval_point`` adapter below
    -- are picklable and can be shipped to multiprocessing workers.  Convention:
    ``widths`` are per-axis widths; kernels are unnormalized (peak = height).
    """
    points = np.asarray(points, dtype=float)
    diff = (points[:, None, :] - centers[None, :, :]) / widths[None, :, :]
    norm_sq = np.sum(diff ** 2, axis=2)
    return np.sum(heights[None, :] * np.exp(-0.5 * norm_sq), axis=1)


def _gaussians_eval_point(x, y, heights, centers, widths):
    """Sum-of-Gaussians value at the single point (a, b) -> float.

    Picklable adapter over ``_sum_at_points`` for grid2d's scalar-func grid path.
    """
    point = np.array([[x, y]], dtype=float)
    return float(_sum_at_points(point, heights, centers, widths)[0])


# ======================================================================
# Single kernel -- only needed for the greedy merge inside compress()
# ======================================================================
class Gaussian:
    """A single 2D Gaussian: scalar height, center (d,), width (d,) [diagonal].

    Supports moment-matching merge via ``+`` and Mahalanobis distance, which is
    all ``Gaussians.compress`` needs.
    """
    def __init__(self, height, center, width):
        self.h = height            # float
        self.c = np.asarray(center)  # (d,)
        self.w = np.asarray(width)   # (d,)

    def __repr__(self):
        return f"Gaussian({self.h}, {self.c}, {self.w})"

    def __str__(self):
        def fmt(a):
            return ", ".join(f"{v:.4f}" for v in a)

        return (
            f"Gaussian with height {self.h:.4f}, "
            f"center [{fmt(self.c)}], widths [{fmt(self.w)}]"
        )

    def __call__(self, s):
        return self.h * np.exp(-0.5 * self.distance(s) ** 2)

    def __add__(self, other):
        """Merge two kernels: sum the heights and match the height-weighted
        mean and second moment per dimension (the Parrinello/OPES merge).

        NB: height-weighting reproduces the true (mass-weighted) moments of the
        merged density only when the two kernels share the same width-product
        prod(w); otherwise it is an approximation, and the merge never conserves
        the analytical norm.  See ``Gaussians.compressed``.
        """
        height = self.h + other.h
        center = (self.h * self.c + other.h * other.c) / height
        ws = self.h * (self.w ** 2 + self.c ** 2)
        wo = other.h * (other.w ** 2 + other.c ** 2)
        width = np.sqrt((ws + wo) / height - center ** 2)
        return Gaussian(height, center, width)

    def __radd__(self, other):
        # lets sum([...]) work: the seed is the int 0
        return self if other == 0 else self + other

    def distance(self, s):
        """Mahalanobis distance from this kernel's center to point ``s``."""
        return np.sqrt(np.sum(((s - self.c) / self.w) ** 2))

    def norm(self):
        return (2 * np.pi) ** (len(self.c) / 2) * self.h * self.w.prod()


# ======================================================================
# Set of kernels -- the evaluation engine
# ======================================================================
class Gaussians:
    """A set of 2D Gaussians with parameters heights (N,), centers (N, 2),
    widths (N, 2) [diagonal].

    Evaluation:
        g(s)                      single point  -> float
        g.evaluate(points)        dataset       -> (M,)
        g.evaluate_grid(x, y)     regular grid  -> (len(y), len(x))

    Plus pure geometry shared by KDE and Metadynamics: norm, norms_kernels,
    renormalize, the two Parrinello norm-factor estimators, and compress.
    """

    def __init__(self, heights, centers, widths):
        self.heights = np.asarray(heights)
        self.centers = np.asarray(centers)
        self.widths = np.asarray(widths)
        self._validate()

    def _validate(self):
        if self.heights.ndim != 1:
            raise ValueError(f"heights must be (N,). Got {self.heights.shape}")
        if self.centers.ndim != 2 or self.centers.shape[1] != 2:
            raise ValueError(f"centers must be (N, 2). Got {self.centers.shape}")
        if self.widths.ndim != 2 or self.widths.shape[1] != 2:
            raise ValueError(f"widths must be (N, 2). Got {self.widths.shape}")
        n = len(self.centers)
        if len(self.heights) != n or len(self.widths) != n:
            raise ValueError("heights, centers, widths must share length N")

    # ---- container protocol ------------------------------------------
    def __len__(self):
        return len(self.centers)

    def __iter__(self):
        yield from zip(self.heights, self.centers, self.widths)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return type(self)._from_arrays(
                self, self.heights[idx], self.centers[idx], self.widths[idx]
            )
        return self.heights[idx], self.centers[idx], self.widths[idx]

    def __add__(self, other):
        """Concatenate two sets (raw).  WeightedGaussians refines this."""
        if not isinstance(other, Gaussians):
            return NotImplemented
        return Gaussians(
            np.concatenate([self.heights, other.heights]),
            np.vstack([self.centers, other.centers]),
            np.vstack([self.widths, other.widths]),
        )

    def __radd__(self, other):
        return self if (other is None or other == 0) else self.__add__(other)

    def __repr__(self):
        return f"{type(self).__name__}(n={len(self)})"

    # internal: rebuild same subclass, preserving extra state where sensible
    @staticmethod
    def _from_arrays(template, heights, centers, widths):
        return Gaussians(heights, centers, widths)

    @property
    def dim(self):
        return self.centers.shape[1]

    # ---- evaluation ---------------------------------------------------
    def _eval_points(self, points):
        """points: (M, d) -> (M,).  Builds an (M, N, d) temporary."""
        return _sum_at_points(points, self.heights, self.centers, self.widths)

    def __call__(self, s):
        """Single point s:(d,) -> float; or a batch s:(M, d) -> (M,)."""
        s = np.asarray(s)
        if s.ndim == 1:
            return _gaussians_eval_point(
                s[0], s[1], self.heights, self.centers, self.widths
            )
        return self.evaluate(s)

    def evaluate(self, points, chunk_size=None):
        """Evaluate over a dataset points:(M, d) -> (M,).

        chunk_size bounds the (chunk, N, d) temporary for large M.
        """
        points = np.asarray(points)
        if chunk_size is None:
            return self._eval_points(points)
        out = np.empty(len(points))
        for i in range(0, len(points), chunk_size):
            out[i : i + chunk_size] = self._eval_points(points[i : i + chunk_size])
        return out

    def evaluate_grid(self, xgrid, ygrid, by_row=True, processes=None):
        """Evaluate over the regular grid (xgrid x ygrid) -> (len(ygrid), len(xgrid)).

        Orientation matches imshow/contourf: rows index y, columns index x.
        by_row keeps memory at (len(xgrid), N) instead of (nx*ny, N).

        processes : int or None
            None -> single-process, vectorized over kernels (default, fast).
            int  -> distribute the grid across worker processes via
                    grid2d.grid_from_arrays.  Note this path evaluates the
                    grid point-by-point (it loses the vectorization over the N
                    kernels and pickles the kernel arrays to the workers), so it
                    is only worth it for heavy per-point work / very large grids.
        """
        xgrid, ygrid = np.asarray(xgrid), np.asarray(ygrid)

        if processes is None:
            if by_row:
                out = np.empty((len(ygrid), len(xgrid)))
                for j, y in enumerate(ygrid):
                    row = np.column_stack([xgrid, np.full(len(xgrid), y)])
                    out[j] = self._eval_points(row)
                return out
            X, Y = np.meshgrid(xgrid, ygrid)
            pts = np.column_stack([X.ravel(), Y.ravel()])
            return self._eval_points(pts).reshape(X.shape)

        # Multiprocessing path via grid2d.  grid_from_arrays returns
        # z[i, j] = f(xgrid[i], ygrid[j]) with shape (nx, ny); transpose to the
        # (ny, nx) imshow orientation this method documents.
        func = partial(
            _gaussians_eval_point,
            heights=self.heights, centers=self.centers, widths=self.widths,
        )
        z = grid_from_arrays(xgrid, ygrid, func, processes=processes, by_row=by_row)
        return np.asarray(z).T

    # ---- analytical norms --------------------------------------------
    def norms_kernels(self):
        """Per-kernel analytical norm: (2pi)^(d/2) * h_k * prod(w_k)."""
        return (
            (2 * np.pi) ** (self.dim / 2) * self.heights * np.prod(self.widths, axis=1)
        )

    def norm(self):
        """Total analytical norm = sum of per-kernel norms."""
        return float(np.sum(self.norms_kernels()))

    def renormalize(self):
        """Scale heights so the analytical norm is 1.  In place; returns self."""
        self.heights = self.heights / self.norm()
        return self

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

    # ---- compression (greedy moment-matching merge) ------------------
    def compressed(self, dist_threshold=1.0, loud=True):
        """Return a new Gaussians with nearby kernels merged.

        Greedy and inherently sequential; O(N * surviving_kernels).  Each merge
        is the OPES height-weighted moment match (see ``Gaussian.__add__``):
        exact for the mean/covariance only while merged kernels share a
        width-product, and it never conserves the analytical norm -- so callers
        must ``renormalize()`` afterward to restore norm() == 1.

        Reference: Supplementary Information for M. Invernizzi, P. M. Piaggi,
        and M. Parrinello, "Unified Approach to Enhanced Sampling",
        Phys. Rev. X 10, 041034 (2020).
        https://journals.aps.org/prx/abstract/10.1103/PhysRevX.10.041034
        """
        merged = []
        params = zip(self.heights, self.centers, self.widths)
        if loud:
            params = tqdm(params, total=len(self))

        for h, c, w in params:
            gn = Gaussian(h, c, w)
            while True:
                if not merged:
                    merged.append(gn)
                    break
                dists = [g.distance(gn.c) for g in merged]
                idx = int(np.argmin(dists))
                if dists[idx] > dist_threshold:
                    merged.append(gn)
                    break
                gn = gn + merged.pop(idx)

        return Gaussians(
            np.stack([g.h for g in merged]),
            np.vstack([g.c for g in merged]),
            np.vstack([g.w for g in merged]),
        )

    # ---- persistence (.npz) ------------------------------------------
    # Arrays only -> no scipy/torch/class needed to read the file back, in any
    # language.  A small "type" tag lets load_gaussians() pick the right class.
    def _npz_arrays(self):
        """The named arrays this object serializes.  Subclasses extend this."""
        return dict(heights=self.heights,
                    centers=self.centers,
                    widths=self.widths)

    def save_npz(self, file):
        """Write the kernels to a .npz file (filename, Path, or open file).

        np.savez appends '.npz' to a bare filename if it lacks the extension.
        """
        np.savez(file, type=np.array(type(self).__name__), **self._npz_arrays())

    @classmethod
    def _from_loaded(cls, a):
        """Build an instance from an open NpzFile ``a``."""
        return cls(a["heights"], a["centers"], a["widths"])

    @classmethod
    def from_npz(cls, file):
        """Reconstruct an instance of this class from a .npz written by save_npz.

        Constructs the class it is called on (``Gaussians.from_npz`` -> a plain
        Gaussians, dropping any stored wsum).  Use the module-level
        ``load_gaussians`` to reconstruct whichever type was saved.
        """
        with np.load(file, allow_pickle=False) as a:
            return cls._from_loaded(a)


# ======================================================================
# Weighted set -- KDE substrate
# ======================================================================
class WeightedGaussians(Gaussians):
    """Gaussians + total weight ``wsum``.

    ``heights`` remains primary (so every inherited transform just works);
    the statistical ``weights`` are *derived*:  weight_k = wsum * kernel_norm_k.
    For a freshly built, uncompressed KDE these equal the original frame
    weights exactly; after compress() they are the effective implied weights.

    Build from frame weights with ``from_weights``; build from heights
    directly with the normal constructor.
    """

    def __init__(self, heights, centers, widths, wsum=None):
        super().__init__(heights, centers, widths)
        self.wsum = None if wsum is None else float(wsum)

    @classmethod
    def from_weights(cls, weights, centers, widths, wsum=None):
        """Construct from statistical frame weights.

        height_k = weight_k / (wsum * (2pi)^(d/2) * prod(w_k))
        which is Parrinello's normalized kernel weight.
        """
        weights = np.asarray(weights, dtype=float)
        centers = np.asarray(centers)
        widths = np.asarray(widths)
        if wsum is None:
            wsum = float(np.sum(weights))
        d = centers.shape[1]
        kernel_norm = (2 * np.pi) ** (d / 2) * np.prod(widths, axis=1)
        heights = weights / (wsum * kernel_norm)
        return cls(heights, centers, widths, wsum=wsum)

    @staticmethod
    def _from_arrays(template, heights, centers, widths):
        # preserve wsum on slicing (note: slicing changes the implied weights)
        return WeightedGaussians(heights, centers, widths, wsum=template.wsum)

    # ---- derived statistical quantities ------------------------------
    @property
    def weights(self):
        """Per-kernel statistical weight implied by the current state."""
        if self.wsum is None:
            return self.norms_kernels()  # mixing proportions, sum -> norm()
        return self.wsum * self.norms_kernels()

    @property
    def mixing_weights(self):
        """GMM mixing proportions pi_k (sum to norm(), ~1 when normalized)."""
        return self.norms_kernels()

    def effective_sample_size(self):
        w = self.weights
        return float(np.sum(w) ** 2 / np.sum(w ** 2))

    # ---- sampling: the KDE viewed as a GMM ---------------------------
    # p(s) = sum_k pi_k N_k(s) with mixing proportions pi_k = mixing_weights.
    # Each kernel has diagonal covariance Sigma_k = diag(width_k ** 2), so a draw
    # is just
    #     s = c_k + width_k * z,   z ~ N(0, I_d),
    # i.e. no scipy and no per-kernel multivariate_normal objects.
    def _select_kernels(self, size, rng):
        """Draw `size` kernel indices in proportion to pi_k."""
        p = self.norms_kernels()
        return rng.choice(len(self), size=size, p=p / np.sum(p))

    def random(self, rng=None):
        """Draw a single sample s ~ p(s).  Returns shape (d,)."""
        rng = np.random if rng is None else rng
        k = int(self._select_kernels(1, rng)[0])
        return self.centers[k] + self.widths[k] * rng.standard_normal(self.dim)

    def random_batch(self, batch_size, shuffle=False, rng=None):
        """Draw `batch_size` samples s ~ p(s).  Returns shape (batch_size, d).

        Pick one kernel per sample (with prob pi_k), then draw one
        diagonal-Gaussian point from each.  Rows already come out in random
        kernel order, so `shuffle` is rarely needed -- kept for API parity.

        rng : np.random.Generator (or the np.random module).  Pass a seeded
              Generator for reproducible trajectories.
        """
        rng = np.random if rng is None else rng
        idx = self._select_kernels(batch_size, rng)
        z = rng.standard_normal((batch_size, self.dim))
        samples = self.centers[idx] + self.widths[idx] * z
        if shuffle:
            rng.shuffle(samples)
        return samples

    # ---- addition: refine base concat with wsum reweighting ----------
    def __add__(self, other):
        """Combine two weighted estimates.  Concatenates kernels AND rescales
        heights by the wsum ratio, so the result stays a normalized estimate
        with combined wsum.  (This is why it overrides the base concat.)
        """
        if not isinstance(other, WeightedGaussians):
            return NotImplemented
        if self.wsum is None or other.wsum is None:
            raise ValueError("both operands need a wsum to add")
        wsum = self.wsum + other.wsum
        heights = np.concatenate([
            self.wsum / wsum * self.heights,
            other.wsum / wsum * other.heights,
        ])
        return WeightedGaussians(
            heights,
            np.vstack([self.centers, other.centers]),
            np.vstack([self.widths, other.widths]),
            wsum=wsum,
        )

    def __radd__(self, other):
        # None + wg (iterating frames) and 0 + wg (sum())
        return self if (other is None or other == 0) else self.__add__(other)

    # ---- compressed: keep it a WeightedGaussians, carry wsum forward ----
    def compressed(self, dist_threshold=1.0, loud=True):
        g = super().compressed(dist_threshold=dist_threshold, loud=loud)
        return WeightedGaussians(g.heights, g.centers, g.widths, wsum=self.wsum)

    # ---- persistence: also carry wsum (omitted from the file if None) ----
    def _npz_arrays(self):
        d = super()._npz_arrays()
        if self.wsum is not None:
            d["wsum"] = np.array(self.wsum)
        return d

    @classmethod
    def _from_loaded(cls, a):
        wsum = float(a["wsum"]) if "wsum" in a.files else None
        return cls(a["heights"], a["centers"], a["widths"], wsum=wsum)

    def __repr__(self):
        return f"WeightedGaussians(n={len(self)}, wsum={self.wsum})"


# ======================================================================
# Polymorphic loader
# ======================================================================
_NPZ_REGISTRY = {
    "Gaussians": Gaussians,
    "WeightedGaussians": WeightedGaussians,
}


def load_gaussians(file):
    """Load a Gaussians or WeightedGaussians from a .npz, dispatching on the
    stored type tag.  At the call site you don't need to know which subclass
    was saved:

        g = load_gaussians("run1.npz")

    Falls back to Gaussians for files with no (or an unknown) type tag.
    """
    with np.load(file, allow_pickle=False) as a:
        tag = str(a["type"]) if "type" in a.files else "Gaussians"
        cls = _NPZ_REGISTRY.get(tag, Gaussians)
        return cls._from_loaded(a)
