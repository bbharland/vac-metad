"""Test suite for gaussians.py.

Run from the terminal:
    pytest -q test_gaussians.py

In a Jupyter notebook either shell out (`!pytest -q test_gaussians.py`) or use
ipytest:
    import ipytest; ipytest.autoconfig()
    # ... paste tests into a cell ...
    ipytest.run()

Every test is a plain function with plain asserts, so any one of them can also
be called directly while debugging.
"""
import numpy as np
import numpy.testing as npt
import pytest

from gaussians import Gaussian, Gaussians, WeightedGaussians

TWO_PI = 2 * np.pi


# ----------------------------------------------------------------------
# Fixtures / builders
# ----------------------------------------------------------------------
@pytest.fixture
def simple_set():
    """A small, well-separated 3-kernel set with distinct widths."""
    heights = np.array([1.0, 2.0, 0.5])
    centers = np.array([[-2.0, 0.0], [2.0, 1.0], [0.0, -3.0]])
    widths = np.array([[0.4, 0.4], [0.6, 0.3], [0.5, 0.5]])
    return Gaussians(heights, centers, widths)


def random_set(n, rng, kind="plain"):
    """Build a random Gaussians (or WeightedGaussians) with n kernels."""
    centers = rng.uniform(-5, 5, size=(n, 2))
    widths = rng.uniform(0.1, 0.8, size=(n, 2))
    if kind == "weighted":
        weights = rng.uniform(0.1, 3.0, size=n)
        return WeightedGaussians.from_weights(weights, centers, widths)
    heights = rng.uniform(0.1, 2.0, size=n)
    return Gaussians(heights, centers, widths)


# ======================================================================
# Gaussian (single kernel)
# ======================================================================
class TestGaussian:
    def test_peak_value_is_height(self):
        g = Gaussian(1.7, np.array([1.0, -2.0]), np.array([0.3, 0.5]))
        assert g(g.c) == pytest.approx(1.7)

    def test_distance_is_mahalanobis(self):
        g = Gaussian(1.0, np.array([0.0, 0.0]), np.array([2.0, 4.0]))
        # point one std along each axis -> distance sqrt(2)
        assert g.distance(np.array([2.0, 4.0])) == pytest.approx(np.sqrt(2))

    def test_norm_analytic(self):
        h, w = 1.3, np.array([0.4, 0.7])
        g = Gaussian(h, np.array([0.0, 0.0]), w)
        assert g.norm() == pytest.approx(TWO_PI * h * w.prod())

    def test_merge_identical_kernels(self):
        # merging two identical kernels: height doubles, center/width unchanged
        c, w = np.array([1.0, 2.0]), np.array([0.5, 0.3])
        g1, g2 = Gaussian(1.0, c, w), Gaussian(1.0, c.copy(), w.copy())
        m = g1 + g2
        assert m.h == pytest.approx(2.0)
        npt.assert_allclose(m.c, c)
        npt.assert_allclose(m.w, w)

    def test_merge_matches_moments(self):
        # the merge must match mass, mean and per-axis 2nd moment about origin
        g1 = Gaussian(1.0, np.array([0.0, 1.0]), np.array([0.5, 0.5]))
        g2 = Gaussian(3.0, np.array([2.0, -1.0]), np.array([0.3, 0.8]))
        m = g1 + g2
        M = g1.h + g2.h
        mean = (g1.h * g1.c + g2.h * g2.c) / M
        second = (g1.h * (g1.w ** 2 + g1.c ** 2)
                  + g2.h * (g2.w ** 2 + g2.c ** 2)) / M
        assert m.h == pytest.approx(M)
        npt.assert_allclose(m.c, mean)
        npt.assert_allclose(m.w ** 2, second - mean ** 2)

    def test_sum_builtin(self):
        gs = [Gaussian(1.0, np.array([0.0, 0.0]), np.array([1.0, 1.0]))] * 3
        assert sum(gs).h == pytest.approx(3.0)


# ======================================================================
# Gaussians: container + validation
# ======================================================================
class TestContainer:
    def test_len_and_iter(self, simple_set):
        assert len(simple_set) == 3
        assert len(list(simple_set)) == 3
        for (h, c, w) in simple_set:
            assert np.isscalar(h) or h.ndim == 0
            assert c.shape == (2,) and w.shape == (2,)

    def test_getitem_single(self, simple_set):
        h, c, w = simple_set[1]
        assert h == pytest.approx(2.0)
        npt.assert_allclose(c, [2.0, 1.0])

    def test_slice_returns_set(self, simple_set):
        sub = simple_set[1:]
        assert isinstance(sub, Gaussians)
        assert len(sub) == 2

    def test_concatenation(self, simple_set):
        combined = simple_set + simple_set
        assert len(combined) == 6
        npt.assert_allclose(combined.heights[:3], simple_set.heights)

    def test_radd_supports_sum_and_none(self, simple_set):
        assert (None + simple_set) is simple_set
        assert (0 + simple_set) is simple_set
        total = sum([simple_set, simple_set])  # seeds with int 0
        assert len(total) == 6

    @pytest.mark.parametrize("bad", [
        dict(heights=np.ones((2, 2)), centers=np.zeros((2, 2)), widths=np.ones((2, 2))),
        dict(heights=np.ones(2), centers=np.zeros((2, 3)), widths=np.ones((2, 2))),
        dict(heights=np.ones(3), centers=np.zeros((2, 2)), widths=np.ones((2, 2))),
    ])
    def test_validation_rejects_bad_shapes(self, bad):
        with pytest.raises(ValueError):
            Gaussians(**bad)


# ======================================================================
# Gaussians: evaluation
# ======================================================================
class TestEvaluation:
    def test_call_single_point_matches_formula(self, simple_set):
        s = np.array([0.5, -0.5])
        expected = np.sum(
            simple_set.heights
            * np.exp(-0.5 * np.sum(((s - simple_set.centers)
                                    / simple_set.widths) ** 2, axis=1))
        )
        assert simple_set(s) == pytest.approx(expected)

    def test_call_single_vs_batch(self, simple_set):
        pts = np.array([[0.5, -0.5], [1.0, 1.0], [-1.0, -1.0]])
        batch = simple_set.evaluate(pts)
        for i, p in enumerate(pts):
            assert simple_set(p) == pytest.approx(batch[i])

    def test_chunking_is_transparent(self):
        rng = np.random.default_rng(1)
        g = random_set(40, rng)
        pts = rng.uniform(-5, 5, size=(137, 2))
        npt.assert_allclose(g.evaluate(pts), g.evaluate(pts, chunk_size=10))

    def test_grid_by_row_matches_meshgrid(self, simple_set):
        x = np.linspace(-4, 4, 23)
        y = np.linspace(-5, 3, 17)
        npt.assert_allclose(
            simple_set.evaluate_grid(x, y, by_row=True),
            simple_set.evaluate_grid(x, y, by_row=False),
        )

    def test_grid_orientation(self, simple_set):
        # F[j, i] must equal f((x_i, y_j))
        x = np.linspace(-4, 4, 11)
        y = np.linspace(-5, 3, 9)
        F = simple_set.evaluate_grid(x, y)
        assert F.shape == (len(y), len(x))
        for (i, j) in [(0, 0), (5, 3), (10, 8)]:
            assert F[j, i] == pytest.approx(simple_set(np.array([x[i], y[j]])))


# ======================================================================
# Gaussians: norms
# ======================================================================
class TestNorms:
    def test_norm_is_sum_of_kernel_norms(self, simple_set):
        assert simple_set.norm() == pytest.approx(simple_set.norms_kernels().sum())

    def test_norm_matches_numerical_integral(self):
        # a single kernel comfortably inside the grid
        g = Gaussians(np.array([1.3]), np.array([[0.0, 0.0]]), np.array([[0.4, 0.6]]))
        x = np.linspace(-6, 6, 600)
        y = np.linspace(-6, 6, 600)
        F = g.evaluate_grid(x, y)
        integral = F.sum() * (x[1] - x[0]) * (y[1] - y[0])
        assert integral == pytest.approx(g.norm(), rel=1e-3)

    def test_renormalize_gives_unit_norm(self, simple_set):
        simple_set.renormalize()
        assert simple_set.norm() == pytest.approx(1.0)

    def test_norm_factor_mc(self, simple_set):
        assert simple_set.norm_factor_mc() == pytest.approx(
            np.mean(simple_set.evaluate(simple_set.centers))
        )

    def test_norm_factor_quad(self, simple_set):
        x = np.linspace(-4, 4, 9)
        y = np.linspace(-4, 4, 9)
        hist = np.zeros((len(x), len(y)))
        hist[2, 3] = 1
        hist[6, 5] = 4
        expected = np.mean([
            simple_set(np.array([x[2], y[3]])),
            simple_set(np.array([x[6], y[5]])),
        ])
        assert simple_set.norm_factor_quad(x, y, hist) == pytest.approx(expected)


# ======================================================================
# Compression -- equivalence against the trusted reference
# ======================================================================
def kde_compression_reference(heights, centers, widths, dist_threshold=1.0):
    """Trusted greedy compression, returning (heights, centers, widths).

    This mirrors the long-trusted implementation; it depends only on
    gaussians.Gaussian (same merge + Mahalanobis distance) so it can serve as
    an independent oracle for Gaussians.compressed.

    Reference: Supplementary Information for M. Invernizzi, P. M. Piaggi, and
    M. Parrinello, "Unified Approach to Enhanced Sampling", Phys. Rev. X 10,
    041034 (2020).
    """
    gaussians = []
    for h, c, w in zip(heights, centers, widths):
        gn = Gaussian(h, c, w)
        keep_merging = True
        while keep_merging:
            if len(gaussians) == 0:
                gaussians.append(gn)
                keep_merging = False
            else:
                dists = [g.distance(gn.c) for g in gaussians]
                idx = np.argmin(dists)
                if dists[idx] > dist_threshold:
                    gaussians.append(gn)
                    keep_merging = False
                else:
                    gn += gaussians[idx]
                    del gaussians[idx]
    return (np.array([g.h for g in gaussians]),
            np.vstack([g.c for g in gaussians]),
            np.vstack([g.w for g in gaussians]))


class TestCompression:
    @pytest.mark.parametrize("seed", [0, 1, 2, 7])
    @pytest.mark.parametrize("threshold", [0.5, 1.0, 2.5])
    def test_matches_reference(self, seed, threshold):
        rng = np.random.default_rng(seed)
        g = random_set(60, rng)
        out = g.compressed(dist_threshold=threshold, loud=False)
        ref_h, ref_c, ref_w = kde_compression_reference(
            g.heights, g.centers, g.widths, dist_threshold=threshold
        )
        npt.assert_allclose(out.heights, ref_h)
        npt.assert_allclose(out.centers, ref_c)
        npt.assert_allclose(out.widths, ref_w)

    def test_conserves_total_height(self):
        # every merge does h = h1 + h2, so sum of heights is invariant
        rng = np.random.default_rng(3)
        g = random_set(80, rng)
        out = g.compressed(dist_threshold=1.5, loud=False)
        assert out.heights.sum() == pytest.approx(g.heights.sum())
        assert len(out) <= len(g)

    def test_tiny_threshold_keeps_all_kernels(self):
        rng = np.random.default_rng(4)
        g = random_set(30, rng)  # distinct centers
        out = g.compressed(dist_threshold=1e-9, loud=False)
        assert len(out) == len(g)
        npt.assert_allclose(np.sort(out.heights), np.sort(g.heights))

    def test_huge_threshold_collapses_to_one(self, simple_set):
        out = simple_set.compressed(dist_threshold=1e6, loud=False)
        assert len(out) == 1
        assert out.heights[0] == pytest.approx(simple_set.heights.sum())


# ======================================================================
# WeightedGaussians
# ======================================================================
class TestWeighted:
    def test_from_weights_height_bridge(self):
        weights = np.array([1.0, 3.0, 2.0])
        centers = np.array([[0.0, 0.0], [1.0, 1.0], [-1.0, 2.0]])
        widths = np.array([[0.3, 0.5], [0.4, 0.4], [0.6, 0.2]])
        wg = WeightedGaussians.from_weights(weights, centers, widths)
        W = weights.sum()
        expected_h = weights / (W * TWO_PI * np.prod(widths, axis=1))
        npt.assert_allclose(wg.heights, expected_h)

    def test_weights_roundtrip(self):
        # .weights must recover the input weights exactly (uncompressed)
        rng = np.random.default_rng(5)
        weights = rng.uniform(0.1, 4.0, size=12)
        centers = rng.uniform(-3, 3, size=(12, 2))
        widths = rng.uniform(0.1, 0.7, size=(12, 2))
        wg = WeightedGaussians.from_weights(weights, centers, widths)
        npt.assert_allclose(wg.weights, weights)

    def test_normalized_density(self):
        # a KDE built from weights integrates (analytically) to 1
        wg = random_set(20, np.random.default_rng(6), kind="weighted")
        assert wg.norm() == pytest.approx(1.0)
        assert wg.mixing_weights.sum() == pytest.approx(1.0)

    def test_effective_sample_size_equal_weights(self):
        n = 15
        centers = np.random.default_rng(7).uniform(-3, 3, size=(n, 2))
        widths = np.full((n, 2), 0.5)
        wg = WeightedGaussians.from_weights(np.ones(n), centers, widths)
        assert wg.effective_sample_size() == pytest.approx(n)

    def test_call_equals_gmm_density(self):
        # WeightedGaussians.__call__ == sum_k pi_k N_k(s)
        wg = random_set(8, np.random.default_rng(8), kind="weighted")
        s = np.array([0.3, -0.7])
        pi = wg.mixing_weights
        normal = (1.0 / (TWO_PI * np.prod(wg.widths, axis=1))) * np.exp(
            -0.5 * np.sum(((s - wg.centers) / wg.widths) ** 2, axis=1)
        )
        assert wg(s) == pytest.approx(np.sum(pi * normal))


class TestWeightedAddition:
    def _two(self, seed=9):
        rng = np.random.default_rng(seed)
        return (random_set(5, rng, kind="weighted"),
                random_set(7, rng, kind="weighted"))

    def test_wsum_adds(self):
        a, b = self._two()
        assert (a + b).wsum == pytest.approx(a.wsum + b.wsum)

    def test_pooled_density_is_weighted_average(self):
        a, b = self._two()
        pooled = a + b
        W = a.wsum + b.wsum
        for s in [np.array([0.0, 0.0]), np.array([1.5, -2.0]), np.array([-3.0, 1.0])]:
            expected = (a.wsum * a(s) + b.wsum * b(s)) / W
            assert pooled(s) == pytest.approx(expected)

    def test_pooled_stays_normalized(self):
        a, b = self._two()
        assert (a + b).norm() == pytest.approx(1.0)

    def test_radd_none_and_zero_and_sum(self):
        a, b = self._two()
        assert (None + a) is a
        assert (0 + a) is a
        assert sum([a, b]).wsum == pytest.approx(a.wsum + b.wsum)

    def test_requires_wsum(self):
        a = WeightedGaussians(np.array([1.0]), np.array([[0.0, 0.0]]),
                              np.array([[0.5, 0.5]]), wsum=None)
        with pytest.raises(ValueError):
            a + a


class TestWeightedCompress:
    def test_returns_weighted_and_keeps_wsum(self):
        wg = random_set(40, np.random.default_rng(10), kind="weighted")
        out = wg.compressed(dist_threshold=1.5, loud=False)
        assert isinstance(out, WeightedGaussians)
        assert out.wsum == pytest.approx(wg.wsum)


# ======================================================================
# Sampling (KDE viewed as a GMM)
# ======================================================================
class TestSampling:
    def test_shapes(self):
        wg = random_set(6, np.random.default_rng(11), kind="weighted")
        rng = np.random.default_rng(0)
        assert wg.random(rng=rng).shape == (2,)
        assert wg.random_batch(100, rng=rng).shape == (100, 2)

    def test_reproducible_with_seed(self):
        wg = random_set(6, np.random.default_rng(12), kind="weighted")
        s1 = wg.random_batch(500, rng=np.random.default_rng(123))
        s2 = wg.random_batch(500, rng=np.random.default_rng(123))
        npt.assert_array_equal(s1, s2)

    def test_kernel_selection_frequencies(self):
        wg = random_set(5, np.random.default_rng(13), kind="weighted")
        rng = np.random.default_rng(1)
        idx = wg._select_kernels(200_000, rng)
        freq = np.bincount(idx, minlength=len(wg)) / len(idx)
        npt.assert_allclose(freq, wg.mixing_weights, atol=5e-3)

    def test_sample_mean_matches_mixture_mean(self):
        wg = random_set(5, np.random.default_rng(14), kind="weighted")
        S = wg.random_batch(300_000, rng=np.random.default_rng(2))
        pi = wg.mixing_weights / wg.mixing_weights.sum()
        expected_mean = (pi[:, None] * wg.centers).sum(axis=0)
        npt.assert_allclose(S.mean(axis=0), expected_mean, atol=2e-2)

    def test_sample_density_matches_evaluate(self):
        # 2D histogram of samples should track the analytic density
        wg = random_set(4, np.random.default_rng(15), kind="weighted")
        S = wg.random_batch(400_000, rng=np.random.default_rng(3))
        x = np.linspace(-6, 6, 60)
        y = np.linspace(-6, 6, 60)
        H, _, _ = np.histogram2d(S[:, 0], S[:, 1], bins=[x, y], density=True)
        xc, yc = 0.5 * (x[:-1] + x[1:]), 0.5 * (y[:-1] + y[1:])
        F = wg.evaluate_grid(xc, yc)            # (len(yc), len(xc))
        corr = np.corrcoef(F.ravel(), H.T.ravel())[0, 1]
        assert corr > 0.99


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
