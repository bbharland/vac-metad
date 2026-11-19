"""Test suite for acf_analysis.py (and the helpers it adds to math.py).

Run from the repo root:
    pytest -q vac_metad/test_acf_analysis.py

The exact-toy tests are deterministic.  The Bartlett-draw tests use a fixed
seed and check means against the spread, so they are not flaky.
"""
import numpy as np
import numpy.testing as npt
import pytest
from scipy.special import softmax

from . import acf_analysis as A
from .acf_toy import bartlett_acf_draws, toy_model_acf
from .dataclass import DataClass
from .math import exp_sum, running_trapezoid, weighted_linear_fit

COEFFS = np.array([0.7, 0.1, 0.1, 0.1])
TIMES = np.array([1850.0, 60.0, 26.0, 2.0])
TAU_INT = float(COEFFS @ TIMES)  # 1303.8 ps
TAU_R = float(COEFFS[1:] @ TIMES[1:] / COEFFS[1:].sum())  # 29.33 ps


@pytest.fixture(scope="module")
def toy():
    t = np.arange(20001.0)
    return A.acf_input(toy_model_acf(t, COEFFS, TIMES), frametime=1.0)


@pytest.fixture(scope="module")
def draws():
    T = 15e6  # 15 us at 1 ps
    d, mean, sd = bartlett_acf_draws(
        COEFFS, TIMES, 1.0, T, 12000, 60, rng=np.random.default_rng(7)
    )
    return d, T


# ----------------------------------------------------------------------
# math.py helpers
# ----------------------------------------------------------------------
def test_exp_sum_matches_toy():
    t = np.linspace(0, 5000, 101)
    npt.assert_allclose(exp_sum(t, COEFFS, TIMES), toy_model_acf(t, COEFFS, TIMES).ct)
    assert np.isclose(exp_sum(0.0, COEFFS, TIMES), 1.0)


def test_weighted_linear_fit():
    x = np.linspace(0, 10, 50)
    y = 3.0 * x - 2.0
    s, b, cov = weighted_linear_fit(x, y)
    assert np.isclose(s, 3.0) and np.isclose(b, -2.0)
    s, b, cov = weighted_linear_fit(x, 3.0 * x, sigma=np.full(50, 0.1), through_origin=True)
    assert np.isclose(s, 3.0) and b == 0.0
    assert np.isclose(cov[0, 0], 0.01 / np.sum(x**2))


def test_running_trapezoid():
    x = np.linspace(0, 2, 201)
    npt.assert_allclose(running_trapezoid(x, x[1]), x**2 / 2, atol=1e-12)


# ----------------------------------------------------------------------
# input
# ----------------------------------------------------------------------
def test_acf_input_variants():
    ct = toy_model_acf(np.arange(100.0), COEFFS, TIMES).ct
    a = A.acf_input(ct, 2.0)
    assert a.sigma is None and a.t[1] == 2.0
    b = A.acf_input(DataClass(ct=ct), 1.0, n_frames=1e6)
    assert b.sigma.shape == ct.shape and np.all(b.sigma == b.sigma[0])
    c = A.acf_input(DataClass(ct=ct, ct_err=np.full(100, 0.5)), 1.0, n_frames=1e6)
    assert np.all(c.sigma == 0.5)  # ct_err wins over Bartlett
    d = A.acf_input(ct, 1.0, n_frames=1e6, noise_factor=4.0)
    assert np.isclose(d.sigma[0], 4.0 * b.sigma[0])


def test_bartlett_sigma_single_exponential():
    # sigma^2 = (1 + 2 sum rho^2)/N ~ t1/N for t1 >> dt
    t1, N = 1000.0, 1e6
    ct = np.exp(-np.arange(20000) / t1)
    assert np.isclose(A.bartlett_sigma(ct, N), np.sqrt(t1 / N), rtol=1e-3)


# ----------------------------------------------------------------------
# 1. timescales
# ----------------------------------------------------------------------
@pytest.mark.parametrize("method", ["nonlinear", "log"])
def test_slow_fit_exact_toy(toy, method):
    f = A.fit_slow_exponential(toy, method=method)
    assert np.isclose(f.t1, 1850.0, rtol=1e-6)
    assert np.isclose(f.a1, 0.7, rtol=1e-6)


def test_slow_fit_unit_amplitude():
    acf = A.acf_input(np.exp(-np.arange(3000) / 500.0), 1.0)
    f = A.fit_slow_exponential(acf, t_min=0.0, unit_amplitude=True)
    assert f.a1 == 1.0 and np.isclose(f.t1, 500.0)


def test_acf_timescales_exact_toy(toy):
    ts = A.acf_timescales(toy)
    assert np.isclose(ts.t1, 1850.0) and np.isclose(ts.a1, 0.7)
    assert np.isclose(ts.tau_int, TAU_INT, rtol=1e-4)  # trapezoid error only
    assert ts.tau_int_sokal < ts.tau_int  # Sokal truncation bias (about -3% at c=5)
    assert np.isclose(ts.tau_0, -1.0 / np.log(toy.ct[1]))
    assert ts.tau_0 < ts.tau_int < ts.t1  # tau_0 <= tau_int <= tau_exp


def test_local_timescales_match_softmax(toy):
    lt = A.local_timescales(toy, t_max=5000)
    tau = np.arange(50.0, 4000.0)
    w = softmax(np.log(COEFFS)[:, None] - tau[None] / TIMES[:, None], axis=0)
    exact = 1.0 / (w / TIMES[:, None]).sum(0)
    npt.assert_allclose(lt.tangent[50:4000], exact, rtol=2e-3)  # Savitzky-Golay discretization
    assert np.isclose(lt.chord[1], -1.0 / np.log(toy.ct[1]))
    assert np.all(np.isnan(lt.chord[5001:]))


def test_window_from_noise():
    ct = toy_model_acf(np.arange(12000.0), COEFFS, TIMES).ct
    acf = A.acf_input(ct, 1.0, n_frames=3e6)
    t_min, t_max = A.fit_window(acf)
    s = acf.sigma[0]
    assert ct[int(t_max)] >= 3 * s > ct[int(t_max) + 1]
    assert np.isclose(t_min, 0.1 * t_max)


# ----------------------------------------------------------------------
# 2. multi-exponential fit
# ----------------------------------------------------------------------
def test_multi_exp_recovers_exact_toy(toy):
    m = A.fit_multi_exponential(toy, 4, t_max=6000)
    npt.assert_allclose(m.timescales, TIMES, rtol=1e-3)
    npt.assert_allclose(m.coeffs, COEFFS, rtol=1e-3)
    assert m.cond > 1e3  # ill-conditioned even on exact data


def test_multi_exp_two_terms_blends_fast_modes(toy):
    m = A.fit_multi_exponential(toy, 2, t_max=6000)
    assert np.isclose(m.timescales[0], 1850.0, rtol=0.01)
    assert 30.0 < m.timescales[1] < 50.0  # an "effective t2", not 60 ps
    s = A.fit_multi_exponential(toy, 2, t_max=6000, sum_to_one=True)
    assert np.isclose(s.coeffs.sum(), 1.0)


# ----------------------------------------------------------------------
# 3. slow mode + residual
# ----------------------------------------------------------------------
def test_slow_mode_exact_toy(toy):
    sm = A.slow_mode_analysis(toy)
    assert np.isclose(sm.a1, 0.7) and np.isclose(sm.t1, 1850.0)
    assert np.isclose(sm.tau_R, TAU_R, rtol=0.02)  # Sokal window at c=8: -1.4%
    assert abs(sm.identity_error) < 0.5  # ps, out of 1304
    assert np.isclose(sm.tau_int_identity, TAU_INT, rtol=1e-3)
    npt.assert_allclose(sm.R, toy.ct - 0.7 * np.exp(-toy.t / 1850.0), atol=1e-12)
    assert np.isclose(sm.R[0], 0.3)
    assert np.isclose(sm.signal_ratio, 0.7 / 0.3)
    assert 60 < sm.gap < 66


def test_slow_mode_window_c_converges(toy):
    vals = [A.slow_mode_analysis(toy, c_window_R=c).tau_R for c in (5, 8, 20)]
    assert vals[0] < vals[1] < vals[2] and np.isclose(vals[2], TAU_R, rtol=2e-3)


# ----------------------------------------------------------------------
# Bartlett draws
# ----------------------------------------------------------------------
def test_draws_slow_mode_unbiased_within_spread(draws):
    d, T = draws
    sp = A.spread_over_draws(d, 1.0, A.slow_mode_analysis, n_frames=T)
    n = sp.t1.size
    assert sp.num_failed == 0 and n == 60
    for name, true in [("t1", 1850.0), ("a1", 0.7), ("tau_int", TAU_INT)]:
        mean, sd = getattr(sp, f"{name}_mean"), getattr(sp, f"{name}_sd")
        assert abs(mean - true) < 4 * sd / np.sqrt(n), name
    # the identity holds draw by draw to within a few ps
    assert np.all(np.abs(sp.identity_error) < 40.0)
    # tau_R is resolved only roughly at 15 us
    assert 0.2 < sp.tau_R_sd / sp.tau_R_mean < 1.5


def test_scan_window_end_flat_on_exact_toy(toy):
    sc = A.scan_window_end(toy, [3000, 5000, 8000], t_min=1000)
    npt.assert_allclose(sc.t1, 1850.0, rtol=1e-6)
    npt.assert_allclose(sc.a1, 0.7, rtol=1e-6)


def test_tau_R_flag_when_fast_modes_sub_frame():
    # Parrinello-like: fast mode (10 ps) far below the 100 ps frame spacing
    t = 100.0 * np.arange(4000)
    ct = exp_sum(t, [0.97, 0.03], [28600.0, 10.0])
    sm = A.slow_mode_analysis(A.acf_input(ct, 100.0))
    assert sm.tau_R_resolved == 0.0 and np.isclose(sm.tau_R, 50.0, rtol=0.01)
    ct = exp_sum(2.0 * np.arange(60000), [0.97, 0.03], [28600.0, 10.0])
    sm = A.slow_mode_analysis(A.acf_input(ct, 2.0))
    assert sm.tau_R_resolved == 1.0 and np.isclose(sm.tau_R, 10.0, rtol=0.02)


def test_block_errors_with_zero_sigma_at_lag0(toy):
    err = np.full(toy.ct.size, 0.01)
    err[0] = 0.0  # as from block ACFs
    acf = A.acf_input(DataClass(ct=toy.ct, ct_err=err), 1.0)
    assert np.all(acf.sigma > 0)
    m = A.fit_multi_exponential(acf, 2, t_max=6000)
    assert np.isclose(m.timescales[0], 1850.0, rtol=0.01)


def test_split_window_exact_and_draws(toy, draws):
    sm = A.slow_mode_analysis(toy, split_window=(300, 1200))
    assert np.isclose(sm.tau_R, TAU_R, rtol=0.02) and sm.R_window == 300.0
    assert np.isclose(sm.a1_tail, 0.7)
    d, T = draws
    plain = A.spread_over_draws(d, 1.0, A.slow_mode_analysis, n_frames=T)
    split = A.spread_over_draws(d, 1.0, A.slow_mode_analysis, n_frames=T,
                                split_window=(300, 1200))
    assert split.tau_R_sd < 0.5 * plain.tau_R_sd
    assert abs(split.tau_R_mean - TAU_R) < 4 * split.tau_R_sd / np.sqrt(60)
