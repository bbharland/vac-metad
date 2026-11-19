"""
Analysis of a normalized ACF, C(tau) with C(0) = 1, once it has been computed.

acf.py computes C(tau) from data and acf_toy.py makes model ACFs; this module
turns either one into timescales and signal measures.  Everything takes a
DataClass and returns a DataClass.

MODEL
-----
For a reversible process, an observable u = sum_i c_i psi_i has

        C(tau) = sum_i a_i exp(-tau / t_i),      a_i = c_i^2,  sum_i a_i = 1.

USAGE
-----
    acf = acf_input(ct, frametime, n_frames=N)      # or acf_input(dataclass, ...)
    ts  = acf_timescales(acf)                       # 1. tau_0, tau_int, t_1 (= tau_exp)
    mx  = fit_multi_exponential(acf, n_terms=2)     # 2. the "problematic" analysis
    sm  = slow_mode_analysis(acf)                   # 3. the "sensible" analysis

THE THREE ANALYSES
------------------
1. acf_timescales
   tau_0    one-frame chord, -dt / ln C(dt).  Tends to -1/C'(0) as dt -> 0,
            so it depends on dt; quote dt with it.
   tau_int  integral of C.  Two estimates: Sokal-windowed (as in acf.py) and
            "tail-corrected" (integral up to the noise cutoff t_max plus the
            fitted slow tail a_1 t_1 exp(-t_max/t_1)).
   t_1      slowest timescale present in u, from a fit of the slow tail.
            Sokal's tau_exp IS t_1 (tau_exp,u = max{t_i : a_i > 0}), so there
            is one number, not two.
   a_1      intercept of the same fit.

2. fit_multi_exponential(acf, n_terms)
   Least-squares fit of n_terms exponentials to C on [0, t_max].  This is
   ill-conditioned: the parameter correlation matrix and its condition number
   are returned so that you can see it.  On the toy, n_terms=2 returns one
   blended "effective t_2" (about 40 ps), not 60 ps.

3. slow_mode_analysis
   Split C(t) = a_1 exp(-t/t_1) + R(t),  R(0) = 1 - a_1, and report
       tau_R = (1/(1-a_1)) int_0^oo R dt     weight-averaged fast timescale
       g     = t_1 / tau_R                   spectral gap (timescale ratio)
   and the identity tau_int = a_1 t_1 + (1 - a_1) tau_R.

NOISE AND THE FIT WINDOW
------------------------
The slow tail is fitted on [t_min, t_max].  t_max is set by the noise: the
last lag before C_hat first drops below k_sigma * sigma_C (default 3 sigma),
not a fixed multiple of t_1 (see notes/acf-numerics/acf-logsumexp-review.md).
sigma_C comes from, in order of preference:
    ct_err (block or jackknife errors, e.g. from acf.compute_acf_with_errors),
    or the large-lag Bartlett formula sigma_C^2 = (1 + 2 sum_k C_k^2) / N,
       which needs n_frames.  It assumes Gaussian statistics and is 4-5x too
       small for a two-state u; use noise_factor or ct_err in that case.
With no noise information (exact model ACFs), t_max is the last lag.
t_min defaults to t_min_frac * t_max (0.1), which must lie past the fast
modes.  Use scan_window_end to check that t_1 and a_1 hold steady.

Bartlett noise is smooth and correlated across lags, so every error bar from
a single fit is too optimistic.  The honest error is the spread over
independent ACFs (blocks, replicas, or Bartlett draws): spread_over_draws.
"""

import numpy as np
from scipy.optimize import curve_fit, least_squares

from .acf import compute_theta
from .dataclass import DataClass
from .math import exp_sum, running_trapezoid, weighted_linear_fit

try:  # Ben's local math.py has derivative(t, y, method="savgol", window, polyorder)
    from .math import derivative
except ImportError:  # fallback for a math.py without it
    from scipy.signal import savgol_filter

    def derivative(t, y, method="savgol", window=11, polyorder=2):
        return savgol_filter(y, window, polyorder, deriv=1, delta=t[1] - t[0])


# --------------------------------------------------------------------------- #
#  Input
# --------------------------------------------------------------------------- #
def bartlett_sigma(ct, n_frames):
    """Large-lag Bartlett standard error of C_hat, constant in tau:

        sigma_C^2 = (1 + 2 sum_{k>=1} C_k^2) / n_frames.

    The sum runs up to the first lag where C_hat <= 0, so the noisy tail
    does not add to it.  Gaussian statistics assumed.
    """
    ct = np.asarray(ct, float)
    neg = np.flatnonzero(ct <= 0.0)
    k_end = neg[0] if neg.size else ct.size
    return float(np.sqrt((1.0 + 2.0 * np.sum(ct[1:k_end] ** 2)) / n_frames))


def acf_input(acf, frametime, n_frames=None, ct_err=None, noise_factor=1.0):
    """Package an ACF for the analysis functions.

    Parameters
    ----------
    acf : DataClass with attribute ct (e.g. from acf.compute_acf_dataclass or
        acf_toy.toy_model_acf), or a bare 1D array C(tau) with C(0) = 1.
        If the DataClass also has ct_err, that is used as sigma_C.
    frametime : time between lags (sets the units of every result)
    n_frames : number of frames in the trajectory, N = T / frametime.
        Used for the Bartlett sigma_C when no ct_err is available.
    ct_err : optional 1-sigma error on C(tau), scalar or array.  Zeros (as
        at lag 0 for block errors) are replaced by the smallest nonzero value.
    noise_factor : multiplies the Bartlett sigma_C (e.g. 4-5 for a two-state u)

    Returns
    -------
    DataClass with
        t : lag times, frametime * arange(len(ct))
        ct : C(tau)
        sigma : 1-sigma error on C(tau) per lag, or None (no noise info)
        frametime : float
        n_frames : float or None
    """
    ct = np.asarray(acf.ct if isinstance(acf, DataClass) else acf, float)
    if ct_err is None and isinstance(acf, DataClass):
        ct_err = getattr(acf, "ct_err", None)

    if ct_err is not None:
        sigma = np.broadcast_to(np.asarray(ct_err, float), ct.shape).copy()
        # Block/jackknife errors are 0 at lag 0 (every block has C(0) = 1);
        # floor them so that weighted fits including lag 0 stay finite.
        pos = sigma[sigma > 0.0]
        if pos.size:
            sigma = np.where(sigma > 0.0, sigma, pos.min())
    elif n_frames is not None:
        sigma = np.full(ct.shape, noise_factor * bartlett_sigma(ct, n_frames))
    else:
        sigma = None

    return DataClass(
        t=frametime * np.arange(ct.size),
        ct=ct,
        sigma=sigma,
        frametime=frametime,
        n_frames=n_frames,
    )


def fit_window(acf, t_min=None, t_max=None, k_sigma=3.0, t_min_frac=0.1):
    """Return (t_min, t_max) for the slow-tail fit.

    t_max : given, or the last lag before C_hat first drops below
            k_sigma * sigma (searching from lag 1), or the last lag if
            acf.sigma is None.
    t_min : given, or t_min_frac * t_max.
    """
    t, ct = acf.t, acf.ct
    if t_max is None:
        if acf.sigma is None:
            t_max = t[-1]
        else:
            low = np.flatnonzero(ct[1:] < k_sigma * acf.sigma[1:])
            t_max = t[low[0]] if low.size else t[-1]  # low[0]+1 is the first bad lag
    if t_min is None:
        t_min = t_min_frac * t_max
    if t_min >= t_max:
        raise ValueError(f"empty fit window: t_min={t_min} >= t_max={t_max}")
    return float(t_min), float(t_max)


# --------------------------------------------------------------------------- #
#  Slow tail: a_1, t_1
# --------------------------------------------------------------------------- #
def fit_slow_exponential(
    acf,
    t_min=None,
    t_max=None,
    method="nonlinear",
    unit_amplitude=False,
    k_sigma=3.0,
    t_min_frac=0.1,
):
    """Fit C(t) ~ a_1 exp(-t / t_1) on [t_min, t_max].

    Parameters
    ----------
    acf : DataClass from acf_input
    t_min, t_max, k_sigma, t_min_frac : fit window, see fit_window
    method : 'nonlinear' (default) fits C itself, weighted by 1/sigma^2.
             'log' fits ln C by a straight line, weighted by C^2/sigma^2
             (the consistent weight for ln C).  Points with C <= 0 are
             dropped, which biases it upward in a noisy tail.
    unit_amplitude : force a_1 = 1 (the fit then goes through C(0) = 1)

    Returns
    -------
    DataClass with a1, t1, a1_err, t1_err, t_min, t_max, num_points.
    The errors assume independent noise per lag and are too small for real
    ACFs; use spread_over_draws for honest errors.
    """
    t_min, t_max = fit_window(acf, t_min, t_max, k_sigma, t_min_frac)
    sel = (acf.t >= t_min) & (acf.t <= t_max)
    t, c = acf.t[sel], acf.ct[sel]
    s = None if acf.sigma is None else acf.sigma[sel]

    # log-linear fit (also the starting guess for the nonlinear fit)
    pos = c > 0.0
    if pos.sum() < 2:
        raise ValueError("fewer than 2 positive C values in the fit window")
    s_log = None if s is None else s[pos] / c[pos]
    slope, icpt, cov = weighted_linear_fit(
        t[pos], np.log(c[pos]), s_log, through_origin=unit_amplitude
    )
    k, a = -slope, np.exp(icpt)
    k_err, a_err = np.sqrt(cov[0, 0]), a * np.sqrt(cov[1, 1])

    if method == "nonlinear":
        if unit_amplitude:
            f = lambda t, k: np.exp(-k * t)
            p0 = [k]
        else:
            f = lambda t, a, k: a * np.exp(-k * t)
            p0 = [a, k]
        p, pcov = curve_fit(f, t, c, p0=p0, sigma=s, absolute_sigma=s is not None)
        if unit_amplitude:
            k, k_err = p[0], np.sqrt(pcov[0, 0])
        else:
            (a, k), (a_err, k_err) = p, np.sqrt(np.diag(pcov))
    elif method != "log":
        raise ValueError("method must be 'nonlinear' or 'log'")

    return DataClass(
        a1=a,
        t1=1.0 / k,
        a1_err=a_err,
        t1_err=k_err / k**2,
        t_min=t_min,
        t_max=t_max,
        num_points=int(sel.sum()),
    )


def scan_window_end(acf, t_maxs, t_min=None, **fit_kwargs):
    """Refit the slow tail for each window end in t_maxs (Frisken's check).

    t_1 and a_1 should be flat in t_max where the fit is trustworthy and drift
    once the window reaches into the noise.  t_min is held fixed (default
    t_min_frac * min(t_maxs)).

    Returns DataClass with t_max, a1, t1, a1_err, t1_err (arrays).
    """
    t_maxs = np.asarray(t_maxs, float)
    if t_min is None:
        t_min = fit_kwargs.pop("t_min_frac", 0.1) * t_maxs.min()
    fits = [fit_slow_exponential(acf, t_min, tm, **fit_kwargs) for tm in t_maxs]
    return DataClass(
        t_max=t_maxs,
        a1=np.array([f.a1 for f in fits]),
        t1=np.array([f.t1 for f in fits]),
        a1_err=np.array([f.a1_err for f in fits]),
        t1_err=np.array([f.t1_err for f in fits]),
    )


# --------------------------------------------------------------------------- #
#  1. Timescales
# --------------------------------------------------------------------------- #
def sokal_window(running, frametime, c_window):
    """Index M of the smallest window (M >= 1) with t_M >= c_window * I(t_M),
    where running[k] = I(k * frametime) is a running integral, I(0) = 0.
    Falls back to the last lag, with a warning, if the window never closes.
    """
    m = np.arange(running.size)
    ok = (m >= 1) & (m * frametime >= c_window * running)
    if not ok.any():
        print("[warn] Sokal window never closed; using the full series.")
        return running.size - 1
    return int(np.argmax(ok))


def sokal_tau_int(acf, c_window=5.0):
    """Sokal-windowed integrated time (trapezoid rule).

    Smallest window M with t_M >= c_window * tau_int(M).  Agrees with
    acf.compute_acf to within C(M) * frametime / 2.

    Returns (tau_int, t_window).
    """
    running = running_trapezoid(acf.ct, acf.frametime)
    M = sokal_window(running, acf.frametime, c_window)
    return float(running[M]), float(acf.t[M])


def tail_corrected_tau_int(acf, a1, t1, t_max):
    """tau_int = int_0^{t_max} C_hat dt + a_1 t_1 exp(-t_max / t_1).

    Integrates the data up to the noise cutoff and adds the fitted slow tail
    analytically, so there is neither a noisy tail nor a truncation bias.
    """
    i = int(round(t_max / acf.frametime))
    running = running_trapezoid(acf.ct[: i + 1], acf.frametime)
    return float(running[i] + a1 * t1 * np.exp(-t_max / t1))


def acf_timescales(acf, c_window=5.0, **fit_kwargs):
    """The main timescales of an ACF.

    Parameters
    ----------
    acf : DataClass from acf_input
    c_window : Sokal window constant for tau_int_sokal
    fit_kwargs : passed to fit_slow_exponential (t_min, t_max, method, ...)

    Returns
    -------
    DataClass with
        tau_0 : one-frame chord -dt / ln C(dt); approximates -1/C'(0) only
            for dt much shorter than the fastest mode
        tau_int_sokal : Sokal-windowed integral of C (as in acf.py)
        tau_int_window : the Sokal window, in time units
        tau_int : tail-corrected integral of C (preferred)
        t1 : slowest timescale in u (= tau_exp), from the slow-tail fit
        a1 : slow-mode weight (fit intercept)
        a1_err, t1_err : single-fit errors (optimistic)
        purity : tau_int / t1, an upper bound on a1 (equal to it when the
            fast modes are negligibly short)
        t_min, t_max : slow-tail fit window
    """
    fit = fit_slow_exponential(acf, **fit_kwargs)
    tau_sokal, t_win = sokal_tau_int(acf, c_window)
    tau_int = tail_corrected_tau_int(acf, fit.a1, fit.t1, fit.t_max)
    c1 = acf.ct[1]
    tau_0 = -acf.frametime / np.log(c1) if 0.0 < c1 < 1.0 else np.nan
    return DataClass(
        tau_0=tau_0,
        tau_int_sokal=tau_sokal,
        tau_int_window=t_win,
        tau_int=tau_int,
        t1=fit.t1,
        a1=fit.a1,
        t1_err=fit.t1_err,
        a1_err=fit.a1_err,
        purity=tau_int / fit.t1,
        t_min=fit.t_min,
        t_max=fit.t_max,
    )


def local_timescales(acf, window=11, polyorder=2, t_max=None):
    """Lag-dependent timescales, as curves.

    chord   t_u(tau)  = -tau / ln C(tau)           (acf.compute_theta)
    tangent t_1(tau)  = -1 / (d ln C / d tau)      (math.derivative, Savitzky-Golay)

    The chord rises from tau_0 to t_1 slowly (like 1/tau); the tangent reaches
    t_1 exponentially fast, on the scale of t_2.  Both are NaN where C <= 0
    and beyond t_max (default: the noise cutoff from fit_window).

    Returns DataClass with t, chord, tangent (arrays).
    """
    if t_max is None:
        _, t_max = fit_window(acf)
    keep = acf.t <= t_max
    chord = compute_theta(acf.ct, acf.frametime)
    chord[~keep] = np.nan

    tangent = np.full(acf.ct.shape, np.nan)
    ok = keep & (acf.ct > 0.0)
    n = np.flatnonzero(~ok[1:])  # derivative over the leading run of usable lags
    n = (n[0] + 1) if n.size else ok.size
    if n > window:
        dlnc = derivative(acf.t[:n], np.log(acf.ct[:n]), method="savgol",
                          window=window, polyorder=polyorder)
        with np.errstate(divide="ignore"):
            tangent[:n] = np.where(dlnc < 0.0, -1.0 / dlnc, np.nan)
    return DataClass(t=acf.t, chord=chord, tangent=tangent)


# --------------------------------------------------------------------------- #
#  2. Multi-exponential fit (the "problematic" analysis)
# --------------------------------------------------------------------------- #
def _unpack(p, n, sum_to_one):
    """Parameters p -> (a, t).  t = exp(log t); a = exp(log a), or softmax."""
    lt = p[-n:]
    if sum_to_one:
        z = np.concatenate([[0.0], p[: n - 1]])
        a = np.exp(z - z.max())
        a /= a.sum()
    else:
        a = np.exp(p[:n])
    return a, np.exp(lt)


def _dat_dp(a, t, sum_to_one):
    """Jacobian of (a, t) with respect to the fit parameters p."""
    n = a.size
    npar = 2 * n - 1 if sum_to_one else 2 * n
    M = np.zeros((2 * n, npar))
    if sum_to_one:  # a = softmax(0, z_2..z_n): da_i/dz_j = a_i (delta_ij - a_j)
        for j in range(1, n):
            M[:n, j - 1] = a * ((np.arange(n) == j) - a[j])
    else:
        M[:n, :n] = np.diag(a)
    M[n:, -n:] = np.diag(t)
    return M


def fit_multi_exponential(
    acf,
    n_terms,
    t_max=None,
    sum_to_one=False,
    n_starts=8,
    rng=None,
    k_sigma=3.0,
):
    """Fit C(t) ~ sum_{i=1}^{n_terms} a_i exp(-t/t_i) on [0, t_max].

    Nonlinear least squares in log(a_i), log(t_i) (so a_i, t_i > 0), weighted
    by 1/sigma^2 when acf.sigma exists, from n_starts random starting points;
    the lowest-cost solution wins.

    Parameters
    ----------
    acf : DataClass from acf_input
    n_terms : number of exponentials
    t_max : end of the fit range (default: noise cutoff from fit_window)
    sum_to_one : impose sum_i a_i = 1 (so the model has C(0) = 1)
    n_starts : number of random starts (log-uniform t_i in [dt, t_max])
    rng : numpy Generator for the starts (default: seeded with 0, repeatable)

    Returns
    -------
    DataClass with
        coeffs, timescales : a_i, t_i sorted slowest first
        coeffs_err, timescales_err : from the Jacobian (optimistic)
        corr : (2n, 2n) correlation matrix of (a_1..a_n, t_1..t_n)
        cond : condition number of corr.  Large (>1e4) means the parameters
            are not separately determined by the data.
        chi2_dof : reduced chi^2 (or residual variance if no sigma)
        n_distinct : number of starts that reached the best cost (within
            1e-6 relative) but with timescales differing by >5%
        fit : model C(t) on acf.t
        t_max : fit range end
    """
    if rng is None:
        rng = np.random.default_rng(0)
    if t_max is None:
        _, t_max = fit_window(acf, t_min=0.0, k_sigma=k_sigma)
    sel = acf.t <= t_max
    t, c = acf.t[sel], acf.ct[sel]
    s = np.ones_like(c) if acf.sigma is None else acf.sigma[sel]
    n = int(n_terms)
    dt = acf.frametime

    def resid(p):
        a, ts = _unpack(p, n, sum_to_one)
        return (exp_sum(t, a, ts) - c) / s

    def jac(p):
        a, ts = _unpack(p, n, sum_to_one)
        E = np.exp(-t[:, None] / ts[None])
        J_at = np.hstack([E, a * t[:, None] * E / ts**2]) / s[:, None]
        return J_at @ _dat_dp(a, ts, sum_to_one)

    lo_t, hi_t = np.log(dt / 10.0), np.log(100.0 * t_max)
    lo = np.r_[np.full(n - 1 if sum_to_one else n, -30.0), np.full(n, lo_t)]
    hi = np.r_[np.full(n - 1 if sum_to_one else n, 30.0 if sum_to_one else 1.0),
               np.full(n, hi_t)]

    sols = []
    for k in range(n_starts):
        if k == 0:  # deterministic start: log-spaced timescales
            lt0 = np.linspace(np.log(t_max / 2), np.log(dt), n)
        else:
            lt0 = np.sort(rng.uniform(np.log(dt), np.log(t_max), n))[::-1]
        la0 = np.zeros(n - 1) if sum_to_one else np.full(n, np.log(1.0 / n))
        p0 = np.clip(np.r_[la0, lt0], lo + 1e-9, hi - 1e-9)
        try:
            r = least_squares(resid, p0, jac=jac, bounds=(lo, hi), x_scale="jac")
            sols.append(r)
        except (ValueError, np.linalg.LinAlgError):
            continue
    if not sols:
        raise RuntimeError("all starts failed")
    best = min(sols, key=lambda r: r.cost)

    # distinct near-optimal solutions (an ill-conditioning symptom)
    best_t = np.sort(_unpack(best.x, n, sum_to_one)[1])
    n_distinct = 0
    for r in sols:
        if r.cost <= best.cost * (1 + 1e-6) + 1e-30:
            tr = np.sort(_unpack(r.x, n, sum_to_one)[1])
            if np.any(np.abs(tr / best_t - 1.0) > 0.05):
                n_distinct += 1

    a, ts = _unpack(best.x, n, sum_to_one)
    order = np.argsort(ts)[::-1]
    a, ts = a[order], ts[order]

    # covariance of (a, t) via the parameter Jacobian
    E = np.exp(-t[:, None] / ts[None])
    J_at = np.hstack([E, a * t[:, None] * E / ts**2]) / s[:, None]
    p_sorted = (
        np.r_[np.log(a[1:] / a[0]), np.log(ts)] if sum_to_one
        else np.r_[np.log(a), np.log(ts)]
    )
    M = _dat_dp(*_unpack(p_sorted, n, sum_to_one), sum_to_one)
    J_p = J_at @ M
    dof = max(t.size - J_p.shape[1], 1)
    chi2_dof = 2.0 * best.cost / dof
    cov_p = np.linalg.pinv(J_p.T @ J_p)
    if acf.sigma is None:
        cov_p *= chi2_dof
    cov = M @ cov_p @ M.T
    sd = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = cov / np.outer(sd, sd)
    # condition number of the free-parameter correlation matrix
    sdp = np.sqrt(np.clip(np.diag(cov_p), 1e-300, None))
    cond = float(np.linalg.cond(cov_p / np.outer(sdp, sdp)))

    return DataClass(
        coeffs=a,
        timescales=ts,
        coeffs_err=sd[:n],
        timescales_err=sd[n:],
        corr=corr,
        cond=cond,
        chi2_dof=chi2_dof,
        n_distinct=n_distinct,
        fit=exp_sum(acf.t, a, ts),
        t_max=float(t_max),
    )


# --------------------------------------------------------------------------- #
#  3. Slow mode + residual (the "sensible" analysis)
# --------------------------------------------------------------------------- #
def slow_mode_analysis(acf, c_window=5.0, c_window_R=8.0, split_window=None,
                       **fit_kwargs):
    """Split C(t) = a_1 exp(-t/t_1) + R(t) and summarize both parts.

    a_1 and t_1 come from fit_slow_exponential.  The residual R is
    normalized, r(t) = R(t)/(1 - a_1) with r(0) = 1, and integrated with a
    Sokal window (smallest M with M >= c_window_R * tau_R(M)), because far
    from 0 R holds only noise plus the error of the slow fit:
    a slow-fit error delta_a would otherwise add delta_a * t_1 / (1 - a_1).

    Even inside the window, delta_a leaks in as delta_a * W / (1 - a_1) for a
    window W, which dominates the error of tau_R when 1 - a_1 is small.
    split_window = (t_lo, t_hi) reduces it: the slow component is refitted
    on the short window [t_lo, t_hi] just past the fast modes (e.g. 8-30 x
    the fast time), so its extrapolation back to t = 0 is short, and R is
    integrated only up to t_lo.  t_1 still comes from the full tail fit.
    The local amplitude is also the better a_1: on synthetic two-state
    replicas it had ~40x (dt 100 ps) and ~17x (dt 1 ps) smaller spread than
    the tail-fit a_1, which is kept as a1_tail.

    Parameters
    ----------
    acf : DataClass from acf_input
    c_window : Sokal constant for tau_int_sokal
    c_window_R : Sokal constant for the residual integral (default split)
    split_window : None, or (t_lo, t_hi) for the local split described above
    fit_kwargs : passed to fit_slow_exponential

    Returns
    -------
    DataClass with
        a1, t1 : slow mode.  With split_window, a1 is the local-fit amplitude
        a1_tail : a1 from the full tail fit (equal to a1 without split_window)
        tau_R : weight-averaged fast timescale, sum_{i>=2} a_i t_i / sum a_i
        R_window : integration window for tau_R (t_lo with split_window)
        tau_R_resolved : 1.0 if R(dt)/(1 - a_1) > 1/e, else 0.0.  At 0.0 the
            fast modes decay within one frame (e.g. Parrinello at 100 ps), so
            tau_R ~ dt/2 is only an upper bound and gap only a lower bound.
        gap : t1 / tau_R (spectral gap as a timescale ratio); log_gap = ln gap
        signal_ratio : a1 / (1 - a1)
        tau_int : tail-corrected integral of C (data, see acf_timescales)
        tau_int_sokal : Sokal-windowed integral of C
        tau_int_identity : a1 t1 + (1 - a1) tau_R
        identity_error : tau_int_identity - tau_int.  Algebraically this is
            -int_{R_window}^{t_max} R dt, so it checks that R really is gone
            past its window and that the slow fit matches the data there.
            With split_window it mainly measures (a1 - a1_tail) * t1, i.e.
            whether the local and tail amplitudes agree.
        slow_share : a1 t1 / tau_int, share of tau_int carried by the slow mode
        t_min, t_max : slow-tail fit window
        t, R : the residual R(t) = C(t) - a1 exp(-t/t1)
    """
    fit = fit_slow_exponential(acf, **fit_kwargs)
    t1 = fit.t1
    if split_window is None:
        a1 = fit.a1
        R = acf.ct - a1 * np.exp(-acf.t / t1)
    else:
        t_lo, t_hi = split_window
        method = fit_kwargs.get("method", "nonlinear")
        loc = fit_slow_exponential(acf, t_min=t_lo, t_max=t_hi, method=method)
        a1 = loc.a1
        R = acf.ct - a1 * np.exp(-acf.t / loc.t1)

    w = 1.0 - a1
    if w <= 0.0:
        tau_R, t_R = np.nan, np.nan
    else:
        running = running_trapezoid(R / w, acf.frametime)
        if split_window is None:
            M = sokal_window(running, acf.frametime, c_window_R)
        else:
            M = int(round(split_window[0] / acf.frametime))
        tau_R, t_R = float(running[M]), float(acf.t[M])
    # The trapezoid needs R to survive at least one frame.  If the fast modes
    # are gone by lag 1, it returns about dt/2 whatever the true tau_R is.
    resolved = w > 0.0 and R[1] / w > np.exp(-1.0)

    tau_int = tail_corrected_tau_int(acf, fit.a1, t1, fit.t_max)
    tau_sokal, _ = sokal_tau_int(acf, c_window)
    tau_id = a1 * t1 + w * tau_R

    return DataClass(
        a1=a1,
        a1_tail=fit.a1,
        t1=t1,
        tau_R=tau_R,
        R_window=t_R,
        tau_R_resolved=float(resolved),
        gap=t1 / tau_R,
        log_gap=np.log(t1 / tau_R) if tau_R > 0 else np.nan,
        signal_ratio=a1 / w if w > 0 else np.inf,
        tau_int=tau_int,
        tau_int_sokal=tau_sokal,
        tau_int_identity=tau_id,
        identity_error=tau_id - tau_int,
        slow_share=a1 * t1 / tau_int,
        t_min=fit.t_min,
        t_max=fit.t_max,
        t=acf.t,
        R=R,
    )


# --------------------------------------------------------------------------- #
#  Spread over independent ACFs
# --------------------------------------------------------------------------- #
def spread_over_draws(draws, frametime, analysis, n_frames=None, ct_err=None,
                      noise_factor=1.0, **kwargs):
    """Run an analysis on many ACFs and collect the spread of its results.

    Parameters
    ----------
    draws : array (num_draws, num_lags), e.g. Bartlett draws or block ACFs
    frametime, n_frames, ct_err, noise_factor : passed to acf_input
    analysis : a function DataClass -> DataClass, e.g. slow_mode_analysis
    kwargs : passed to analysis

    Returns
    -------
    DataClass with, for every float field x of the analysis result:
        x : array of values per draw
        x_mean, x_sd : mean and standard deviation over draws (NaNs ignored)
    plus num_failed (draws whose analysis raised).
    """
    vals, failed = {}, 0
    for d in np.atleast_2d(draws):
        acf = acf_input(d, frametime, n_frames, ct_err, noise_factor)
        try:
            res = analysis(acf, **kwargs)
        except (ValueError, RuntimeError):
            failed += 1
            continue
        for k, v in res.__dict__.items():
            if isinstance(v, float):
                vals.setdefault(k, []).append(v)
    out = {"num_failed": failed}
    for k, v in vals.items():
        v = np.array(v)
        out[k] = v
        out[f"{k}_mean"] = np.nanmean(v)
        out[f"{k}_sd"] = np.nanstd(v, ddof=1) if v.size > 1 else np.nan
    return DataClass(**out)
