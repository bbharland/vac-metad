import numpy as np
from scipy.optimize import curve_fit, least_squares

from ..dataclass import DataClass
from ..math import weighted_linear_fit, running_trapezoid

"""
See claude/2026.11.acf-analysis/acf.analysis.md

Given an acf DataClass with fields:
    t, ct : array (num_lags,)               C(t), normalized C(0) = 1
    frametime : float                       dt (in, say, ps)
    num_frames : int                        length of full simulation
    ct_err : (optional) array (num_lags,)   used as sigma in place of Bartlett
                                            if present
Usage:
    ts = acf_timescales(acf)
    sm = slow_mode_analysis(acf, split_window=(300.0, 1200.0))
    plot_timescales_slow_mode(acf, ts, sm)

    for n in (2, 3, 4):
        m = fit_multi_exponential(acf, n, sum_to_one=True)
        print(n, m.coeffs.round(3), m.timescales.round(1), f"cond={m.cond:.2g}")
"""


def acf_sigma(acf, noise_factor=1.0):
    """1-sigma error on C(t): ct_err if present, else Bartlett, else None."""
    if getattr(acf, "ct_err", None) is not None:
        return acf.ct_err
    if acf.num_frames is not None:
        return np.full(acf.ct.shape, noise_factor * bartlett_sigma(acf.ct, acf.num_frames))
    return None


def bartlett_sigma(ct, num_frames):
    """Large-lag Bartlett standard error of C_hat, constant in tau:

        sigma_C^2 = (1 + 2 sum_{k>=1} C_k^2) / n_frames.

    The sum runs up to the first lag where C_hat <= 0, so the noisy tail
    does not add to it.  Gaussian statistics assumed.
    """
    ct = np.asarray(ct, float)
    neg = np.flatnonzero(ct <= 0.0)
    k_end = neg[0] if neg.size else ct.size
    return float(np.sqrt((1.0 + 2.0 * np.sum(ct[1:k_end] ** 2)) / num_frames))


def fit_window(acf, t_min=None, t_max=None, k_sigma=3.0, t_min_frac=0.1, noise_factor=1.0):
    """Return (t_min, t_max) for the slow-tail fit.

    t_max : given, or the last lag before C_hat first drops below
            k_sigma * sigma (searching from lag 1), or the last lag if
            there is no noise information (acf_sigma is None).
    t_min : given, or t_min_frac * t_max.
    """
    t, ct = acf.t, acf.ct
    if t_max is None:
        sigma = acf_sigma(acf, noise_factor)
        if sigma is None:
            t_max = t[-1]
        else:
            low = np.flatnonzero(ct[1:] < k_sigma * sigma[1:])
            t_max = t[low[0]] if low.size else t[-1]  # low[0]+1 is the first bad lag
    if t_min is None:
        t_min = t_min_frac * t_max
    if t_min >= t_max:
        raise ValueError(f"empty fit window: t_min={t_min} >= t_max={t_max}")
    return float(t_min), float(t_max)


def fit_slow_exponential(
    acf,
    t_min=None,
    t_max=None,
    method="nonlinear",
    unit_amplitude=False,
    k_sigma=3.0,
    t_min_frac=0.1,
    noise_factor=1.0,
):
    """Fit C(t) ~ a_1 exp(-t / t_1) on [t_min, t_max].

    Parameters
    ----------
    acf : ACF DataClass (t, ct, frametime, num_frames, optional ct_err)
    t_min, t_max, k_sigma, t_min_frac : fit window, see fit_window
    method : 'nonlinear' (default) fits C itself, weighted by 1/sigma^2.
             'log' fits ln C by a straight line, weighted by C^2/sigma^2
             (the consistent weight for ln C).  Points with C <= 0 are
             dropped, which biases it upward in a noisy tail.
    unit_amplitude : force a_1 = 1 (the fit then goes through C(0) = 1)
    noise_factor : multiplies the Bartlett sigma_C, see acf_sigma

    Returns
    -------
    DataClass with a1, t1, a1_err, t1_err, t_min, t_max, num_points.
    The errors assume independent noise per lag and are too small for real
    ACFs.
    """
    t_min, t_max = fit_window(acf, t_min, t_max, k_sigma, t_min_frac, noise_factor)
    sigma = acf_sigma(acf, noise_factor)
    sel = (acf.t >= t_min) & (acf.t <= t_max)
    t, c = acf.t[sel], acf.ct[sel]
    s = None if sigma is None else sigma[sel]

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
    acf : ACF DataClass (t, ct, frametime, num_frames, optional ct_err)
    c_window : Sokal window constant for tau_int_sokal
    fit_kwargs : passed to fit_slow_exponential (t_min, t_max, method,
        noise_factor, ...)

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


def slow_mode_analysis(acf, c_window_R=8.0, split_window=None, **fit_kwargs):
    """Split C(t) = a_1 exp(-t/t_1) + R(t), R(0) = 1 - a_1, and report

        tau_R = int_0^W R dt / (1 - a_1)        weight-averaged fast timescale
        gap   = t_1 / tau_R                     spectral gap
        tau_int = a_1 t_1 + (1 - a_1) tau_R     identity (checked by identity_error)

    W (the R window):
        split_window = None : Sokal window, smallest W with W >= c_window_R * tau_R(W).
        split_window = (t_lo, t_hi) : a_1 is refitted on [t_lo, t_hi], just past
            the fast modes, and W = t_lo.  A slow-fit error da leaks into tau_R as
            da * W / (1 - a_1), so a short W helps; the local a_1 is also less
            noisy than the tail a_1.  t_1 still comes from the full tail fit.

    Parameters
    ----------
    acf : ACF DataClass (t, ct, frametime, num_frames, optional ct_err)
    c_window_R : Sokal constant for the R integral (split_window = None only)
    split_window : None, or (t_lo, t_hi)
    fit_kwargs : passed to fit_slow_exponential (both fits, apart from t_min, t_max)

    Returns
    -------
    DataClass with
        a1, t1 : slow mode (a1 from the local fit if split_window)
        a1_tail : a1 from the full tail fit
        tau_R, R_window : fast timescale and its window W
        tau_R_resolved : 1.0 if R(dt)/(1 - a_1) > 1/e, else 0.0 (then tau_R ~ dt/2
            is only an upper bound, gap only a lower bound)
        gap : t1 / tau_R
        tau_int : tail-corrected integral of C
        identity_error : a1 t1 + (1 - a1) tau_R - tau_int
        t_min, t_max : tail-fit window
        R : C(t) - a1 exp(-t/t1), on acf.t
    """
    fit = fit_slow_exponential(acf, **fit_kwargs)
    t1 = fit.t1
    if split_window is None:
        a1 = fit.a1
        R = acf.ct - a1 * np.exp(-acf.t / t1)
    else:
        t_lo, t_hi = split_window
        loc_kwargs = {**fit_kwargs, "t_min": t_lo, "t_max": t_hi}
        loc = fit_slow_exponential(acf, **loc_kwargs)
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
    # The trapezoid needs R to survive one frame; otherwise tau_R ~ dt/2.
    resolved = w > 0.0 and R[1] / w > np.exp(-1.0)

    tau_int = tail_corrected_tau_int(acf, fit.a1, t1, fit.t_max)

    return DataClass(
        a1=a1,
        a1_tail=fit.a1,
        t1=t1,
        tau_R=tau_R,
        R_window=t_R,
        tau_R_resolved=float(resolved),
        gap=t1 / tau_R,
        tau_int=tau_int,
        identity_error=a1 * t1 + w * tau_R - tau_int,
        t_min=fit.t_min,
        t_max=fit.t_max,
        R=R,
    )


def _unpack(p, n, sum_to_one):
    """Fit parameters p -> (a, t).  t = exp(log t); a = exp(log a), or softmax."""
    if sum_to_one:
        z = np.concatenate([[0.0], p[: n - 1]])
        a = np.exp(z - z.max())
        a /= a.sum()
    else:
        a = np.exp(p[:n])
    return a, np.exp(p[-n:])


def _dat_dp(a, t, sum_to_one):
    """Jacobian d(a, t)/dp of the (a, t) parametrization."""
    n = a.size
    M = np.zeros((2 * n, 2 * n - 1 if sum_to_one else 2 * n))
    if sum_to_one:  # a = softmax(0, z_2..z_n): da_i/dz_j = a_i (delta_ij - a_j)
        for j in range(1, n):
            M[:n, j - 1] = a * ((np.arange(n) == j) - a[j])
    else:
        M[:n, :n] = np.diag(a)
    M[n:, -n:] = np.diag(t)
    return M


def _jac_at(t, a, ts, s):
    """Weighted Jacobian of C(t) = sum_i a_i exp(-t/t_i) w.r.t. (a, t)."""
    E = np.exp(-t[:, None] / ts)
    return np.hstack([E, a * t[:, None] * E / ts**2]) / s[:, None]


def fit_multi_exponential(
    acf,
    n_terms,
    t_max=None,
    sum_to_one=False,
    n_starts=8,
    rng=None,
    k_sigma=3.0,
    noise_factor=1.0,
):
    """Fit C(t) ~ sum_{i=1}^{n} a_i exp(-t/t_i) on [0, t_max].

    Least squares in p = (log a_i, log t_i), weighted by 1/sigma^2 when
    acf_sigma(acf, noise_factor) exists, from n_starts starts; lowest cost wins.
    With sum_to_one, a = softmax(0, z_2..z_n), so sum_i a_i = 1.

    Parameters
    ----------
    acf : ACF DataClass (t, ct, frametime, num_frames, optional ct_err)
    n_terms : number of exponentials n
    t_max : end of the fit range (default: noise cutoff from fit_window;
        the last lag if noise-free).  Cost ~ num_points * n_starts: on the
        noise-free toy (20k lags) n=4 takes 35 s; t_max ~ 3 t_1 gives 4 s.
    sum_to_one : impose sum_i a_i = 1
    n_starts : start 0 has log-spaced t_i; the rest are log-uniform in [dt, t_max]
    rng : numpy Generator for the starts (default: seeded with 0, repeatable)
    k_sigma, noise_factor : noise cutoff, see fit_window and acf_sigma

    Returns
    -------
    DataClass with
        coeffs, timescales : a_i, t_i, slowest first
        coeffs_err, timescales_err : from the Jacobian (optimistic)
        corr : (2n, 2n) correlation matrix of (a_1..a_n, t_1..t_n)
        cond : condition number of the correlation matrix of p (> 1e4: the
            parameters are not separately determined)
        chi2_dof : reduced chi^2 (residual variance if noise-free)
        n_distinct : starts reaching the best cost (rel. 1e-6) with some t_i
            more than 5% different
        fit : model C(t) on acf.t
        t_max : fit range end
    """
    if rng is None:
        rng = np.random.default_rng(0)
    if t_max is None:
        _, t_max = fit_window(acf, t_min=0.0, k_sigma=k_sigma, noise_factor=noise_factor)
    sigma = acf_sigma(acf, noise_factor)
    sel = acf.t <= t_max
    t, c = acf.t[sel], acf.ct[sel]
    s = np.ones_like(c) if sigma is None else sigma[sel]
    n = int(n_terms)
    dt = acf.frametime

    def resid(p):
        a, ts = _unpack(p, n, sum_to_one)
        return (np.exp(-t[:, None] / ts) @ a - c) / s

    def jac(p):
        a, ts = _unpack(p, n, sum_to_one)
        return _jac_at(t, a, ts, s) @ _dat_dp(a, ts, sum_to_one)

    # bounds: t_i in [dt/10, 100 t_max]; log a_i <= 1 (or softmax z in [-30, 30])
    n_a = n - 1 if sum_to_one else n
    lo = np.r_[np.full(n_a, -30.0), np.full(n, np.log(dt / 10.0))]
    hi = np.r_[np.full(n_a, 30.0 if sum_to_one else 1.0), np.full(n, np.log(100.0 * t_max))]

    sols = []
    for k in range(n_starts):
        if k == 0:
            lt0 = np.linspace(np.log(t_max / 2), np.log(dt), n)
        else:
            lt0 = np.sort(rng.uniform(np.log(dt), np.log(t_max), n))[::-1]
        la0 = np.zeros(n_a) if sum_to_one else np.full(n, np.log(1.0 / n))
        p0 = np.clip(np.r_[la0, lt0], lo + 1e-9, hi - 1e-9)
        try:
            sols.append(least_squares(resid, p0, jac=jac, bounds=(lo, hi), x_scale="jac"))
        except (ValueError, np.linalg.LinAlgError):
            continue
    if not sols:
        raise RuntimeError("all starts failed")
    best = min(sols, key=lambda r: r.cost)

    best_t = np.sort(_unpack(best.x, n, sum_to_one)[1])
    n_distinct = 0
    for r in sols:
        if r.cost <= best.cost * (1 + 1e-6) + 1e-30:
            tr = np.sort(_unpack(r.x, n, sum_to_one)[1])
            n_distinct += int(np.any(np.abs(tr / best_t - 1.0) > 0.05))

    a, ts = _unpack(best.x, n, sum_to_one)
    order = np.argsort(ts)[::-1]
    a, ts = a[order], ts[order]

    # covariance of (a, t): cov_p = (J_p^T J_p)^-1, cov = M cov_p M^T, M = d(a,t)/dp
    p_sorted = np.r_[np.log(a[1:] / a[0]) if sum_to_one else np.log(a), np.log(ts)]
    M = _dat_dp(*_unpack(p_sorted, n, sum_to_one), sum_to_one)
    J_p = _jac_at(t, a, ts, s) @ M
    chi2_dof = 2.0 * best.cost / max(t.size - J_p.shape[1], 1)
    cov_p = np.linalg.pinv(J_p.T @ J_p)
    if sigma is None:
        cov_p *= chi2_dof
    cov = M @ cov_p @ M.T
    sd = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = cov / np.outer(sd, sd)
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
        fit=np.exp(-acf.t[:, None] / ts) @ a,
        t_max=float(t_max),
    )
