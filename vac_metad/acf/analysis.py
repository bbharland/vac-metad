import numpy as np
from scipy.optimize import curve_fit

from ..dataclass import DataClass
from ..math import weighted_linear_fit, running_trapezoid


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
        tau_int_sokal : Sokal-windowed integral of C(t)
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


