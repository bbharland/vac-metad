"""
Autocorrelation / relaxation-time analysis for collective variables (CVs)
from long, unbiased, reversible molecular-dynamics trajectories.

CONVENTIONS
-----------
Everything is built on the *centered, normalized* autocorrelation function

        C(tau) = < du(0) du(tau) > / < du(0)^2 >,      du = u - <u>,        (1)

so that  C(0) = 1  and, for an ergodic / mixing CV,  C(tau) -> 0  as tau -> oo.

Why center, always:
    A transfer-operator eigenfunction phi_i (i >= 1) is orthogonal to the
    constant eigenfunction phi_0 = 1 under the stationary density mu, hence
    <phi_i>_mu = 0 and <phi_i^2>_mu = 1 by construction.  For those CVs Eq. (1)
    collapses to the raw < u(0) u(tau) >, and centering / normalizing are
    no-ops up to finite-sample fluctuations.  A dihedral angle instead has
    <u> != 0, so WITHOUT centering C(tau) -> <u>^2 / <u^2> != 0 and
    tau = int C dtau diverges.  Eq. (1) is the *single* formula that is correct
    for both classes; that is the "modification" rather than a special case.

Why angles need more than centering:
    A dihedral is a *circular* variable.  The arithmetic mean and the
    subtraction u - <u> are not invariant to the 2*pi branch cut, so a state
    straddling +/- pi yields a meaningless <u>.  Embed the angle on the unit
    circle, z = exp(i*phi), and use the Hermitian ACF

        C(tau) = Re < dz(0)^* dz(tau) > / < |dz|^2 >,   dz = z - <z>,       (2)

    with <z> = Rbar * exp(i*phibar) the mean resultant vector.  Eq. (2) is
    invariant to the branch cut and to any global rotation of the angle's zero,
    and is real and even for reversible dynamics (the imaginary part is a free
    time-reversal diagnostic, ~ 0 within noise).

ESTIMATOR
---------
C(tau) is computed by the Wiener-Khinchin route in O(N log N):
    zero-pad to >= 2N-1 (linear, not circular, correlation),
    power spectrum |FFT(du)|^2, inverse FFT, normalize.
'unbiased' divides lag k by (N-k); 'biased' divides by N (lower variance and
positive semidefinite at large lag, but shrinks the tail).

TIMESCALE
---------
tau = int_0^infty C(t) dt  is estimated as  frametime * tau_int  with the
Madras-Sokal integrated time and automatic windowing,
    tau_int(M) = 1/2 + sum_{k=1}^{M} C(k),
    M = smallest window with M >= c_window * tau_int(M),
which is the trapezoidal integral (C(0)=1) truncated before the noisy tail.
The naive  frametime * ct.sum()  differs in two ways: it is a left Riemann sum
(overcounts by ~ frametime/2) and it integrates the entire noisy tail.
"""

import numpy as np
from .dataclass import DataClass
from .util import (
    save_pickle,
    load_pickle
)


# --------------------------------------------------------------------------- #
#  Low-level pieces
# --------------------------------------------------------------------------- #
def _embed(u, kind):
    """Return centered signal du (real for 'linear', complex for 'circular')."""
    u = np.asarray(u)
    if kind == "linear":
        x = u.astype(float)
        return x - x.mean()
    elif kind == "circular":
        # u is an angle in radians; embed on the unit circle and subtract the
        # mean resultant vector <z> = Rbar * exp(i phibar).
        z = np.exp(1j * u.astype(float))
        return z - z.mean()
    raise ValueError("kind must be 'linear' or 'circular'")


def _autocov_fft(du, nlags, bias="unbiased"):
    """Unnormalized autocovariance via FFT.

    Returns g with g[k] ~ (1/denom) sum_t conj(du_t) du_{t+k}, g[0] = variance.
    Real for real input, complex (Hermitian) for circular input.
    """
    n = du.shape[0]
    nlags = int(min(nlags, n))
    nfft = 1 << (2 * n - 1).bit_length()  # next pow2 >= 2n-1
    f = np.fft.fft(du, n=nfft)
    g = np.fft.ifft(f * np.conj(f))[:nlags]  # linear autocorrelation
    if np.isrealobj(du):
        g = g.real
    if bias == "unbiased":
        g = g / (n - np.arange(nlags))  # exact # of terms per lag
    elif bias == "biased":
        g = g / n
    else:
        raise ValueError("bias must be 'unbiased' or 'biased'")
    return g


def auto_window(tau_int_running, c_window=5.0):
    """Madras-Sokal automatic window: smallest M with M >= c_window*tau_int(M)."""
    m = np.arange(tau_int_running.size)
    ok = m >= c_window * tau_int_running
    if not ok.any():
        print(
            "[warn] Sokal window never closed; using the full series "
            "(tau may be tail-contaminated). Use a longer trajectory or "
            "a smaller c_window."
        )
        return tau_int_running.size - 1
    return int(np.argmax(ok))


def _integrated_time(ct, frametime, c_window=5.0):
    """tau (physical units) from a normalized ACF ct (ct[0]=1) via Sokal."""
    tau_running = np.cumsum(ct) - 0.5  # tau_running[M] = 1/2 + sum_{1..M} ct
    M = auto_window(tau_running, c_window)
    tau_int = tau_running[M]  # in frames
    return frametime * tau_int, M, tau_int


# --------------------------------------------------------------------------- #
#  Public API
# --------------------------------------------------------------------------- #
def compute_acf(
    u, frametime, num_frames_acf, kind="linear", bias="unbiased", c_window=5.0
):
    """Centered, normalized autocorrelation function and relaxation time.

    Parameters
    ----------
    u : array, shape (num_frames,)
        Trajectory of the CV.  For kind='circular', u is an angle in radians.
    frametime : float
        The frame spacing (Delta t): time between successive samples of ``u``.
        Sets the physical units of the returned relaxation time and of
        theta(tau).  This is NOT the MSM lag time tau: the ACF lag axis is
        measured in frames, so the correct multiplier is the frame spacing, and
        the lag time never enters an autocorrelation integral.  (For a CV
        sampled every frame, ``frametime`` is the SimulationData ``frametime``,
        which equals ``lagtime`` only when ``lagframes == 1``.)
    num_frames_acf : int
        Number of lags to compute (0 ... num_frames_acf-1).
    kind : {'linear', 'circular'}
        'linear' for unbounded CVs (TICA/VAC eigenfunctions, distances, ...);
        'circular' for angles (dihedrals) -- uses Eq. (2).
    bias : {'unbiased', 'biased'}
        Lag-k normalization: 1/(N-k) (unbiased, noisier tail) or 1/N (biased,
        positive semidefinite, shrinks tail).  Use 'unbiased' for displaying
        C(tau)/theta(tau); the windowed tau below is robust to either.
    c_window : float
        Sokal window constant (5-10 typical).

    Returns
    -------
    tau : float
        Relaxation time = int_0^infty C(t) dt, Sokal-windowed.
    ct : ndarray, shape (num_frames_acf,)
        Normalized ACF, C(0)=1 (real part for circular input).
    info : dict
        c0 (variance / 1-Rbar^2), window M, tau_int (frames),
        and 'imag' (Hermitian imaginary part) for circular reversibility checks.
    """
    du = _embed(u, kind)
    g = _autocov_fft(du, num_frames_acf, bias=bias)
    c0 = g[0].real if np.iscomplexobj(g) else g[0]
    ct_full = g / c0  # complex if circular
    ct = ct_full.real
    tau, M, tau_int = _integrated_time(ct, frametime, c_window)
    info = {
        "c0": float(c0),
        "window": int(M),
        "tau_int_frames": float(tau_int),
        "imag": (ct_full.imag if np.iscomplexobj(g) else np.zeros_like(ct)),
    }
    return tau, ct, info


def compute_theta(ct, frametime, ct_err=None):
    """Lag-resolved relaxation time  theta(tau) = -tau / ln C(tau).

    Flat (== t1) for a single exponential; drift/curvature exposes
    multi-exponential structure.  Undefined points (tau=0, C<=0, C>=1) are
    returned as NaN rather than raising.

    Parameters
    ----------
    ct : ndarray
        Normalized ACF with ct[0] = 1.
    frametime : float
        Time between frames.
    ct_err : ndarray, optional
        1-sigma error on C(tau).  If given, the linearized propagation
        sigma_theta = tau / (C (ln C)^2) * sigma_C  is also returned.

    Returns
    -------
    theta : ndarray
        theta(tau), NaN where undefined.
    theta_err : ndarray            (only if ct_err is not None)
    """
    ct = np.asarray(ct, float)
    t = frametime * np.arange(ct.size)
    lnC = np.log(np.where(ct > 0.0, ct, np.nan))  # mask C <= 0
    theta = np.full_like(ct, np.nan)
    good = np.isfinite(lnC) & (lnC < 0.0)  # keep 0 < C < 1 only
    theta[good] = -t[good] / lnC[good]
    if ct_err is None:
        return theta
    ct_err = np.asarray(ct_err, float)
    theta_err = np.full_like(ct, np.nan)
    theta_err[good] = (t[good] / (ct[good] * lnC[good] ** 2)) * ct_err[good]
    return theta, theta_err


def compute_acf_with_errors(
    u,
    frametime,
    num_frames_acf,
    num_blocks=8,
    kind="linear",
    bias="unbiased",
    c_window=5.0,
):
    """Error bars on C(tau), tau and theta(tau) by block / ensemble averaging.

    `u` may be either
      * a single 1-D trajectory -> split into `num_blocks` contiguous blocks
        (Flyvbjerg-Petersen spirit: blocks must be long compared with tau to be
        quasi-independent), or
      * a list / tuple of 1-D trajectories (independent runs) -> each run is one
        block and `num_blocks` is ignored.  This is the cleaner route of the two.

    Each block is analysed independently (its own mean, variance, ACF); the
    reported error is the standard error of the mean across blocks,
    SEM = std / sqrt(n_eff).  Interpret only features of C/theta that exceed it.

    Returns
    -------
    dict with: lag, time, ct_mean, ct_err, theta_mean, theta_err,
               tau_mean, tau_err, num_blocks.
    """
    if isinstance(u, (list, tuple)) or (isinstance(u, np.ndarray) and u.ndim == 2):
        blocks = [np.asarray(b) for b in u]
    else:
        u = np.asarray(u)
        L = u.size // num_blocks
        blocks = [u[i * L : (i + 1) * L] for i in range(num_blocks)]

    # Cap the number of lags at the shortest block so every block returns an
    # ACF of the same length (handles short blocks and unequal-length runs).
    nlags = int(min(num_frames_acf, min(len(b) for b in blocks)))
    if nlags < num_frames_acf:
        print(
            f"[warn] reducing num_frames_acf {num_frames_acf} -> {nlags} "
            f"(limited by the shortest block)."
        )

    cts, taus = [], []
    for b in blocks:
        tau_b, ct_b, _ = compute_acf(
            b, frametime, nlags, kind=kind, bias=bias, c_window=c_window
        )
        cts.append(ct_b)
        taus.append(tau_b)
    cts = np.vstack(cts)
    taus = np.asarray(taus)
    nb = len(blocks)

    ct_mean = cts.mean(axis=0)
    ct_err = cts.std(axis=0, ddof=1) / np.sqrt(nb)
    tau_mean = taus.mean()
    tau_err = taus.std(ddof=1) / np.sqrt(nb)

    # theta per block, then average (robust to the nonlinearity of -t/lnC).
    # Lags where every block has C<=0 are all-NaN; report NaN there quietly.
    thetas = np.vstack([compute_theta(c, frametime) for c in cts])
    n_ok = np.sum(np.isfinite(thetas), axis=0)
    theta_mean = np.full(nlags, np.nan)
    theta_err = np.full(nlags, np.nan)
    enough = n_ok >= 2
    theta_mean[n_ok >= 1] = np.nanmean(thetas[:, n_ok >= 1], axis=0)
    theta_err[enough] = np.nanstd(thetas[:, enough], axis=0, ddof=1) / np.sqrt(
        n_ok[enough]
    )

    return {
        "lag": np.arange(nlags),
        "time": frametime * np.arange(nlags),
        "ct_mean": ct_mean,
        "ct_err": ct_err,
        "theta_mean": theta_mean,
        "theta_err": theta_err,
        "tau_mean": float(tau_mean),
        "tau_err": float(tau_err),
        "num_blocks": nb,
    }


def sokal_tau_error(tau_int_frames, window, n_frames, frametime=1.0):
    """Madras-Sokal 1-sigma error on the integrated time (single trajectory):
        sigma(tau_int) ~ tau_int * sqrt(2 (2M+1) / N).
    Returned in units of `frametime` (default 1.0 -> frames); pass the real
    frametime to get physical time.  Quick analytic estimate; prefer
    block/ensemble SEM when feasible.
    """
    rel = np.sqrt(2.0 * (2.0 * window + 1.0) / n_frames)
    return frametime * tau_int_frames * rel


def compute_acf_original(u, frametime, num_frames_acf):
    """Compute autocorrelation function C(t).

    June 24, 2026
    -------------
    FFT 'compute_acf' was validated against this function using the unbiased 3.0 us dataset.  The curves have perfect overlap in matplotlib.

        Time for FFT call = 0:02.5
        Time for this call = 11:54
        <|C_FFT(t) - C(t)|> = 1.5e-10

    Parameters
    ----------
    u : array, shape (num_frames,)
        Full trajectory of CV, u(t)
    frametime : float
        Time separating trajectory frames.  Units will match timescale estimate
    num_frames_acf : int
        How far out to compute ACF (this is expensive).

    Return
    ------
    tau : float
        Computed timescale = int_0^infty dt C(t)
    ct : ndarray with shape (num_frames_acf,)
        The autocorrelation function, C(t)
    """

    def acf(u, frame_acf):
        if frame_acf == 0:
            return np.mean(u * u)
        else:
            return np.mean(u[:-frame_acf] * u[frame_acf:])

    c0 = acf(u, 0)
    ct = np.array([acf(u, t) for t in range(num_frames_acf)]) / c0
    tau = frametime * np.sum(ct)
    return tau, ct


def compute_acfs(wd, signals, frametime, num_frames_acf, calculate=True):
    file = wd / "acfs.pickle"
    if calculate:
        acfs = {
            label: compute_acf_dataclass(u, frametime, num_frames_acf)
            for label, u in signals.items()
        }
        save_pickle(file, acfs)
    else:
        acfs = load_pickle(file)
    return acfs


def compute_acf_dataclass(*args, **kwargs):
    tau, ct, info = compute_acf(*args, **kwargs)
    return DataClass(
        tau=tau,
        ct=ct,
        c0=info["c0"],
        window=info["window"],
        tau_int_frames=info["tau_int_frames"],
        imag=info["imag"],
    )
