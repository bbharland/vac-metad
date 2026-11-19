import numpy as np

from ..dataclass import DataClass


DEFAULT_COEFFS = (0.7, 0.1, 0.1, 0.1)
DEFAULT_TIMESCALES = (1850, 60, 26, 2)


def acf_terms(t, coeffs, timescales):
    """Return the individual ACF terms, a_i * exp(-t / tau_i), at time(s) t.

    t may be a scalar or a 1D array, in ps. The first axis of the result
    indexes the terms:
        scalar t -> shape (n_terms,)
        1D t     -> shape (n_terms, len(t))
    """
    return np.array([a * np.exp(-t / ti) for a, ti in zip(coeffs, timescales)])


def toy_model(
    frametime,
    num_lags,
    num_frames=None,
    coeffs=DEFAULT_COEFFS,
    timescales=DEFAULT_TIMESCALES,
):
    """Exact toy ACF, C(t) = sum_i a_i exp(-t / t_i)

    Parameters
    ----------
    frametime : time separating frames (ps, e.g.)
    num_lags : number of frames computed for ACF
    num_frames : length of a pretend trajectory, used only for noise
        estimates.  None means noise-free.
    coeffs : a_i terms
    timescales : t_1 terms
    """
    t = frametime * np.arange(num_lags)
    terms = acf_terms(t, coeffs, timescales)

    if num_frames is not None and num_lags > num_frames:
        raise ValueError("A real ACF can’t have more lags computed than the number of simulation frames")

    return DataClass(
        t=t,
        ct=terms.sum(axis=0),
        frametime=frametime,
        num_frames=num_frames,
        coeffs=np.array(coeffs),
        timescales=np.array(timescales),
        terms=terms,
    )



def bartlett_draws(acf, num_draws, rng=None, mean_shift=True, batch_size=50):
    """Synthetic estimates C_hat(t) of the toy ACF, as if from num_frames frames.

    Uses the moving-average form of Bartlett's formula (Brockwell & Davis,
    Thm 7.2.1), so no K x K covariance is built:

        sqrt(N) [C_hat(h) - rho(h)] ~ sum_k [rho(k+h) + rho(k-h) - 2 rho(h) rho(k)] W_k,

    W_k iid N(0, 1).  Gaussian statistics assumed: for a two-state u this
    underestimates the noise (~4-5x).

    Parameters
    ----------
    acf : toy DataClass from toy_model, with num_frames set
    num_draws : number of synthetic ACFs
    rng : numpy Generator (default: a fresh unseeded one)
    mean_shift : add the mean-subtraction bias -b (1 - rho), b = 2 tau_int / T
    batch_size : draws per FFT batch; lower it to save memory

    Returns
    -------
    DataClass with
        t : lag times (same as acf.t)
        draws : (num_draws, num_lags), draws[:, 0] = 1
        mean : expected estimate, rho - b (1 - rho) (or rho)
        sigma : Bartlett standard deviation per lag
    """
    if acf.num_frames is None:
        raise ValueError("bartlett_draws needs acf.num_frames (trajectory length)")
    if rng is None:
        rng = np.random.default_rng()

    # N = num_frames, K = num_lags
    N, dt = acf.num_frames, acf.frametime
    K = acf.ct.size
    M = K + int(np.ceil(20 * acf.timescales.max() / dt))  # rho has decayed by lag M
    rho = acf_terms(dt * np.arange(M + 1), acf.coeffs, acf.timescales).sum(axis=0)
    p = rho[:K]

    mean = p.copy()
    if mean_shift:
        b = 2.0 * np.sum(acf.coeffs * acf.timescales) / (N * dt)
        mean -= b * (1.0 - mean)

    # Diagonal of Bartlett's covariance: S(0) + S(2h) + 2 rho_h^2 S(0) - 4 rho_h S(h),
    # with S(m) = sum_k rho(k) rho(k+m), computed by FFT.
    r = np.concatenate([rho[:0:-1], rho])  # rho over lags -M..M
    nfft = 1 << (2 * r.size - 1).bit_length()
    f = np.fft.rfft(r, nfft)
    S = np.fft.irfft(f * f.conj(), nfft)[: 2 * K - 1]
    var = S[0] + S[: 2 * K - 1 : 2] + 2.0 * p**2 * S[0] - 4.0 * p * S[:K]
    sigma = np.sqrt(np.clip(var, 0.0, None) / N)

    # y(h) = sum_{k=-M..M} V_k rho(|h-k|), V_k = W_|k|, V_0 = 0; one FFT per batch.
    kern = np.fft.rfft(r, nfft)
    draws = np.empty((num_draws, K))
    for start in range(0, num_draws, batch_size):
        n = min(batch_size, num_draws - start)
        W = rng.standard_normal((n, M))
        V = np.concatenate([W[:, ::-1], np.zeros((n, 1)), W], axis=1)
        y = np.fft.irfft(np.fft.rfft(V, nfft, axis=1) * kern, nfft, axis=1)
        y = y[:, 2 * M : 2 * M + K]  # index 2M <-> lag 0
        draws[start : start + n] = mean + (y - p * y[:, :1]) / np.sqrt(N)

    return DataClass(t=acf.t, draws=draws, mean=mean, sigma=sigma)
