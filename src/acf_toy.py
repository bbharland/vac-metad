import numpy as np

from .dataclass import DataClass


def toy_model_acf(taus, coeffs=(0.7, 0.1, 0.1, 0.1), timescales=(1850, 60, 26, 2)):
    """Return computed ACF data for a toy model over grid, 'taus'.

        C(t) = sum_i a_i * exp(-tau / tau_i)

    timescales in ps; coeffs are a_i (= c_i^2), sum to 1 so C(0) = 1.
    """
    terms = [a * np.exp(-taus / ti) for a, ti in zip(coeffs, timescales)]
    return DataClass(
        timescales=np.array(timescales),
        coeffs=np.array(coeffs),
        terms=np.array(terms),
        ct=np.sum(terms, axis=0),
    )


"""
Synthetic ACF estimates from Bartlett's formula (see claude/bartlett-draws.md).

For a stationary GAUSSIAN process with normalised ACF rho(k), the estimate
C_hat(h) from N frames is approximately multivariate normal with

    Cov[C_hat(h), C_hat(l)] ~ (1/N) [ S(h-l) + S(h+l) + 2 rho_h rho_l S(0)
                                      - 2 rho_h S(l) - 2 rho_l S(h) ],     (1)

    S(m) = sum_{k=-oo}^{oo} rho(k) rho(k+m),

(Bartlett 1946; the S-terms are his sums over k rewritten as lagged
self-products of rho).  Subtracting the sample mean adds a deterministic shift

    E C_hat(h) ~ rho(h) - b (1 - rho(h)),     b = 2 tau_int / T.           (2)

Both depend only on the model ACF and T: tau_int = sum_i a_i t_i follows from
the toy parameters, so it is not an input.

CAVEAT: (1) assumes Gaussian statistics.  For a strongly bimodal (two-state)
u it underestimates the noise by ~ 1/(e sqrt(p q)); see the theory note §3.2.
"""

def bartlett_acf_draws(
    coeffs,
    timescales,
    frametime,
    T,
    num_frames_acf,
    num_draws,
    rng=None,
    mean_shift=True,
    batch_size=50,
):
    """Draws of C_hat(tau) for the toy model, as if estimated from length T.

    Same inputs and outputs as before, but no K x K covariance is built.  It
    uses the moving-average form of Bartlett's formula (Brockwell & Davis,
    Time Series: Theory and Methods, Thm 7.2.1):

        sqrt(N) [C_hat(h) - rho(h)] ~ sum_{k>=1} [rho(k+h) + rho(k-h) - 2 rho(h) rho(k)] W_k,

    with W_k iid N(0, 1).  Its covariance is exactly formula (1).  With
    V_k = W_|k| (V_0 = 0) and y = V * rho (a convolution),

        noise(h) = [y(h) - rho(h) y(0)] / sqrt(N),

    so each draw costs one FFT convolution of length ~ K + 20 t_max / dt.
    Memory is O(batch_size * (K + decay)); time is O(num_draws * M log M).

    Parameters
    ----------
    coeffs, timescales : a_i (sum to 1) and t_i of C(tau) = sum_i a_i exp(-tau/t_i)
    frametime : Delta t, same units as timescales and T
    T : trajectory length; n_frames = T / frametime
    num_frames_acf : number of lags (0 ... num_frames_acf-1)
    num_draws : number of synthetic ACFs
    rng : numpy Generator (default: a fresh unseeded one)
    mean_shift : add the mean-subtraction bias (2)
    batch_size : draws generated per FFT batch; lower it to save memory

    Returns
    -------
    draws : ndarray, shape (num_draws, num_frames_acf), draws[:, 0] = 1
    mean : ndarray, the expected estimate rho - b (1 - rho) (or rho)
    sd : ndarray, Bartlett standard deviation per lag
    """
    if rng is None:
        rng = np.random.default_rng()

    coeffs, timescales = np.asarray(coeffs, float), np.asarray(timescales, float)
    n_frames = T / frametime
    K = num_frames_acf
    D = int(np.ceil(20 * timescales.max() / frametime))  # decay length of rho
    M = K + D  # W_k with k > M cannot reach lags < K
    rho = toy_model_acf(frametime * np.arange(M + 1), coeffs, timescales).ct

    mean = rho[:K].copy()
    if mean_shift:
        b = 2.0 * np.sum(coeffs * timescales) / T
        mean -= b * (1.0 - mean)

    # sd from the diagonal of (1): S(0) + S(2h) + 2 rho_h^2 S(0) - 4 rho_h S(h)
    r = np.concatenate([rho[:0:-1], rho])
    nfft_S = 1 << (2 * r.size - 1).bit_length()
    f = np.fft.rfft(r, nfft_S)
    S = np.fft.irfft(f * f.conj(), nfft_S)[: 2 * K - 1]
    p = rho[:K]
    var = S[0] + S[: 2 * K - 1 : 2] + 2.0 * p**2 * S[0] - 4.0 * p * S[:K]
    sd = np.sqrt(np.clip(var, 0.0, None) / n_frames)

    # y(h) = sum_{k=-M..M} V_k rho(|h-k|) for h = 0..K-1, by FFT.
    # V is laid out over k = -M..M; rho kernel over lags -M..M.
    nfft = 1 << (4 * M + 1).bit_length()
    kern = np.fft.rfft(np.concatenate([rho[:0:-1], rho]), nfft)
    scale = 1.0 / np.sqrt(n_frames)

    draws = np.empty((num_draws, K))
    for start in range(0, num_draws, batch_size):
        n = min(batch_size, num_draws - start)
        W = rng.standard_normal((n, M))
        V = np.concatenate([W[:, ::-1], np.zeros((n, 1)), W], axis=1)  # k = -M..M
        y = np.fft.irfft(np.fft.rfft(V, nfft, axis=1) * kern, nfft, axis=1)
        y = y[:, 2 * M : 2 * M + K]  # full-conv index 2M <-> lag h = 0
        draws[start : start + n] = mean + scale * (y - p * y[:, :1])
    return draws, mean, sd
