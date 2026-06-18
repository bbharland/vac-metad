import numpy as np
from scipy.optimize import curve_fit


def fit_exponential(x, y, linearize=False):
    """Return least squares estimate for k in exponential PDF:
            y = k * exp(-k * x)

    To avoid "RuntimeWarning: overflow encountered in exp" (overflowing float64 in np.exp), linearize and instead fit
            log(y) = log(k) - k * x
    """
    if linearize:
        x, y = x[y > 0], y[y > 0] # avoid np.inf/np.nan
        k, var = curve_fit(lambda x, k: np.log(k) - k * x, x, np.log(y))
    else:
        k, var = curve_fit(lambda x, k: k * np.exp(-k * x), x, y, p0=0.001)

    return k[0]


def eig_sorted(a):
    """Return
    ------
        w : eigenvalues, sorted largest to smallest, shape (num_eigvals,)
        v : right eigenvectors, sorted, shape (num_eigvals, num_eigvals)
    """
    w, v = np.linalg.eig(a)
    idx = np.argsort(w)[::-1]
    return np.real(w[idx]), np.real(v[:, idx])


def timescale_from_eigval(eigval, lagtime):
    return -lagtime / np.log(eigval)


def rayleigh_quotient(u, lagframes=1):
    """Parameters:
    ----------
    u : ndarray with shape (num_frames,)
        Some generic observable, u(x_t)
    lagframes : int
        How many frames to skip when computing correlation function

    Return:
    ------
    C(k tau) = E[u(x_t) u(x_{t + k tau})] / E[u(x_t) u(x_t)]
    """
    u0 = u[:-lagframes]
    uk = u[lagframes:]
    return np.mean(u0 * uk) / np.mean(u0 * u0)
