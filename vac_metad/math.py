import numpy as np
from scipy.optimize import curve_fit
from scipy.ndimage import uniform_filter1d
from scipy.signal import savgol_filter


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


def kl_divergence(p, q, dx=1):
    """KL(p||q) is only defined if q(x) = 0 implies that p(x) = 0
    Order does matter here since simulation histograms will have 0's!

            KL(p||q) = sum( dx p(x) log(p(x) / q(x)) )

    Parameters:
    -----------
    p, q : arrays with shape (num_points,) or (num_points, num_points)
        Either 1D or 2D probability distributions
    dx : float
        The area element = dx (1D) or dx * dy (2D)
    """
    mask = (p != 0) & (q != 0)
    return dx * np.sum(p[mask] * np.log(p[mask] / q[mask]))
# def kl_divergence(p ,q):
#     with np.errstate(divide='ignore', invalid='ignore'):
#         div = p * np.log(p / q)
#         div[np.isinf(div) | np.isneginf(div) | np.isnan(div)] = 0
#         return div.sum()


def mse(p, q, dx=1):
    """MSE(p, q) = sum( dx MSE(p(x) - q(x))

    Parameters:
    -----------
    p, q : arrays with shape (num_points,) or (num_points, num_points)
        Either 1D or 2D probability distributions
    dx : float
        The area element = dx (1D) or dx * dy (2D)
    """
    return dx * np.sum((p - q) ** 2)


def derivative(t, x, method="central", window=None, polyorder=2, smooth=None):
    """dx/dt on a regular grid; output has the same length as x.

    method : "central"  - 2nd-order central differences, 2nd-order one-sided ends (np.gradient)
             "forward"  - (x[i+1]-x[i])/dt, last value copied from the one before
             "savgol"   - Savitzky-Golay: local polynomial fit over `window` points (odd)
    smooth : optional odd window, moving average of x before differencing (not used with savgol)
    """
    t = np.asarray(t, float); x = np.asarray(x, float)
    dt = t[1] - t[0]
    if not np.allclose(np.diff(t), dt, rtol=1e-8, atol=0):
        raise ValueError("t must be a regular grid")
    if method == "savgol":
        if window is None or window % 2 == 0:
            raise ValueError("savgol needs an odd window")
        return savgol_filter(x, window, polyorder, deriv=1, delta=dt, mode="interp")
    if smooth:
        x = uniform_filter1d(x, smooth, mode="nearest")
    if method == "central":
        return np.gradient(x, dt, edge_order=2)
    if method == "forward":
        d = np.empty_like(x)
        d[:-1] = np.diff(x) / dt
        d[-1] = d[-2]
        return d
    raise ValueError(f"unknown method {method!r}")


def exp_sum(t, coeffs, timescales):
    """Sum of exponentials, f(t) = sum_i a_i exp(-t / t_i).

    t may be a scalar or an array; coeffs and timescales are sequences of the
    same length.  Returns an array shaped like t.
    """
    t = np.asarray(t, float)
    a = np.asarray(coeffs, float).reshape((-1,) + (1,) * t.ndim)
    ts = np.asarray(timescales, float).reshape((-1,) + (1,) * t.ndim)
    return np.sum(a * np.exp(-t[None] / ts), axis=0)


def weighted_linear_fit(x, y, sigma=None, through_origin=False):
    """Weighted least-squares straight line, y = slope * x + intercept.

    Parameters
    ----------
    x, y : 1D arrays
    sigma : None or array like y
        1-sigma error of each y.  Weights are 1/sigma^2 and the returned
        covariance is absolute (not rescaled by the residuals).  If None, all
        weights are 1 and the covariance is rescaled by the residual variance,
        as np.polyfit(..., cov=True) does.
    through_origin : bool
        Force intercept = 0 (one-parameter fit).

    Returns
    -------
    slope, intercept : floats
    cov : ndarray, shape (2, 2), covariance of (slope, intercept).  With
        through_origin the intercept row and column are zero.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    w = np.ones_like(y) if sigma is None else 1.0 / np.asarray(sigma, float) ** 2
    X = x[:, None] if through_origin else np.column_stack([x, np.ones_like(x)])
    A = X.T @ (w[:, None] * X)
    beta = np.linalg.solve(A, X.T @ (w * y))
    cov_beta = np.linalg.inv(A)
    if sigma is None:
        dof = max(y.size - X.shape[1], 1)
        cov_beta = cov_beta * np.sum(w * (y - X @ beta) ** 2) / dof

    cov = np.zeros((2, 2))
    if through_origin:
        cov[0, 0] = cov_beta[0, 0]
        return float(beta[0]), 0.0, cov
    cov[:] = cov_beta
    return float(beta[0]), float(beta[1]), cov


def running_trapezoid(y, dx):
    """Running cumulative trapezoid integral, I[k] = int_0^{k dx} y, with I[0] = 0."""
    y = np.asarray(y, float)
    out = np.zeros_like(y)
    out[1:] = np.cumsum(0.5 * dx * (y[1:] + y[:-1]))
    return out
