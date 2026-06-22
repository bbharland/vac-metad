"""Utilities for cutting out, cropping, and fitting peaks in 2D histograms.

For general grid/histogram utilities, see hist2d_new.py.
"""

from collections import namedtuple

import numpy as np
from scipy.stats import multivariate_normal

from .hist2d_new import argopt2d, grid_norm, histogram2d, slices_from_ranges


def peak_subarrays(x, y, z, ranges):
    """Return views into x, y, and z = f(x, y) corresponding to a single peak.

        ranges = ((xmin, xmax), (ymin, ymax))

    Return:
    -------
    x_subarray : array, shape (xnum,)
    y_subarray : array, shape (ynum,)
    z_subarray : array, shape (xnum, ynum)
    """
    sx, sy = slices_from_ranges(x, y, ranges)
    return x[sx], y[sy], z[sx, sy]


def argopt2d_subarray(x, y, z, ranges, op):
    slicex, slicey = slices_from_ranges(x, y, ranges)
    ix, iy = argopt2d(z[slicex, slicey], op)
    return ix + slicex.start, iy + slicey.start


def argmin2d_subarray(x, y, z, ranges):
    return argopt2d_subarray(x, y, z, ranges, 'min')


def argmax2d_subarray(x, y, z, ranges):
    return argopt2d_subarray(x, y, z, ranges, 'max')


Peak = namedtuple('Peak', 'x y z')


def peaks_from_ranges(x, y, z, ranges, use_namedtuple=False):
    """Return a dictionary of peaks (peak = (xpeak, ypeak, zpeak)) according to the ranges appearing in 'ranges_dict'

    Parameters:
    -----------
    x : array, shape (xnum,)
    y : array, shape (ynum,)
    z : array, shape (xnum, ynum)
    ranges : dict, 'label' ((xmin, xmax), (ymin, ymax))
        Dict containing ranges for all peaks.
    use_namedtuple : Bool
        If true, pack peaks dictionary with namedtuble instead of tuple

    Return:
    -------
    peaks_dict : dict, 'label': (x_subarray, y_subarray, z_subarray)
    """
    peaks = {l: peak_subarrays(x, y, z, r) for l, r in ranges.items()}
    if use_namedtuple:
        return {key: Peak(*val) for key, val in peaks.items()}
    else:
        return peaks


def crop_peak(peak, ipad=0, return_slices=False):
    '''Return peak where the empty parts of the histogram have been cropped out (rectangular crop).

    Parameters
    ----------
    peak : tuple or Peak (named tuple)
        tuple : (x, y, hist)
        Peak : namedtuple('Peak', 'x y z')
    ipad : int
        How many grid points around nonzero histogram to include in subarrays?
    return_slices : Bool
        Return the xgrid, ygrid subarray or slices?
    '''
    if isinstance(peak, Peak):
        x, y, hist = peak.x, peak.y, peak.z
    else:
        x, y, hist = peak

    xis, yis = np.where(hist > 0)
    xslice = slice(np.min(xis) - ipad, np.max(xis) + 1 + ipad)
    yslice = slice(np.min(yis) - ipad, np.max(yis) + 1 + ipad)

    cpeak = (x[xslice], y[yslice], hist[xslice, yslice])
    if isinstance(peak, Peak):
        cpeak = Peak(*cpeak)

    if return_slices:
        cpeak = (cpeak, xslice, yslice)
    return cpeak


def norms_from_peaks(peaks_dict):
    """Return norms of each peak, where peaks_dict is understood to contain PDF grids.
    -------
    norms : dict, 'label': norm
    total_norm : float
    """
    norms = {l: grid_norm(*p) for l, p in peaks_dict.items()}
    return norms, sum([norm for norm in norms.values()])


def fit_gaussian2d(x, y, p):
    """Compute the best fit of a single Gaussian over a single peak from histogram.  In order for the usual formulas to work, the peak must first be renormalized such that
        P[i, j] = Prob(x[i], y[j]) such that np.sum(P) = 1

    Then:
        mean = sum_s P(s) s
        cov = sum_s P(s) (s - mean)(s - mean)'

    Parameters:
    -----------
    x, y : arrays with shape (num_points_x,) and (num_points_y)
    p : subarray of the full histogram array, shape (num_points_x, num_points_y)

    Return:
    -------
    height : float
        The norm, or height scaling factor of the 2d Gaussian to that it matches the peak
    mean, cov : arrays, shape (2,) and (2, 2)
        The fit Gaussian parameters
    """
    height = grid_norm(x, y, p)

    # renormalized array of probabilities
    P = p / np.sum(p)

    # mean of p
    xmean = np.sum(np.sum(P, axis=1) * x)
    ymean = np.sum(np.sum(P, axis=0) * y)
    mean = np.array([xmean, ymean])

    # covariance matrix of p
    cov = np.zeros((2, 2))
    for i in range(len(x)):
        for j in range(len(y)):
            s = np.array([x[i], y[j]])
            cov += P[i, j] * np.outer(s - mean, s - mean)
    # for i, xi in enumrate(x):
    #     for j, yj in enumrate(y):
    #         s = np.array([xi, yj])
    #         cov += P[i, j] * np.outer(s - mean, s - mean)

    return height, mean, cov


def fit_multivariate_normal(x, y, p):
    """Fit a scipy.stats.multivariate_normal object.

    Return:
    -------
    height : float
        The norm, or height scaling factor of the 2d Gaussian
    rv : scipy.stats._multivariate.multivariate_normal_frozen

        rv.mean = array, shape (2,)
        rv.cov = array, shape (2, 2)
    """
    height, mean, cov = fit_gaussian2d(x, y, p)
    return height, multivariate_normal(mean, cov)


def subarrays_hist2d_nonzero(xgrid, ygrid, data, ipad=0, return_slices=False):
    '''Return subarrays (xgrid, ygrid, hist) which is a rectangular subarray where the empty parts of the histogram have been cropped out.

    Parameters
    ----------
    xgrid, ygrids : arrays to compute hist2d from
    data : array, shape (num_points, 2)
        The data to bin into histogram
    ipad : int
        How many grid points around nonzero histogram to include in subarrays?
    return_slices : Bool
        Return the xgrid, ygrid subarray or slices?
    '''
    hist = histogram2d(xgrid, ygrid, data)
    xis, yis = np.where(hist > 0)
    xslice = slice(np.min(xis) - ipad, np.max(xis) + 1 + ipad)
    yslice = slice(np.min(yis) - ipad, np.max(yis) + 1 + ipad)
    if return_slices:
        return xslice, yslice, hist[xslice, yslice]
    else:
        return xgrid[xslice], ygrid[yslice], hist[xslice, yslice]
