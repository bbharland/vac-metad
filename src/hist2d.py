"""Core utilities for working with 2D grids and histograms.

For peak extraction, cropping, and fitting utilities, see peaks.py.
"""

import numpy as np


def sgrids(xrange=[-1.7, 6.7], dx=0.016, yrange=[-1.4, 2.2], dy=0.0):
    """Return the s1, s2 grids that were decided in test-(?).pynb
    """
    return grids_ranges2d([[-1.7, 6.7], [-1.4, 2.2]], grid_spacing=0.016)


def ranges_data(data, pad=0.07):
    """Find the ranges in each dimension of a data matrix (shape = (num_points, dim)).

    Parameters:
    ----------
    data : array, shape (num_points, dim)
        Row-wise data, each with 'dim' elements
    pad : float
        The fraction of (x.max - x.min) to subtract from x.min and add to x.max

    Return:
    -------
    [[xmin, xmax], [ymin, ymax], ...]
    """
    def xrange(x, pad):
        dx = pad * np.ptp(x)
        return [np.min(x) - dx, np.max(x) + dx]

    return [xrange(data[:, n], pad) for n in range(data.shape[1])]


def grid_from_range(range, dx=None, num_points=100):
    """Return a grid (over range) based on either:

    dx : float = grid spacing
    num_points: int = length of grid
    """
    if dx is not None:
        return np.arange(*ranges[0], dx)
    else:
        return np.linspace(*ranges[0], num_points_x)


def grid_da(x, y):
    """Return the area element of finite grid
    """
    return (x[1] - x[0]) * (y[1] - y[0])


def grid_norm(x, y, z):
    """Return the integral under grid, z = f(x, y)
    """
    return grid_da(x, y) * np.sum(z)


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


def histogram2d(x, y, data, weights=None, density=True, check_bin_edges=False):
    """Compute the p(x, y) from the data

    Parameters
    ----------
    x, y : array, shape (num_points,)
        Grid of points defining CENTERS of bins
    data : array, shape (num_frames, 2)
        Data with rows (xi, yi)
    weights : array, shape (num_points,)
        If None, weights = 1
    check_bin_edges : Bool
        If True, use np.linspace to check bin edges

    Return
    ------
    H : array, shape (num_points, num_points)
        The distribution from histogram2d
            H_ij = p(x_i, y_j)

        ** to data.py
    """
    def bin_edges(x, do_linspace=False):
        dx = x[1] - x[0]
        xmin = x[0] - dx / 2
        xmax = x[-1] + dx / 2
        if do_linspace:
            return np.linspace(xmin, xmax, len(x) + 1)
        else:
            return xmin, xmax, len(x)

    xmin, xmax, xnum_bins = bin_edges(x)
    ymin, ymax, ynum_bins = bin_edges(y)

    bins = [xnum_bins, ynum_bins]
    ranges = [[xmin, xmax], [ymin, ymax]]

    H, xedges, yedges = np.histogram2d(
        data[:, 0], data[:, 1], bins=bins,
        range=ranges, density=density, weights=weights
        )
    if check_bin_edges:
        xbin_edges = bin_edges(x, do_linspace=True)
        if np.all(xedges == xbin_edges):
            print('np.histogram2d gave the correct x bin edges/grid')
        else:
            print('np.histogram2d, bad bin edges, x')

        ybin_edges = bin_edges(y, do_linspace=True)
        if np.all(yedges == ybin_edges):
            print('np.histogram2d gave the correct y bin edges/grid')
        else:
            print('np.histogram2d, bad bin edges, y')
    return H


def range_indexes(x, xmin, xmax):
    """Return the indices that correspond to a range in a 1D array

    Parameters:
    -----------
    x : array, shape (num_points)
    xmin : float
    xmax : float

    Return:
    -------
    (imin, imax) : the lower and upper index such that xmin < x < xmax
    """
    return np.where(x > xmin)[0][0], np.where(x < xmax)[0][-1]


def subarray1d(x, irange):
    """Return a view into 1D grid array, x, which includes all values in rangex=(xmin, xmax)

    Parameters:
    -----------
    x : array, shape (num_points,)
    irange : list(int) = [imin, imax]

    TODO: verify this is dead code.  See subarray2d.
    """
    assert isinstance(irange, tuple), (
        f'{type(irange) = } must be tuple'
    )
    return x[irange[0]:irange[1] + 1]


def subarray2d(a, irangex, irangey):
    """Return a view into 2D array, a, which includes all indices in irangex, irangey (inclusive of last point)

    Parameters:
    -----------
    irangex : tuple(int) = (ixmin, ixmax)
    irangey : tuple(int) = (iymin, iymax)
    a : array with shape (num_points, num_points)

    TODO: verify this is dead code?  Check that doing things with slices does indeed add 1 to slice.stop
    """
    assert isinstance(irangex, tuple), (
        f'{type(irangex) = } must be tuple'
    )
    assert isinstance(irangey, tuple), (
        f'{type(irangey) = } must be tuple'
    )
    return a[irangex[0]:irangex[1] + 1, irangey[0]:irangey[1] + 1]


def slices_from_ranges(x, y, ranges):
    rangex, rangey = ranges
    return (
        slice(*range_indexes(x, *rangex)),
        slice(*range_indexes(y, *rangey))
    )


def argopt2d(a, op):
    assert len(a.shape) == 2, (
        f'Only use for 2d arrays.  Found {len(a.shape)=}'
    )
    assert op in ('min', 'max'), (
        f'Only do min or max, found {op=}'
    )
    f = {'min': np.argmin, 'max': np.argmax}
    return np.unravel_index(f[op](a), a.shape)
    # iloc = np.unravel_index(f[op](a), a.shape)
    # return tuple(int(i) for i in iloc) # avoid annoying np.int64 repr


def argmin2d(a):
    return argopt2d(a, 'min')


def argmax2d(a):
    return argopt2d(a, 'max')
