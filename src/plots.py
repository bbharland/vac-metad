import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss

from .math import fit_exponential


"""
A note for all histograms plotted by matplotlib:

https://stackoverflow.com/questions/43568370/matplotlib-2d-histogram-seems-transposed

"Apart from the fact that there seems to be a mistake concerning the exact shape of the arrays, we see that the first dimension of the returned histogram array is x and the second y.

However, matplotlib always expects y to be the first dimenstion. Therefore, while plt.hist2d produces the correct plot, plt.pcolormesh needs a transposed version of the array."

plt.pcolormesh(X,Y, counts.T)
"""

def plot_dihedrals_hist2d(fig, ax, dihedrals, weights=None, label=''):
    """Parameters:
    -----------
    dihedrals : array with shape (num_fraces, 2)
    weights : array with shape (num_frames,)
    label : str
        In case you want to place text on the plot
    pos : list-like (x, y)
        Location of label, if there is one
    """
    h = ax.hist2d(*dihedrals.T, bins=75, weights=weights, density=True,
                  cmin=1e-10, cmap='magma_r')
    fig.colorbar(h[3], ax=ax)
    ax.patch.set_facecolor('0.6')
    ax.set_xlabel(r'$\phi$', fontsize=12)
    ax.set_ylabel(r'$\psi$', fontsize=12, rotation=0)
    ax.set_xlim([-np.pi, np.pi])
    ax.set_ylim([-np.pi, np.pi])
    if label:
        if len(label) < 8:
            pos = (1, -2.5)
        else:
            pos = (-0.5, -2.5)
        ax.text(*pos, label, fontsize=12)


def plot_cvs_hist2d(fig, ax, s1, s2, cvs, weights=None, label='', pos=None):
    """NB:  ax.hist2d expects bins to be of the form:
            bins = [x_edges, y_edges]

    This is good enough for visualizing the CV histogram, but not for doing anything quantitative with it.

    Parameters:
    -----------
    s1, s2 : arrays with shape (num_points,)
    cvs : array with shape (num_fraces, 2)
    weights : array with shape (num_frames,)
    label : str
        In case you want to place text on the plot
    pos : list-like (x, y)
        Location of label, if there is one
    """
    h, xedges, yedges, im = ax.hist2d(
        *cvs.T, bins=[s1, s2], weights=weights, density=True, cmin=1e-6,
        cmap='magma_r'
    )
    fig.colorbar(im, ax=ax);
    ax.patch.set_facecolor('0.6')
    ax.set_xlabel("$s_1$", fontsize=14)
    ax.set_ylabel("$s_2$", fontsize=14, rotation=0)
    if label:
        assert pos is not None, 'Need to specify a position for the label'
        ax.text(*pos, label, fontsize=12)


def plot_eigfuncs(fig, axs, theta, psi_grid, timescales):
    for i, ax in enumerate(axs.flatten()):
        if i % axs.shape[1] == 0: # left
            ax.set_ylabel(r'$\psi$', fontsize=12, rotation=0)
        if i >= np.prod(axs.shape) - axs.shape[1]: # bottom
            ax.set_xlabel(r'$\phi$', fontsize=12)
        cb = ax.pcolormesh(theta, theta, psi_grid[:, :, i].T)
        fig.colorbar(cb, ax=ax)
        ax.text(1.8, 2.2, f'$\\psi_{i+1}$', fontsize=18)
        timescale = timescales[i]
        if timescale >= 10_000:
            timescale /= 1000
            units = 'ns'
        else:
            units = 'ps'
        ax.text(1.4, 1.2, f'{timescale:.1f} {units}')


def plot_histogram(*args, **kwargs):
    """Plot 2D histogram.  This now wraps 'plot_colormesh'.
    """
    # if kwargs['cvs'] and 'vmax' not in kwargs:
    #     kwargs['vmax'] = 0.7
    if 'cmap' not in kwargs:
        kwargs['cmap'] = 'viridis'
    cb = plot_surface(*args, **kwargs)
    return cb


def plot_surface(ax, x, y, z, cvs=False, dih=False, vmin=None, vmax=None, label='', pos=None, pad=0.1, cmap='magma'):
    """Plot z = f(x, y)

    Parameters
    ----------
    x : array with shape (nx,)
    y : array with shape (ny,)
    z : array with shape (nx, ny)

    cvs : Bool
        If True, axes are 's1', 's2'
    dih : Bool
        If True, axes are 'phi', 'psi'.  If both cvs and dih are False, axes are 'x', 'y'
    vmin, vmax : float
        If None, set to np.min(z) or np.max(z)
    label : str
        If not None, will be displayed on graph at position, pos
    pos : [float, float]
        If label and pos are both not None, location of label
    pad : float
        If label is not None but pos is, then this will determine how far from the edge of the graph the label will appear.
    """
    def spos(x, pad):
        return min(x) + pad * np.ptp(x)

    if vmin is None:
        vmin = np.min(z)
    if vmax is None:
        vmax = np.max(z)
    if cvs:
        ax.set_xlabel("$s_1$", fontsize=14)
        ax.set_ylabel("$s_2$", fontsize=14, rotation=0)
    elif dih:
        ax.set_xlabel("$\\phi$", fontsize=14)
        ax.set_ylabel("$\\psi$", fontsize=14, rotation=0)
    else:
        ax.set_xlabel("$x$", fontsize=14)
        ax.set_ylabel("$y$", fontsize=14, rotation=0)

    cb = ax.pcolormesh(x, y, z.T, vmin=vmin, vmax=vmax, cmap=cmap)
    if label:
        if pos is None:
            pos = spos(x, pad), spos(y, pad)
        color = 'white' if (cmap == 'viridis') else 'black'
        ax.text(*pos, label, color=color, fontsize=12)
    return cb
