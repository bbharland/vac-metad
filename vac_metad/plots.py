import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss

from .dataclass import DataClass
from .math import fit_exponential


"""
A note for all histograms plotted by matplotlib:

https://stackoverflow.com/questions/43568370/matplotlib-2d-histogram-seems-transposed

"Apart from the fact that there seems to be a mistake concerning the exact shape of the arrays, we see that the first dimension of the returned histogram array is x and the second y.

However, matplotlib always expects y to be the first dimenstion. Therefore, while plt.hist2d produces the correct plot, plt.pcolormesh needs a transposed version of the array."

plt.pcolormesh(X,Y, counts.T)
"""

def plot_dihedrals_hist2d(fig, ax, data, weights=None, label=""):
    """'data' can either be the dihedrals dataset, or a precomputed histogram, type DataClass with (x, y, p)
    """
    cmin = 1e-10

    if isinstance(data, np.ndarray):
        dihedrals = data
        H = ax.hist2d(
            *dihedrals.T,
            bins=100,
            weights=weights,
            density=True,
            cmin=cmin,
            cmap="magma_r",
        )
        fig.colorbar(H[3], ax=ax)
    elif isinstance(data, DataClass):
        pdf = data
        H = np.ma.masked_less(pdf.p, cmin)
        im = ax.pcolormesh(pdf.x, pdf.y, H.T, cmap="magma_r")
        fig.colorbar(im, ax=ax)

    ax.patch.set_facecolor("0.6")
    ax.set_xlabel(r"$\phi$", fontsize=12)
    ax.set_ylabel(r"$\psi$", fontsize=12, rotation=0)
    ax.set_xlim([-np.pi, np.pi])
    ax.set_ylim([-np.pi, np.pi])
    if label:
        if len(label) < 8:
            pos = (1, -2.5)
        else:
            pos = (-0.5, -2.5)
        ax.text(*pos, label, fontsize=12)


def plot_cvs_hist2d(fig, ax, data, weights=None, label="", pos=None):
    """If passing data, histogra uses approximation [s1, s2] ~ [x_edges, y_edges] and is not quantitatively accurate.

    Parameters:
    -----------
    data: (tuple, DataClass)
        If tuple, expects (x, y, cvs), where (x, y) are the grids and cvs is the data array, shape (num_frames, num_cvs)
    weights : array with shape (num_frames,)
    label : str
        In case you want to place text on the plot
    pos : list-like (x, y)
        Location of label, if there is one
    """
    cmin = 1e-6

    if isinstance(data, DataClass):
        H = np.ma.masked_less(data.p, cmin)
        im = ax.pcolormesh(data.x, data.y, H.T, cmap="magma_r")
    else:
        (x, y, cvs) = data
        H, _, _, im = ax.hist2d(
            *cvs.T, bins=[x, y], weights=weights, density=True, cmin=cmin, cmap="magma_r"
        )
    fig.colorbar(im, ax=ax)
    ax.patch.set_facecolor("0.6")
    ax.set_xlabel("$s_1$", fontsize=14)
    ax.set_ylabel("$s_2$", fontsize=14, rotation=0)
    if label:
        assert pos is not None, "Need to specify a position for the label"
        ax.text(*pos, label, fontsize=12)


def plot_dihedrals_cvs(sd):
    fig, axs = plt.subplots(1, 2, figsize=(6.5, 3))
    plot_dihedrals_hist2d(fig, axs[0], sd.dihedrals_pdf)
    plot_cvs_hist2d(fig, axs[1], sd.cvs_pdf)
    fig.tight_layout()


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


def plot_acf(ax, ns, acf, time_eigval, num_frames, label=True, xlabel=True):
    t = ns[:num_frames]
    ct = acf.ct[:num_frames]
    time_acf = acf.tau / 1000

    ax.plot(t, ct, alpha=0.4, lw=3)
    if label:
        ax.text(
            t[num_frames // 2],
            0.8,
            rf"$\tau_1$ = {time_eigval:.1f} ns,  $\tau_\mathrm{{int}}$={time_acf:.1f} ns",
        )
    if xlabel:
        ax.set_xlabel(r"$\tau$ (ns)")
    ax.set_ylabel(r"$C(\tau)$")


def plot_theta(ax, ns, theta, time_eigval, time_acf, num_frames, legend=True):
    t = ns[:num_frames]
    theta = theta[:num_frames] / 1000

    ax.plot(t, theta, alpha=0.4, lw=3)
    label = rf"$\tau_1$ = {time_eigval:.1f} ns"
    ax.axhline(time_eigval, ls=":", color="k", lw=1, label=label)
    label = rf"$\tau_\mathrm{{int}}$ = {time_acf:.1f} ns"
    ax.axhline(time_acf, ls="--", color="k", lw=1, label=label)
    ax.set_xlabel(r"$\tau$ (ns)")
    ax.set_ylabel(r"$\theta(\tau)$ (ns)")
    if legend:
        ax.legend()


def plot_acf_theta(ns, acf, theta, time_eigval, num_frames):
    time_acf = acf.tau / 1000
    fig, axs = plt.subplots(2, 1, figsize=(6, 4), sharex=True)
    plot_acf(axs[0], ns, acf, time_eigval, num_frames, label=False, xlabel=False)
    plot_theta(axs[1], ns, theta, time_eigval, time_acf, num_frames, legend=False)
    axs[0].legend(*axs[1].get_legend_handles_labels())
    fig.tight_layout()
