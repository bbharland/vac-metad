import matplotlib.pyplot as plt
import numpy as np


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
