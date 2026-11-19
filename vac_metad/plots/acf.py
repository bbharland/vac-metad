import matplotlib.pyplot as plt
import numpy as np

from ..math import running_trapezoid
from .style import CURVE, REF

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


def plot_timescales_slow_mode(acf, ts, sm):
    t, ct = acf.t, acf.ct

    fig, axs = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)

    # (a) ln C with the slow-tail fit and its window
    ax = axs[0, 0]
    ax.semilogy(t, ct, **CURVE, label="C(t)")
    ax.semilogy(
        t, ts.a1 * np.exp(-t / ts.t1), ls="--", **REF, label=f"a1 e^(-t/t1), t1={ts.t1:.0f}"
    )
    ax.axvspan(ts.t_min, ts.t_max, alpha=0.1, label="fit window")
    ax.set(xlabel="t (ps)", ylabel="C(t)", title="Slow mode", ylim=(1e-4, 1))
    ax.legend()

    # (b) running integral of C: Sokal vs tail-corrected tau_int
    ax = axs[0, 1]
    ax.plot(t, running_trapezoid(ct, acf.frametime), **CURVE, label="∫₀ᵗ C")
    ax.axhline(ts.tau_int, ls="--", **REF, label=f"tau_int={ts.tau_int:.0f}")
    ax.axhline(ts.tau_int_sokal, ls=":", **REF, label=f"Sokal={ts.tau_int_sokal:.0f}")
    ax.axvline(ts.tau_int_window, ls=":", **REF)
    ax.set(xlabel="t (ps)", ylabel="ps", title="Integrated time")
    ax.legend()

    # (c) residual r(t) = R / (1 - a1), fast part
    w = 1 - sm.a1
    ax = axs[1, 0]
    ax.plot(t, sm.R / w, **CURVE, label="r(t)")
    ax.axvspan(300, 1200, alpha=0.1, label="split_window")
    ax.axhline(0, c="k", lw=0.5)
    ax.set(xscale="log", xlim=(1, 3000), xlabel="t (ps)", ylabel="r(t)", title="Residual")
    ax.legend()

    # (d) running integral of r: tau_R
    ax = axs[1, 1]
    ax.plot(t, running_trapezoid(sm.R / w, acf.frametime), **CURVE)
    ax.axhline(sm.tau_R, ls="--", **REF, label=f"tau_R={sm.tau_R:.1f}, gap={sm.gap:.0f}")
    ax.axvline(sm.R_window, ls=":", **REF, label="R_window")
    ax.set(xscale="log", xlim=(1, 3000), xlabel="t (ps)", ylabel="ps", title="Fast time")
    ax.legend()
