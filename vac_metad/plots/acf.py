import matplotlib.pyplot as plt
import numpy as np

from ..math import running_trapezoid
from ..acf.analysis import acf_sigma
from .style import CURVE, REF, FILL

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


def plot_multi_fits(acf, multi, noise_factor=1.0, t_plot=None):
    """Goodness of fit for a set of multi-exponential fits.

    Panels
    ------
    1. C(t) and each fit, log y.  The band is C +- sigma_C (if noise info).
    2. Residuals against log t, since the fits differ at the fast modes.
       With noise info: z(t) = (C - fit) / sigma_C, with z = 0, +-1 marked.
       A good fit stays inside +-1 with no systematic swing.
       Noise-free: |C - fit| on log y.
    3. Spectrum: a_i against t_i (log x), with the fit's error bars, and the
       true (t_i, a_i) as black open circles if acf has coeffs/timescales.
       The legend gives cond and chi^2/dof per n.

    Parameters
    ----------
    acf : ACF DataClass (t, ct, frametime, num_frames, optional ct_err)
    multi : dict {n: fit_multi_exponential result}
    noise_factor : passed to acf_sigma
    t_plot : x range of panels 1-2 (default: largest fit t_max)

    Returns
    -------
    fig, axes
    """
    LW, ALPHA = CURVE["lw"], CURVE["alpha"]

    sigma = acf_sigma(acf, noise_factor)
    if t_plot is None:
        t_plot = max(m.t_max for m in multi.values())
    sel = acf.t <= t_plot
    t, c = acf.t[sel], acf.ct[sel]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4.8))
    colors = plt.cm.viridis(np.linspace(0.0, 0.85, len(multi)))

    # 1. data and fits
    ax1.plot(t, c, color="k", lw=LW, alpha=0.3, label=r"$\hat C$")
    if sigma is not None:
        s = sigma[sel]
        ax1.fill_between(t, c - s, c + s, color="k", alpha=0.15, lw=0)
    for col, (n, m) in zip(colors, multi.items()):
        ax1.plot(t, m.fit[sel], color=col, lw=LW, alpha=ALPHA, label=f"n={n}")
    ax1.set_yscale("log")
    ax1.set_ylim(max(c[c > 0].min(), 1e-4) / 2, 1.05)
    ax1.set_xlabel("t")
    ax1.set_ylabel("C(t)")
    ax1.legend()

    # 2. residuals
    for col, (n, m) in zip(colors, multi.items()):
        r = (c - m.fit[sel])[1:]  # skip t = 0 (log x)
        if sigma is None:
            ax2.plot(t[1:], np.abs(r), color=col, lw=LW, alpha=ALPHA, label=f"n={n}")
        else:
            ax2.plot(
                t[1:], r / sigma[sel][1:], color=col, lw=LW, alpha=ALPHA, label=f"n={n}"
            )
        ax2.axvline(m.t_max, color=col, ls=":", lw=1.5)
    if sigma is None:
        ax2.set_yscale("log")
        ax2.set_ylabel(r"$|\hat C - $fit$|$")
    else:
        for y, ls in ((0, "-"), (1, "--"), (-1, "--")):
            ax2.axhline(y, ls=ls, **REF)
        ax2.set_ylabel(r"$(\hat C - $fit$)\,/\,\sigma_C$")
    ax2.set_xscale("log")
    ax2.set_xlabel("t")
    ax2.legend()

    # 3. spectrum
    if hasattr(acf, "timescales") and hasattr(acf, "coeffs"):
        ax3.plot(
            acf.timescales,
            acf.coeffs,
            "o",
            ms=14,
            mfc="none",
            mec="k",
            mew=1.5,
            label="true",
        )
    for col, (n, m) in zip(colors, multi.items()):
        ax3.errorbar(
            m.timescales,
            m.coeffs,
            xerr=m.timescales_err,
            yerr=m.coeffs_err,
            fmt="o",
            ms=8,
            color=col,
            alpha=ALPHA + 0.2,
            capsize=4,
            lw=2,
            label=f"n={n}: cond={m.cond:.1g}, "
            + r"$\chi^2$/dof"
            + f"={m.chi2_dof:.2g}",
        )
    ax3.set_xscale("log")
    ax3.set_xlabel(r"$t_i$")
    ax3.set_ylabel(r"$a_i$")
    ax3.legend(fontsize=9)

    fig.tight_layout()
    return fig, (ax1, ax2, ax3)


def plot_bartlett_draws(b, k_sigma=3.0):
    """Synthetic ACF draws with the mean and a ±k_sigma Bartlett band.

    b : DataClass from bartlett_draws (t, draws, mean, sigma).
    Left: linear y.  Right: log y.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    lo, hi = b.mean - k_sigma * b.sigma, b.mean + k_sigma * b.sigma
    for ax in axes:
        ax.fill_between(b.t, lo, hi, **FILL, label=rf"$\pm{k_sigma:g}\sigma_C$")
        ax.plot(b.t, b.draws.T, color="C0", lw=0.5)
        ax.plot(b.t, b.mean, **REF, label="mean")
        ax.set_xlabel("t (ps)")
    axes[0].set_ylabel(r"$\hat C(t)$")
    axes[0].axhline(0, **REF, ls=":")
    axes[1].set_yscale("log")
    axes[1].set_ylim(np.min(b.sigma[1:]) / 10, 1.2)
    axes[0].legend()
    fig.tight_layout()
    return fig, axes
