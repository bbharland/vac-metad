"""
Helpers for notes/timescale-plan/timescale-plan.md.

Drop this file into vac_metad/ (or a notebook cell) and import from it.  It only
glues together functions that already exist:

    acf.compute_acf_dataclass        C(tau) from a time series
    acf_analysis.*                   timescales, multi-exponential fit, slow-mode split
    twostate.core_entry_indices etc. committed two-state trajectory

DataClass in, DataClass out, except rc_analysis, which returns a dict of
DataClass objects (a DataClass can't hold another DataClass).  Cache it with
@cached(sd, "....pickle").
"""

import numpy as np
import matplotlib.pyplot as plt

from .dataclass import DataClass
from .acf import compute_acf_dataclass
from .acf_analysis import (
    acf_input,
    acf_timescales,
    slow_mode_analysis,
    fit_multi_exponential,
    local_timescales,
    spread_over_draws,
)
from .twostate import (
    core_entry_indices,
    two_state_model_from_entries,
    committed_trajectory_from_entries,
)


# --------------------------------------------------------------------------- #
#  One-time statistics of u
# --------------------------------------------------------------------------- #
def two_point_check(u):
    """How close u is to a two-valued (telegraph) variable.

    A two-point variable with mean 0, variance 1 and rare-state population q
    has excess kurtosis = skew^2 - 2, so kurt_gap = 0.  Its q follows from the
    skew alone: q = (1 - skew / sqrt(skew^2 + 4)) / 2.
    """
    z = (u - u.mean()) / u.std()
    skew = float(np.mean(z**3))
    exkurt = float(np.mean(z**4) - 3.0)
    return DataClass(
        skew=skew,
        exkurt=exkurt,
        kurt_gap=exkurt - (skew**2 - 2.0),
        q_skew=0.5 * (1.0 - abs(skew) / np.sqrt(skew**2 + 4.0)),
    )


def committed_state(phi, frametime, core0=None, core1=None):
    """Committed two-state trajectory from phi cores (default: first-pass cores).

    core0 = C7eq/beta: phi < -0.5 or phi > 2.8;  core1 = C7ax/alpha_L: 0.5 < phi < 2.0.
    These are independent of any SRV, so they give an outside reference for psi_1.

    Returns DataClass with state (int array, 0/1), n_round_trips, q, t1 (=tau_relax),
    k01, k10.
    """
    if core0 is None:
        core0 = (phi < -0.5) | (phi > 2.8)
    if core1 is None:
        core1 = (phi > 0.5) & (phi < 2.0)
    e1, e0 = core_entry_indices(core0, core1)
    m = two_state_model_from_entries(e1, e0, phi.size, frametime)
    state = committed_trajectory_from_entries(e1, e0, phi.size)
    k01, k10 = m["rates"]
    return DataClass(
        state=np.asarray(state, float),
        n_round_trips=float(min(m["n01"], m["n10"])),
        q=float(min(m["pi"])),
        t1=float(m["tau_relax"]),
        k01=float(k01),
        k10=float(k10),
    )


def between_state_fraction(u, state):
    """corr(u, state)^2 = Var(E[u | state]) / Var(u): the share of u's variance
    carried by the two-state switch.  For u = c_1 psi_1 + ... with psi_1 ~ the
    standardised state indicator, this estimates a_1 without any fit."""
    return float(np.corrcoef(u, state)[0, 1] ** 2)


# --------------------------------------------------------------------------- #
#  ACF + errors + all three analyses for one RC
# --------------------------------------------------------------------------- #
def block_acfs(u, frametime, num_frames_acf, num_blocks=8, kind="linear"):
    """ACF of each of num_blocks contiguous blocks.

    Returns DataClass with cts (num_blocks, num_frames_acf), ct_err (standard
    error of the block mean, per lag) and ct_sd (per-block sd, per lag).
    Blocks must be many t_1 long (Parrinello: 11 us = 380 t_1; solvated:
    375 ns = 200 t_1; OpenMM vacuum: 1.9 us = 50 t_1, marginal).
    """
    L = len(u) // num_blocks
    cts = np.array([
        compute_acf_dataclass(u[i * L:(i + 1) * L], frametime, num_frames_acf, kind=kind).ct
        for i in range(num_blocks)
    ])
    sd = cts.std(axis=0, ddof=1)
    return DataClass(cts=cts, ct_sd=sd, ct_err=sd / np.sqrt(num_blocks))


def rc_analysis(u, frametime, num_frames_acf, kind="linear", num_blocks=8,
                n_terms=(1, 2, 3, 4), split_window=None, noise_factor=4.5,
                fit_kwargs=None):
    """Everything for one reaction coordinate u.

    C(tau) comes from the whole trajectory; its error bar sigma_C(tau) comes
    from the spread of block ACFs (this captures two-state noise, which the
    Gaussian Bartlett formula underestimates 4-5x).  Rough error bars on a1,
    t1, tau_R, ... come from running slow_mode_analysis on each block; for a
    two-state u they understate the a1 error, so prefer null_spread.

    num_blocks : None skips the blocks and uses the Bartlett sigma_C times
        noise_factor instead (for runs with too few transitions per block,
        e.g. OpenMM vacuum with 9 round trips).
    split_window : (t_lo, t_hi), passed to slow_mode_analysis.  a1 and R(t)
        then come from a short fit just past the fast modes (much more
        precise, see the plan, section 2.3); t1 still comes from the tail.

    Returns dict of DataClass:
        acf    : acf_input output (t, ct, sigma, frametime, n_frames)
        ts     : acf_timescales  (tau_0, tau_int, t1, a1, purity, ...)
        sm     : slow_mode_analysis (a1, t1, tau_R, gap, identity, R(t), ...)
        local  : local_timescales (chord, tangent)
        blocks : slow_mode_analysis per block -> x, x_mean, x_sd for each field
        multi_n: fit_multi_exponential with n terms, for n in n_terms
    """
    fit_kwargs = fit_kwargs or {}
    full = compute_acf_dataclass(u, frametime, num_frames_acf, kind=kind)
    if num_blocks is None:
        blk = None
        acf = acf_input(full, frametime, n_frames=len(u), noise_factor=noise_factor)
    else:
        blk = block_acfs(u, frametime, num_frames_acf, num_blocks, kind)
        acf = acf_input(full, frametime, n_frames=len(u), ct_err=blk.ct_err)

    out = {
        "acf": acf,
        "ts": acf_timescales(acf, **fit_kwargs),
        "sm": slow_mode_analysis(acf, split_window=split_window, **fit_kwargs),
        "local": local_timescales(acf),
        "blocks": None if blk is None else spread_over_draws(
            blk.cts, frametime, slow_mode_analysis, ct_err=blk.ct_sd,
            split_window=split_window, **fit_kwargs),
    }
    for n in n_terms:
        try:
            out[f"multi_{n}"] = fit_multi_exponential(acf, n, sum_to_one=True)
        except RuntimeError as e:
            print(f"[warn] {n}-exponential fit failed: {e}")
    return out


def error_on_full(res, key):
    """1-sigma error of the full-trajectory estimate of `key` from the block
    spread: sd over blocks / sqrt(num_blocks)."""
    b = res["blocks"]
    if b is None or not hasattr(b, key):
        return np.nan
    n = np.sum(np.isfinite(getattr(b, key)))
    return getattr(b, f"{key}_sd") / np.sqrt(n)


# --------------------------------------------------------------------------- #
#  Printing
# --------------------------------------------------------------------------- #
SUMMARY_KEYS = ("t1", "a1", "tau_int", "tau_R", "gap", "slow_share", "identity_error")


def summary_row(label, res, keys=SUMMARY_KEYS):
    """One markdown table row: value ± block error for each key."""
    sm = res["sm"]
    cells = [label]
    for k in keys:
        v = getattr(sm, k)
        e = error_on_full(res, k)
        cells.append(f"{v:.4g} ± {e:.2g}" if np.isfinite(e) else f"{v:.4g}")
    return "| " + " | ".join(cells) + " |"


def summary_table(results, keys=SUMMARY_KEYS):
    """results: dict label -> rc_analysis output.  Prints a markdown table."""
    print("| RC | " + " | ".join(keys) + " |")
    print("|---" * (len(keys) + 1) + "|")
    for label, res in results.items():
        print(summary_row(label, res, keys))


def print_multi(res):
    """The n-exponential fits side by side, with condition numbers."""
    for k, m in res.items():
        if not k.startswith("multi_"):
            continue
        a = ", ".join(f"{x:.3f}" for x in m.coeffs)
        t = ", ".join(f"{x:.4g}" for x in m.timescales)
        print(f"n={k[6:]}: a=[{a}]  t=[{t}]  cond={m.cond:.2g}  "
              f"chi2/dof={m.chi2_dof:.3g}  distinct={m.n_distinct:.0f}")


# --------------------------------------------------------------------------- #
#  Plots
# --------------------------------------------------------------------------- #
def plot_rc(res, title="", slow_unit=1000.0, slow_label="ns", fast_max=None):
    """Four panels for one RC.

    (a) C(tau) with the slow fit a1 exp(-t/t1) and the +-sigma band
    (b) ln C(tau): the fit window [t_min, t_max] and the 3 sigma noise floor
    (c) chord and tangent timescales vs lag (log axis), with t1 and tau_int
    (d) residual r(t) = R(t)/(1-a1) and its running integral (-> tau_R)

    Times in the data are ps; slow_unit/slow_label rescale panels (a)-(c).
    fast_max (ps) sets the x range of panel (d); default 10 * R_window.
    """
    acf, sm, loc = res["acf"], res["sm"], res["local"]
    t, ct = acf.t, acf.ct
    ts = t / slow_unit
    fit = sm.a1 * np.exp(-t / sm.t1)
    sel = t <= 1.5 * sm.t_max

    fig, axs = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    ax = axs[0, 0]
    ax.plot(ts[sel], ct[sel], lw=1.5, label=r"$\hat C(\tau)$")
    if acf.sigma is not None:
        ax.fill_between(ts[sel], (ct - acf.sigma)[sel], (ct + acf.sigma)[sel],
                        alpha=0.3, lw=0)
    ax.plot(ts[sel], fit[sel], "k--", lw=1,
            label=rf"$a_1 e^{{-\tau/t_1}}$, $a_1$={sm.a1:.3f}, $t_1$={sm.t1 / slow_unit:.3g} {slow_label}")
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xlabel(rf"$\tau$ ({slow_label})")
    ax.set_ylabel(r"$C(\tau)$")
    ax.legend(fontsize=8, frameon=False)

    ax = axs[0, 1]
    pos = sel & (ct > 0)
    ax.semilogy(ts[pos], ct[pos], lw=1.5)
    ax.semilogy(ts[sel], fit[sel], "k--", lw=1)
    if acf.sigma is not None:
        ax.semilogy(ts[sel], 3 * acf.sigma[sel], color="tab:red", lw=1, label=r"$3\sigma_C$")
    for x in (sm.t_min, sm.t_max):
        ax.axvline(x / slow_unit, color="gray", ls=":", lw=1)
    ax.set_xlabel(rf"$\tau$ ({slow_label})")
    ax.set_ylabel(r"$C(\tau)$ (log)")
    ax.legend(fontsize=8, frameon=False)

    ax = axs[1, 0]
    ok = t > 0
    ax.semilogx(ts[ok], loc.chord[ok] / slow_unit, label="chord $-\\tau/\\ln C$")
    ax.semilogx(ts[ok], loc.tangent[ok] / slow_unit, label="tangent $-1/(d\\ln C/d\\tau)$")
    ax.axhline(sm.t1 / slow_unit, color="k", ls="--", lw=1, label="$t_1$")
    ax.axhline(sm.tau_int / slow_unit, color="k", ls=":", lw=1, label=r"$\tau_{\rm int}$")
    ax.set_ylim(0, 1.5 * sm.t1 / slow_unit)
    ax.set_xlabel(rf"$\tau$ ({slow_label})")
    ax.set_ylabel(f"timescale ({slow_label})")
    ax.legend(fontsize=8, frameon=False)

    ax = axs[1, 1]
    w = 1.0 - sm.a1
    fast_max = fast_max or 10 * max(sm.R_window, acf.frametime)
    f = t <= fast_max
    ax.plot(t[f], sm.R[f] / w, ".-", ms=3, lw=1, label=r"$r(t)=R(t)/(1-a_1)$")
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(sm.R_window, color="gray", ls=":", lw=1)
    ax.set_xlabel(r"$t$ (ps)")
    ax.set_ylabel(r"$r(t)$")
    ax2 = ax.twinx()
    run = np.concatenate([[0], np.cumsum(0.5 * acf.frametime * (sm.R[1:] + sm.R[:-1]))]) / w
    ax2.plot(t[f], run[f], color="tab:orange", lw=1)
    ax2.set_ylabel(r"$\int_0^t r$ (ps)", color="tab:orange")
    ax.set_title(rf"$\tau_R$ = {sm.tau_R:.3g} ps,  $g=t_1/\tau_R$ = {sm.gap:.3g}", fontsize=9)
    ax.legend(fontsize=8, frameon=False, loc="center right")

    fig.suptitle(title)
    return fig, axs


def plot_ranking(results, ax=None):
    """a1 against log10 gap, one point per RC, with block error bars."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    for label, res in results.items():
        sm = res["sm"]
        ea = error_on_full(res, "a1")
        lg = np.log10(sm.gap)
        elg = error_on_full(res, "log_gap") / np.log(10)
        ax.errorbar(lg, sm.a1, xerr=elg, yerr=ea, fmt="o", capsize=3, label=label)
    ax.set_xlabel(r"$\log_{10} g = \log_{10}(t_1/\tau_R)$")
    ax.set_ylabel(r"$a_1$ (signal strength)")
    ax.legend(fontsize=8, frameon=False)
    return ax


# --------------------------------------------------------------------------- #
#  Synthetic stand-in for psi_1: telegraph + fast noise
# --------------------------------------------------------------------------- #
def simulate_telegraph(k01, k10, n_frames, frametime, rng):
    """Two-state Markov jump process sampled every frametime, values {0, 1}."""
    T = n_frames * frametime
    s = int(rng.random() < k01 / (k01 + k10))
    edges, states, t = [], [], 0.0
    while t < T:
        n = int(2 * T * (k01 * k10) / (k01 + k10)) + 20
        rates = np.where(np.arange(n) % 2 == 0, [k01, k10][s], [k10, k01][s])
        e = t + np.cumsum(rng.exponential(1.0 / rates))
        st = (s + np.arange(n)) % 2
        edges.append(e)
        states.append(st)
        t, s = e[-1], (st[-1] + 1) % 2
    edges, states = np.concatenate(edges), np.concatenate(states)
    idx = np.searchsorted(edges, np.arange(n_frames) * frametime, side="right")
    return states[idx].astype(float)


def synthetic_psi1(t1, q, a1, t_fast, n_frames, frametime, rng):
    """u = sqrt(a1) * standardised telegraph + sqrt(1-a1) * AR(1) noise.

    Exact ACF: a1 exp(-t/t1) + (1-a1) exp(-t/t_fast), with two-state
    statistics in the slow part.  A null model for psi_1 with known answers:
    tau_R = t_fast, gap = t1 / t_fast.
    """
    k01, k10 = q / t1, (1.0 - q) / t1            # rate into the rare state is k01
    s = simulate_telegraph(k01, k10, n_frames, frametime, rng)
    s = (s - s.mean()) / s.std()
    phi = np.exp(-frametime / t_fast)
    eps = rng.standard_normal(n_frames) * np.sqrt(1 - phi**2)
    x = np.empty(n_frames)
    x[0] = rng.standard_normal()
    for i in range(1, n_frames):                  # fine for a few 1e6 frames
        x[i] = phi * x[i - 1] + eps[i]
    return np.sqrt(a1) * s + np.sqrt(1.0 - a1) * x


def null_spread(t1, q, a1, t_fast, n_frames, frametime, num_frames_acf,
                num_reps=20, rng=None, noise_factor=4.5, **analysis_kwargs):
    """Parametric bootstrap: slow_mode_analysis on num_reps synthetic psi_1
    replicas with the measured (t1, q, a1) and a fast time t_fast.

    The replica sd of each field is the honest error bar for a dataset of
    this length and number of transitions.  Returns spread_over_draws output
    (x, x_mean, x_sd per field).  Each replica of 3e6 frames takes ~1 s.
    """
    rng = np.random.default_rng() if rng is None else rng
    cts = np.array([
        compute_acf_dataclass(
            synthetic_psi1(t1, q, a1, t_fast, n_frames, frametime, rng),
            frametime, num_frames_acf).ct
        for _ in range(num_reps)
    ])
    return spread_over_draws(cts, frametime, slow_mode_analysis, n_frames=n_frames,
                             noise_factor=noise_factor, **analysis_kwargs)
