"""
Checks to run next to twostate.two_state_model.

    chord_scan          t_u(tau) = -tau / ln C_u(tau) at a handful of lags, for a
                        configurational RC such as psi_1. Rises towards t_1 once tau
                        is past the recrossing / within-state times.
    core_sensitivity    n, q and tau_relax as the core thresholds move. A robust
                        two-state model barely changes; a large drift means the
                        cores are cutting through a state or missing visits.
    dwell_cv_null       how often a Markov two-state process with the fitted rates
                        and the same T gives a dwell CV at least as large as observed.

For a pure telegraph u (u = committed state, standardised) the lag-1-frame chord
equals tau_relax from two_state_model exactly:
    C(1) = 1 - (#jumps / N) / (2 p q) = 1 - n01/N0 - n10/N1 = lambda_2.
So a lag-1 SRV t_1 shorter than tau_relax measures how rough psi_1 is at one
frame (recrossings, within-state motion), not a different slow process.
"""

import numpy as np

from .twostate import core_entry_indices, two_state_model_from_entries


def chord_scan(u, lagframes_list, frametime):
    """Normalised ACF and chord t_u(tau) at selected lags (direct sums, global mean).

    One O(N) dot product per lag, so cheap for ~20 lags on 15 M frames.

    Returns (lags_time, C, chord) arrays.
    """
    du = np.asarray(u, dtype=float)
    du = du - du.mean()
    var = du @ du / du.size
    lags_time, C = [], []
    for lf in lagframes_list:
        lags_time.append(lf * frametime)
        C.append(du[:-lf] @ du[lf:] / (du.size - lf) / var)
    lags_time, C = np.array(lags_time), np.array(C)
    with np.errstate(divide='ignore', invalid='ignore'):
        chord = np.where((C > 0) & (C < 1), -lags_time / np.log(C), np.nan)
    return lags_time, C, chord


def core_sensitivity(cv, lows, highs, frametime):
    """Two-state numbers for every (low, high) pair: core0 = cv < low, core1 = cv > high.

    Returns a list of dicts with low, high, n01, n10, q (= p1), tau_relax,
    mean dwells and the CV of each state's dwells.
    """
    cv = np.asarray(cv)
    rows = []
    for lo in lows:
        for hi in highs:
            if hi <= lo:
                continue
            e1, e0 = core_entry_indices(cv < lo, cv > hi)
            m = two_state_model_from_entries(e1, e0, cv.shape[0], frametime)
            d0, d1 = m['dwells']
            rows.append(dict(low=lo, high=hi, n01=m['n01'], n10=m['n10'],
                             q=m['pi'][1], tau_relax=m['tau_relax'],
                             mean_d0=d0.mean(), mean_d1=d1.mean(),
                             cv_d0=d0.std(ddof=1) / d0.mean() if d0.size > 1 else np.nan,
                             cv_d1=d1.std(ddof=1) / d1.mean() if d1.size > 1 else np.nan))
    return rows


def print_core_sensitivity(rows, unit='ns'):
    print(f"{'low':>6} {'high':>6} {'n01':>5} {'n10':>5} {'q':>8} "
          f"{'tau_relax':>10} {'<t0>':>10} {'<t1>':>9} {'CV0':>5} {'CV1':>5}   ({unit})")
    for r in rows:
        print(f"{r['low']:6.2f} {r['high']:6.2f} {r['n01']:5d} {r['n10']:5d} {r['q']:8.4f} "
              f"{r['tau_relax']:10.2f} {r['mean_d0']:10.1f} {r['mean_d1']:9.2f} "
              f"{r['cv_d0']:5.2f} {r['cv_d1']:5.2f}")


def dwell_cv_null(k01, k10, total_time, observed_cv, state=1, n_rep=5000, seed=0):
    """P(CV of complete dwells in `state` >= observed_cv) for a Markov two-state process.

    Simulated in continuous time (exponential dwells, stationary start), so it is
    fast for any frametime. An exponential has CV = 1, but with ~10 dwells the
    sample CV scatters widely; this gives the right reference.

    Returns dict: p_value, median, ci (16-84%), n_dwells (median count).
    """
    rng = np.random.default_rng(seed)
    rate = (k01, k10)
    p0 = k10 / (k01 + k10)
    cvs, counts = [], []
    for _ in range(n_rep):
        s = 0 if rng.random() < p0 else 1
        t, first, d = 0.0, True, []
        while True:
            dw = rng.exponential(1.0 / rate[s])
            if t + dw >= total_time:
                break
            if not first and s == state:
                d.append(dw)
            first = False
            t += dw
            s = 1 - s
        if len(d) >= 3:
            d = np.array(d)
            cvs.append(d.std(ddof=1) / d.mean())
            counts.append(d.size)
    cvs = np.array(cvs)
    return dict(p_value=(cvs >= observed_cv).mean(), median=np.median(cvs),
                ci=np.percentile(cvs, [16, 84]), n_dwells=np.median(counts))
