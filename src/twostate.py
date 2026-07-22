import numpy as np


# =====================================================================
# Two-state MSM built directly from core-entry indices.
#
# The committed state is piecewise constant and changes only at the
# entry indices, so every core quantity is reconstructed from the ~2N
# change points -- O(transitions), never O(frames). This supersedes the
# {1..K, 0-for-buffer} labelled-trajectory machinery for the 2-state case
# (trajectory_from_psi / replace_zero_states / first_passage_times_twostate).
# =====================================================================


def core_entry_indices(in_core0, in_core1):
    """Frame indices of genuine core-to-core entries for a two-state system.

    A buffer frame inherits the last core visited (forward-fill). The initial
    buffer -> first-core assignment is NOT a transition and is excluded, so
    both arrays contain only genuine cross-core commitments.

    Parameters
    ----------
    in_core0, in_core1 : array(bool)
        Mutually exclusive core masks, e.g. in_core0 = cv < 0, in_core1 = cv > 6.

    Returns
    -------
    enter1_from0 : array(int)
        Frames where the system first enters core 1 having last been in core 0.
    enter0_from1 : array(int)
        Frames where the system first enters core 0 having last been in core 1.
    """
    in_core0 = np.asarray(in_core0, dtype=bool)
    in_core1 = np.asarray(in_core1, dtype=bool)

    committed = np.full(in_core0.shape, -1, dtype=np.int8)
    committed[in_core0] = 0
    committed[in_core1] = 1

    valid = committed != -1
    idx = np.where(valid, np.arange(committed.size), 0)
    np.maximum.accumulate(idx, out=idx)
    committed = committed[idx]                       # forward-filled; -1 only leading

    changes = np.flatnonzero(np.diff(committed) != 0) + 1
    # Drop the lone buffer(-1) -> first-core change: an initial assignment,
    # not a transition from the other core.
    changes = changes[committed[changes - 1] != -1]

    enter1_from0 = changes[committed[changes] == 1]
    enter0_from1 = changes[committed[changes] == 0]
    return enter1_from0, enter0_from1


def two_state_model_from_entries(enter1_from0, enter0_from1, n_frames, frametime):
    """Full two-state MSM reconstructed analytically from the entry indices.

    Parameters
    ----------
    enter1_from0, enter0_from1 : array(int)
        Output of core_entry_indices (genuine, alternating entries).
    n_frames : int
        Total frames in the trajectory (e.g. cv.size). Needed only to close the
        right-censored final segment for the populations.
    frametime : float
        Time between frames; sets the units of every returned time.

    Returns
    -------
    dict:
        dwells    : (dwells0, dwells1)  complete dwell times per state
        counts    : 2x2 transition counts at lag = 1 frame
        T         : 2x2 transition matrix at lag = 1 frame
        pi        : (p0, p1) stationary populations (time fractions)
        eigvals   : (1.0, lambda2)
        t2        : slow relaxation timescale from lambda2
        tau_relax : 1 / (k01 + k10)   (== t2 up to discretisation)
        rates     : (k01, k10)
        mfpt      : (mfpt_0to1, mfpt_1to0)  == mean dwells for a 2-state system
        n01, n10  : transition counts
    """
    e1 = np.sort(np.asarray(enter1_from0, dtype=np.int64))
    e0 = np.sort(np.asarray(enter0_from1, dtype=np.int64))

    # Merge into one alternating list of change points, labelled by entered state.
    c = np.concatenate([e1, e0])
    lab = np.concatenate([np.ones(e1.size, np.int8), np.zeros(e0.size, np.int8)])
    order = np.argsort(c, kind='stable')
    c, lab = c[order], lab[order]

    if c.size < 2:
        raise ValueError('need at least two entries to define a dwell')
    if np.any(np.diff(lab) == 0):
        raise ValueError('entries do not alternate; check cores / '
                         'that these came from core_entry_indices')

    # Complete dwells: segment k spans [c[k], c[k+1]) in state lab[k].
    inner_dur = np.diff(c)
    inner_state = lab[:-1]
    dwells0 = inner_dur[inner_state == 0] * frametime
    dwells1 = inner_dur[inner_state == 1] * frametime

    # Total frames per state, including the two censored end segments:
    #   left  [0, c[0])          -> state 1 - lab[0]  (the first committed state)
    #   right [c[-1], n_frames)  -> state lab[-1]
    N = np.zeros(2, dtype=np.int64)
    np.add.at(N, inner_state, inner_dur)
    N[1 - lab[0]] += c[0]
    N[lab[-1]] += n_frames - c[-1]
    N0, N1 = int(N[0]), int(N[1])

    n01, n10 = e1.size, e0.size                      # 0->1, 1->0 counts

    # Lag-1 transition matrix. Off-diagonal = exits / frames-in-state; the final
    # frame having no successor is an O(1) edge effect, ignored.
    T = np.array([[1 - n01 / N0,      n01 / N0],
                  [     n10 / N1, 1 - n10 / N1]])
    counts = np.array([[N0 - n01, n01],
                       [n10,      N1 - n10]], dtype=np.int64)

    pi = np.array([N0, N1], dtype=float) / (N0 + N1)
    k01, k10 = n01 / (N0 * frametime), n10 / (N1 * frametime)
    tau_relax = 1.0 / (k01 + k10)

    lam2 = 1.0 - n01 / N0 - n10 / N1
    t2 = -frametime / np.log(lam2) if 0.0 < lam2 < 1.0 else np.inf

    mfpt = (1.0 / k01, 1.0 / k10)                    # 2-state MFPT == mean dwell

    return dict(dwells=(dwells0, dwells1), counts=counts, T=T, pi=pi,
                eigvals=(1.0, lam2), t2=t2, tau_relax=tau_relax,
                rates=(k01, k10), mfpt=mfpt, n01=n01, n10=n10)


def two_state_model(cv, core0=lambda x: x < 0, core1=lambda x: x > 6,
                    frametime=1e-3):
    """Convenience: CV + core predicates -> entries -> full analytic model."""
    e1, e0 = core_entry_indices(core0(cv), core1(cv))
    return two_state_model_from_entries(e1, e0, cv.shape[0], frametime)


def bootstrap_tau_relax(dwells0, dwells1, n_boot=2000, seed=0):
    """Nonparametric error bar on tau_relax by resampling the dwell times.

    tau_relax = 1 / (k01 + k10), k = 1 / mean_dwell. The spread is the
    ~1/sqrt(N) counting error that dominates with only a handful of crossings.

    Returns dict: mean, std, ci (16-84th percentile), samples.
    """
    rng = np.random.default_rng(seed)
    d0, d1 = np.asarray(dwells0, float), np.asarray(dwells1, float)
    taus = np.empty(n_boot)
    for b in range(n_boot):
        s0 = rng.choice(d0, d0.size, replace=True)
        s1 = rng.choice(d1, d1.size, replace=True)
        taus[b] = 1.0 / (1.0 / s0.mean() + 1.0 / s1.mean())
    return dict(mean=taus.mean(), std=taus.std(),
                ci=np.percentile(taus, [16, 84]), samples=taus)


# ---------------------------------------------------------------------
# Markovianity / implied-timescale test. This is the one thing the entry
# indices alone can't give: it must count transitions from the actual
# trajectory at each lag (assuming a generator would make it flat by
# construction and test nothing). Rebuild the committed path from the
# change points, then count at each lag.
# ---------------------------------------------------------------------

def committed_trajectory_from_entries(enter1_from0, enter0_from1, n_frames):
    """Rebuild the committed {0, 1} trajectory from the change points."""
    e1 = np.asarray(enter1_from0, np.int64)
    e0 = np.asarray(enter0_from1, np.int64)
    c = np.concatenate([e1, e0])
    lab = np.concatenate([np.ones(e1.size, np.int8), np.zeros(e0.size, np.int8)])
    order = np.argsort(c, kind='stable')
    c, lab = c[order], lab[order]

    traj = np.empty(n_frames, np.int8)
    traj[:c[0]] = 1 - lab[0]                          # first committed state
    for k in range(c.size - 1):
        traj[c[k]:c[k + 1]] = lab[k]
    traj[c[-1]:] = lab[-1]
    return traj


def implied_timescale_scan(committed, lagframes_list, frametime):
    """Slow implied timescale t2(tau) vs lag, counting from the committed path.

    Flat t2(tau) supports the two-state Markov model; watch for it degrading as
    the lag approaches the minority-state lifetime, where short visits stop
    being resolved.

    Returns (lags_time, t2), both arrays.
    """
    committed = np.asarray(committed)
    lags_time, t2 = [], []
    for lf in lagframes_list:
        a, b = committed[:-lf], committed[lf:]
        C = np.zeros((2, 2))
        np.add.at(C, (a, b), 1.0)
        T = C / C.sum(1, keepdims=True)
        lam2 = np.sort(np.linalg.eigvals(T).real)[::-1][1]
        lags_time.append(lf * frametime)
        t2.append(-lf * frametime / np.log(lam2) if 0 < lam2 < 1 else np.inf)
    return np.array(lags_time), np.array(t2)
