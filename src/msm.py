import numpy as np

from .math import (
    eig_sorted,
    timescale_from_eigval
)


state_names = {1: r'$C_5$', 2: r'$C_7^{eq}$', 3: r'$\alpha_P$',
               4: r'$\alpha_R$', 5: r'$C_7^{ax}$', 6: r'$\alpha_L$'}


# def replace_zero_states(states):
#     """Return new sequence of states with 0s replaced by last state core visited.
#     """
#     new_states = states.copy()
#     state = new_states[0]
#     for i, s in enumerate(states):
#         if s == 0:
#             new_states[i] = state
#         else:
#             state = new_states[i]
#     return new_states


def trajectory_from_psi(psi, use_state_cores=True, merge_states=None):
    """Parameters
    ----------
    psi : array with shape (num_frames, num_eigfuncs)
    use_state_cores : Bool
        Decide whether to use 'state_psi' or 'core_state_psi'
    merge_states : list(list(int))
        If you want to lump states together, provide a list of list of state numbers to merge

    Return
    ------
    trajectory : array with shape (num_frames,)
        The sequence of states used for analysis (0's are replaced with last state core visited.)
    """
    # REVIEW: This path coarse-grains *then* fills zeros, whereas
    # `trajectory_with_state_cores` fills zeros *then* coarse-grains. The two
    # orders happen to agree (the "last core visited" is the same state either
    # way, so its coarse label is the same), but that equivalence is non-obvious
    # and easy to break. Consolidating the two trajectory builders onto one
    # ordering would remove the need to reason about it.
    states = states_from_psi(
        psi, use_state_cores=use_state_cores, merge_states=merge_states
    )
    if use_state_cores: # fill in 0's with last core visited
        return replace_zero_states(states)
    else:
        return states



def transition_counts_matrix(states, lagframes=1):
    """Parameters
    ----------
    states : array(int) with shape (num_frames,)
        Integers indicating each state.  If using state cores, this must be dealt with first.

    lagframes : int
        How many frames separate each counted transition?

    Return
    ------
    counts : array(int) with shape (num_states, num_states)
        Counts of transitions found in 'states'
    """
    # REVIEW: `lagframes=0` silently returns an all-zero matrix, because
    # states[:-0] == states[:0] == empty. Guard it.
    assert lagframes >= 1, f'lagframes must be >= 1, got {lagframes}'

    unique = np.unique(states)
    num_states = len(unique)
    assert np.all(unique == np.arange(1, num_states + 1)), (
        f'states must be in range [1, num_states] & all states must be present'
    )

    # REVIEW: vectorised the count loop. For MD trajectories with millions of
    # frames the pure-Python loop dominates. This flattens each (from, to) pair
    # to a single index and bincounts it -- identical result, orders of
    # magnitude faster. Revert to the loop if you value the explicitness more.
    src = states[:-lagframes] - 1
    dst = states[lagframes:] - 1
    flat = src * num_states + dst
    counts = np.bincount(flat, minlength=num_states ** 2)
    return counts.reshape(num_states, num_states).astype(int)


def equilibrium_distribution_mle(counts):
    """Reversible estimate of the equilibrium distribution from transition counts.
            pi_i ~ sum_j C_sym_ij / sum_ij C_sym_ij,   C_sym = (C + C.T) / 2

    NB: This is the symmetrised-counts reversible estimator, not the maximum
    likelihood reversible estimator.  The true reversible MLE requires an
    iterative fixed-point solve under detailed balance (cf. Prinz et al. 2011,
    as implemented in deeptime/pyEMMA).  The symmetrised estimator is a
    consistent, internally-consistent reversible estimator (pi below is exactly
    stationary for the T returned by `transition_matrix_mle`), but the "mle" in
    the name is a misnomer.
    """
    counts_sym = 0.5 * (counts + counts.T)
    return counts_sym.sum(axis=1) / counts_sym.sum()


def transition_matrix_mle(counts):
    """Reversible estimate of the transition matrix from transition counts.
            T_ij ~ C_sym_ij / sum_j C_sym_ij,   C_sym = (C + C.T) / 2

    See the note in `equilibrium_distribution_mle`: this is the symmetrised
    reversible estimator, not the reversible MLE.
    """
    counts_sym = 0.5 * (counts + counts.T)
    return counts_sym / counts_sym.sum(axis=1).reshape(-1, 1)


def eig_transition_matrix(T):
    """Eigenvectors normalized per Noe:
        <phi_i|psi_j> = <phi_i|phi_j>_pi^-1 = <psi_i|psi_j>_pi = delta_ij

    Parameters
    ----------
    T : ndarray, size (n, n)
        Transition matrix

    Return
    ------
    w : ndarray, size (n,)
        Eigenvalues
    vl : ndarray, size (n, n)
        Left eigenvectors, row-wise (phi)
    vr : ndarray, size (n, n)
        Right eigenvectors, col-wise (psi)
    """
    w, vl = eig_sorted(T.T)
    vl = vl.T
    # REVIEW: was `np.zeros(vl.shape)`, which is float64. If eig_sorted returns
    # a complex dtype (np.linalg.eig does, even for a reversible T with real
    # spectrum), assigning complex slices into a float array drops the
    # imaginary part with a ComplexWarning. `zeros_like` keeps vr's dtype in
    # step with vl so real-but-complex-typed vectors survive intact. If
    # eig_sorted already casts to real, this is a no-op.
    vr = np.zeros_like(vl)
    pi = vl[0, :] / vl[0, :].sum()

    for i, phi in enumerate(vl):
        if i == 0:
            vl[i, :] = pi
        else:
            vl[i, :] = phi / np.sqrt(np.inner(phi, phi / pi))
        vr[:, i] = vl[i, :] / pi
    return w, vl, vr


def mfpt_matrix(T, pi, lagtime):
    """
    Return matrix of mean first passage times, M, where:
        M_ij = MFPT(i->j) (shares units with lagtime), and M_ii = 0

    Snell's formula
        M_ij = (Z_jj - Z_ij) / pi_j

    where
        Z = [1 - T + W]^{-1} and W_ij = pi_j

    Parameters
    ----------
    T : array with shape (num_states, num_states)
        The transition matrix
    pi : array with shape (num_states,)
        The equilibrium distribution
    lagtime : float
        The lagtime.  Whatever units will be used by M
    """
    assert T.shape[0] == T.shape[1] == len(pi), (
        f'Size mismatch: {T.shape = }, {len(pi) = }'
    )
    zero = np.zeros(T.shape)
    eye = np.eye(len(pi))

    W = zero + pi
    Z = np.linalg.inv(eye - T + W)
    Z_w = zero + np.diag(Z)
    return lagtime * (Z_w - Z) / W


def trajectory_with_state_cores(psi, merge_states=None):
    """Parameters
    ----------
    psi : array with shape (num_frames, num_eigfuncs)
    merge_states : list(list(int))
        If you want to lump states together, provide a list of list of state numbers to merge

    Return
    ------
    states : array with shape (num_frames,)
        States are labelled 1-6.  If the system is not in a state core, it is labelled 0.
    trajectory : array with shape (num_frames,)
        The sequence of states used for analysis (0's are replaced with last state core visited.)
    """
    states = np.array([core_state_psi(x) for x in psi])
    states[0] = state_psi(psi[0]) # use non-core states for initial state
    trajectory = deal_with_state_core_zeros(states)
    if merge_states is not None:
        trajectory = course_grain_state_trajectory(trajectory, merge_states)
    return states, trajectory


def msm_analysis_mfpt(trajectory, lagtime):
    # REVIEW: counts are always taken at lagframes=1 (adjacent frames) while
    # `lagtime` is threaded straight into the MFPT/timescale units. That's
    # self-consistent only if `lagtime` is the physical time between adjacent
    # frames of `trajectory`. Fine as a convention, but nothing enforces it --
    # a docstring line stating "lagtime == time per frame of trajectory" would
    # save a future footgun if anyone ever passes a strided trajectory.
    counts = transition_counts_matrix(trajectory)
    num_states = len(counts)
    pi = equilibrium_distribution_mle(counts)
    T = transition_matrix_mle(counts)
    M = mfpt_matrix(T, pi, lagtime)
    return num_states, counts, pi, T, M


def msm_analysis_eig(transition_matrix, lagtime):
    eigvals, vl, vr = eig_transition_matrix(transition_matrix)
    timescales = np.array([timescale_from_eigval(ev, lagtime) for ev in eigvals])
    return timescales, eigvals, vl, vr


def msm_analysis_state_cores(psi, lagtime, merge_states=None):
    states, trajectory = trajectory_with_state_cores(psi, merge_states)
    num_states, counts, pi, T, M = msm_analysis_mfpt(trajectory, lagtime)
    timescales, eigvals, vl, vr = msm_analysis_eig(T, lagtime)

    if num_states == 2:
        fpts12, fpts21 = first_passage_times_twostate(trajectory, lagtime)
    else:
        fpts12, fpts21 = None, None

    return {'states': states,
            'trajectory': trajectory,
            'counts': counts,
            'equilibrium_distribution': pi,
            'transition_matrix': T,
            'mfpt_matrix': M,
            'eigvals': eigvals,
            'eigvecs': vr,
            'timescales': timescales,
            'num_states': num_states,
            'fpts12': fpts12,
            'fpts21': fpts21,
           }


def first_passage_times_twostate(trajectory, lagtime):
    """Parameters
    ----------
    trajectory : array(int) with shape (num_frames,)
        Two state system, each state in [1, 2]
    lagtime : float
        Lagtime.  FPTs returned in same units as lagtime

    Return
    ------
    first_passage_times : list(list(float))
        first_passage_times[0] = list of FPTs 1 => 2
        first_passage_times[1] = list of FPTs 2 => 1

    NB: The final (trailing) dwell is intentionally not recorded -- there is no
    transition out of it, so its passage time is right-censored and dropping it
    is correct.
    """
    first_passage_times = [[], []]
    present_state = trajectory[0]
    frame_count = 1

    for state in trajectory[1:]:
        if state != present_state: # transition occurred
            first_passage_times[present_state - 1].append(frame_count * lagtime)
            present_state = state
            frame_count = 0
        frame_count += 1

    return [np.array(times) for times in first_passage_times]
