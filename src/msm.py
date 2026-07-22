import numpy as np

from .calcs import (
    eig_sorted,
    timescale_from_eigval
)


state_names = {1: r'$C_5$', 2: r'$C_7^{eq}$', 3: r'$\alpha_P$',
               4: r'$\alpha_R$', 5: r'$C_7^{ax}$', 6: r'$\alpha_L$'}


def core_state_psi(psi):
    """Identify state corresponding to psi(x).

    NB: The numbers are valid only for one particular solution to the eigenvalue problem!  In this case, it came from solving the unbiased 350 ns simulation.

    Parameters:
    ----------
    psi : array with shape (num_eigvals,)
        Value of the eigenfunctions for a structure, psi(x)

    Return
    ------
    state : int
        0   : Structure is not in a state core
        1-6 : Structure is in corresponding state core
    """
    state = 0 # not in a state core

    s1234 = -0.26 < psi[0] < -0.13
    s56 = 5 < psi[0] < 5.42

    if s1234:
        s12 = 0.2 < psi[1] < 0.78
        s34 = -1.57 < psi[1] < -1

        if s12:
            if 1 < psi[3] < 2.47:
                state = 1
            elif -2.008 < psi[3] < -0.7:
                state = 2
        if s34:
            if -5.6 < psi[4] < -1:
                state = 3
            elif 0.8 < psi[4] < 2.31:
                state = 4
    elif s56:
        if 17 < psi[2] < 20:
            state = 5
        elif -1.8 < psi[2] < -0.7:
            state = 6
    return state


def core_state_psi_withnans(psi):
    if np.isnan(psi[0]):
        return np.nan

    state = core_state_psi(psi)
    if state == 0:
        return np.nan
    else:
        return state


def state_psi(psi):
    s1234 = psi[0] < 2
    s56 = psi[0] > 2

    if s1234:
        s12 = psi[1] > 0
        s34 = psi[1] < 0

        if s12:
            if psi[3] > 0:
                return 1
            else:
                return 2
        elif s34:
            if psi[4] < 0:
                return 3
            else:
                return 4
    elif s56:
        if psi[2] > 5:
            return 5
        else:
            return 6


def state_psi_withnans(psi):
    if np.isnan(psi[0]):
        return np.nan
    else:
        return state_psi(psi)


def replace_zero_states(states):
    """Return new sequence of states with 0s replaced by last state core visited.
    """
    new_states = states.copy()
    state = new_states[0]
    for i, s in enumerate(states):
        if s == 0:
            new_states[i] = state
        else:
            state = new_states[i]
    return new_states


def coarse_grain_states(states, merge_states):
    """Return new coarse-grained sequence of states.

    If states contain 0s, these get mapped to 0.  Other states are mapped according to 'merge_states':
    E.g. ((1, 2, 3), (4, 5)) results in a two state model with s=1 corresponding to input states (1, 2, 3)
    """
    new_states_dict = {0: 0}
    for n, state_set in enumerate(merge_states):
        new_states_dict.update({i: n + 1 for i in state_set})
    return np.array([new_states_dict[s] for s in states])


def states_from_psi(psi, use_state_cores=True, merge_states=None):
    """From psi array:
        1. Determine which state index each psi[i] corresponds to (using defined state cores or not)
        2. Do coarse graining of states according to 'merge_states'

    Parameters
    ----------
    psi : array with shape (num_frames, num_eigfuncs)
    use_state_cores : Bool
        Decide whether to use 'state_psi' or 'core_state_psi'
    merge_states : list(list(int))
        If you want to lump states together, provide a list of list of state numbers to merge

    Return
    ------
    states : array(int) with shape (num_frames,)
        use_state_cores=False : States are labelled 1-6.
        use_state_cores=True : States are labelled 0-6 (0 = outside state core)
    """
    if use_state_cores:
        states = np.array([core_state_psi(x) for x in psi])
        states[0] = state_psi(psi[0]) # use non-core states for initial state
    else:
        states = np.array([state_psi(x) for x in psi])

    if merge_states is not None:
        return coarse_grain_states(states, merge_states)
    else:
        return states


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
    states = states_from_psi(
        psi, use_state_cores=use_state_cores, merge_states=merge_states
    )
    if use_state_cores: # fill in 0's with last core visited
        return replace_zero_states(states)
    else:
        return states



def deal_with_state_core_zeros(states):
    """Take a sequence of states where 0 indicates that the system is not in a state core and replace it with the last state core the system visited.
    """
    assert states[0] != 0, (
        "Trajectory cannot start with a zero state!"
    )
    trajectory = [states[0]]
    for state in states[1:]:
        si = trajectory[-1]
        sj = state if state != 0 else si
        trajectory.append(sj)
    return np.array(trajectory)


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
    unique = np.unique(states)
    num_states = len(unique)
    assert np.all(unique == np.arange(1, num_states + 1)), (
        f'states must be in range [1, num_states] & all states must be present'
    )
    counts = np.zeros((num_states, num_states), dtype=int)
    for si, sj in zip(states[:-lagframes], states[lagframes:]):
        counts[si - 1, sj - 1] += 1
    return counts


def equilibrium_distribution_mle(counts):
    """MLE estimate for the equilibrium distribution from transition counts.
            pi_i ~ sum_j C_ij / sum_ij C_ij
    """
    counts_sym = 0.5 * (counts + counts.T)
    return counts_sym.sum(axis=1) / counts_sym.sum()


def transition_matrix_mle(counts):
    """MLE estimate for the transition matrix from transition counts.
            T_ij ~ C_ij / sum_j C_ij
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
    vr = np.zeros(vl.shape)
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
        M_ij = MFPT(i->j) (shares units with lagtime)

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


def course_grain_state_trajectory(trajectory, lumped_states):
    """Parameters
    ----------
    trajectory : array(int) with shape (num_frames,)
        Sequence of states, each state in [1, num_states]
    lumped_states : list(list(int))
        A tuple of tuples indicating states to be lumped together
        E.g. ((1, 2, 3), (4, 5)) results in a two state model with s=1 corresponding to input states (1, 2, 3)

    Return
    ------
    new_trajectory : array(int) with shape (num_frames,)
        Coarse grained sequence of states, each state in [1, new_num_states]
    """
    new_states_dict = {}
    for n, state_set in enumerate(lumped_states):
        new_states_dict.update({i: n + 1 for i in state_set})
    return np.array([new_states_dict[s] for s in trajectory])


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
