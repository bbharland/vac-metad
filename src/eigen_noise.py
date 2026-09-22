"""Compare SRV / VAC eigenvalues with the sampling-noise floor.

Null model
----------
A mode whose true eigenvalue is 0, and which decorrelates within one frame,
has whitened lag-tau covariance entries that are averages of N nearly
independent unit-variance products, so each entry has sd

    sigma = 1 / sqrt(N),        N = number of (x, y) frame pairs in the fit.

Because the lagged covariance is symmetrised (c1 = (c01 + c01') / 2), the block
of m such modes is a random symmetric matrix with diagonal variance sigma^2 and
off-diagonal variance sigma^2 / 2.  Its eigenvalues are what "pure noise"
looks like; the distribution of the *largest* one (simulated here) is the
threshold a real mode has to clear.  Coupling to the slow modes (lambda ~ 1)
shifts the noise eigenvalues only at O(sigma^2), so it is ignored.

A threshold lambda* on the eigenvalue becomes a threshold on the timescale via
t = tau / ln(1/lambda):

    t* = tau / ln(1/lambda*)  ~  tau / ln(sqrt(N) / z).

N enters only logarithmically: resolving faster processes needs a shorter lag,
not a longer trajectory.  Noise eigenvalues (|lambda| ~ sigma) map to
t ~ tau / ln(sqrt(N)), a value set by tau and N alone -- which is why noise
"timescales" grow in proportion to the lag while real ones do not.

Caveat: the 1/sqrt(N) null applies only to modes whose true eigenvalue is 0.
It says nothing about the error on the slow eigenvalues, which are strongly
time-correlated and need a correlated-data estimate (e.g. block resampling).
"""

import numpy as np

# Eigenvalues this small are the ~1e-24 residue of directions removed by
# 'trunc' whitening (see SRV._solve), not noise; they are excluded from tests.
_TRUNCATED = 1e-12


def _num_pairs(dataset):
    """Number of (x, y) frame pairs the SRV was fitted on."""
    if hasattr(dataset, "trajectory") and hasattr(dataset, "lagframes"):
        return len(dataset.trajectory) - dataset.lagframes
    return len(dataset)


def _null_eigvals(m, sigma, num_draws, seed):
    """Eigenvalues of num_draws random symmetrised m x m noise blocks."""
    rng = np.random.default_rng(seed)
    a = rng.normal(0.0, sigma, size=(num_draws, m, m))
    k = 0.5 * (a + np.swapaxes(a, 1, 2))
    return np.linalg.eigvalsh(k)  # (num_draws, m), ascending


def _timescales(eigvals, lagtime):
    """t_i = -tau / ln(lambda_i); nan for lambda <= 0 (same as SRV.timescales)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return -lagtime / np.log(eigvals)


def eigval_noise_report(
    sd,
    dataset,
    num_slow=1,
    num_draws=100_000,
    seed=0,
    reference_timescales=(10, 15, 20, 50, 100),
):
    """Print how a fitted SRV's eigenvalues compare with sampling noise.

    Parameters
    ----------
    sd : SimulationData
        Supplies ``eigvals`` (descending) and ``lagtime`` (tau, e.g. ps).
    dataset : TrajectoryDataset
        The dataset ``srv`` was fitted on; supplies N, the number of frame
        pairs.  (N is taken from the data, not from SimulationParameters,
        whose ``num_frames`` is only right when ``simulation_time`` is set.)
    num_slow : int
        Leading modes treated as real.  The remaining (non-truncated) modes form
        the null block tested against pure noise.  The report warns if this
        choice looks wrong in either direction.
    num_draws : int
        Monte Carlo draws for the null distribution.
    seed : int
        Random seed, so repeated reports are identical.
    reference_timescales : sequence of float or None
        True timescales (units of ``srv.lagtime``) for the SNR table; None
        skips it.
    """
    eigvals = np.asarray(sd.eigvals, dtype=float)
    tau = float(sd.lagtime)
    n = _num_pairs(dataset)
    sigma = 1.0 / np.sqrt(n)
    t = _timescales(eigvals, tau)

    idx_null = np.arange(num_slow, eigvals.size)
    truncated = np.abs(eigvals) < _TRUNCATED
    idx_null = idx_null[~truncated[idx_null]]
    lam_null = eigvals[idx_null]
    m = lam_null.size

    print("SRV eigenvalue noise report")
    print(f"  frame pairs N = {n:,}   lag tau = {tau:g} ps   "
          f"sigma = 1/sqrt(N) = {sigma:.5f}")

    if m == 0:
        print("  No modes left for the null block; lower num_slow.")
        return

    null = _null_eigvals(m, sigma, num_draws, seed)
    largest = null[:, -1]
    sumsq = np.sum(null**2, axis=1)
    q = {c: np.quantile(largest, c) for c in (0.95, 0.99, 0.999)}
    lam_star = q[0.99]

    print(f"  modes 1-{num_slow} assumed real; null block = {m} mode(s)")
    print()
    print(f"  {'mode':>4} {'lambda':>11} {'lambda/sigma':>13} "
          f"{'t (ps)':>11}   verdict")
    for i, (lam, ti) in enumerate(zip(eigvals, t)):
        if truncated[i]:
            verdict = "truncated (rank)"
        elif lam > lam_star:
            verdict = "resolved"
        elif lam <= 0:
            verdict = "noise (lambda <= 0)"
        else:
            verdict = "noise"
        t_str = f"{ti:11.1f}" if np.isfinite(ti) else f"{'nan':>11}"
        print(f"  {i + 1:>4} {lam:>11.6f} {lam / sigma:>13.2f} {t_str}   {verdict}")

    ss_obs = float(np.sum(lam_null**2))
    ss_exp = m * (m + 1) / 2 * sigma**2
    big_obs = float(lam_null.max())
    print()
    modes = ", ".join(str(i + 1) for i in idx_null)
    print(f"  Null-block tests (modes {modes}):")
    print(f"    sum lambda^2   observed {ss_obs:.3e}   expected {ss_exp:.3e} "
          f"(ratio {ss_obs / ss_exp:.2f})   p = {np.mean(sumsq >= ss_obs):.2f}")
    print(f"    largest        observed {big_obs:.5f}   null median "
          f"{np.median(largest):.5f}, 99% {lam_star:.5f}   "
          f"p = {np.mean(largest >= big_obs):.2f}")

    floors = {c: tau / np.log(1.0 / v) for c, v in q.items()}
    print()
    print(f"  Timescale detection floor t* = tau / ln(1/lambda*):   "
          f"95% {floors[0.95]:.1f}   99% {floors[0.99]:.1f}   "
          f"99.9% {floors[0.999]:.1f} ps")
    print(f"  Noise fingerprint tau / ln(sqrt(N)) = "
          f"{tau / np.log(np.sqrt(n)):.1f} ps (grows in proportion to tau)")

    # Self-checks on num_slow.
    weak = [i + 1 for i in range(min(num_slow, eigvals.size))
            if not eigvals[i] > lam_star]
    if weak:
        print(f"  [warn] mode(s) {weak} assumed real but below the 99% noise "
              f"floor; lower num_slow.")
    if big_obs > lam_star:
        print(f"  [warn] null block contains a mode above the 99% noise floor; "
              f"it may be real -- raise num_slow and rerun.")

    if reference_timescales:
        print()
        print(f"  Resolvability at tau = {tau:g} ps (SNR = sqrt(N) exp(-tau/t)):")
        print(f"    {'true t (ps)':>11} {'lambda':>11} {'SNR':>9} {'sigma_t/t':>10}")
        for tr in reference_timescales:
            lam = np.exp(-tau / tr)
            rel = sigma / (lam * abs(np.log(lam)))
            print(f"    {tr:>11g} {lam:>11.3e} {lam / sigma:>9.2f} "
                  f"{100 * rel:>9.2f}%")
