"""Detect whether minimum-image wrapping corrupted a heavy-atom distance feature set.

The features written by ``SimulationData.save_feature_data`` are the pairwise
distances between the N heavy atoms, in ``itertools.combinations`` order, so a
frame's 45 numbers reconstruct the full 10x10 distance matrix.

A true set of pairwise distances among N points in R^3 is *embeddable*: the
double-centred squared-distance (Gram) matrix

    B = -1/2 J D^2 J,        J = I - 11^T / N

is positive semi-definite with rank exactly 3.  Minimum-image convention
shortens some pairs and not others, which is not a rigid motion, so the folded
distance matrix is no longer realisable in R^3 -- extra eigenvalues appear and
the smallest turns significantly negative.  (Embeddability implies every
triangle inequality, so this subsumes the weaker triangle test.)

The test therefore detects whether wrapping *changed any numbers*, which is the
actionable question: if the box was large enough that nothing folded, the
features are correct regardless of the ``periodic`` flag.
"""

import numpy as np


def _sample_blocks(features, n_sample, n_blocks):
    """Read ~n_sample frames as n_blocks contiguous runs spread over the file.

    Contiguous runs matter when *features* is a memmap: a strided fancy-index
    over millions of rows turns into millions of scattered reads, whereas this
    is n_blocks sequential ones.
    """
    n_frames = features.shape[0]
    n_sample = int(min(n_sample, n_frames))
    n_blocks = int(max(1, min(n_blocks, n_sample)))
    block = max(1, n_sample // n_blocks)
    starts = np.unique(np.linspace(0, n_frames - block, n_blocks).astype(int))
    return np.concatenate(
        [np.asarray(features[s:s + block]) for s in starts], axis=0
    )


def _embedding_residuals(dist_rows):
    """Per-frame embeddability diagnostics for rows of condensed distances.

    Parameters
    ----------
    dist_rows : (m, n_pairs) array

    Returns
    -------
    resid : (m,) float
        Eigenvalue mass outside the leading 3, relative to the leading 3.
        ~1e-7 for clean float32 data; orders of magnitude larger if folded.
    negfrac : (m,) float
        Most-negative eigenvalue relative to the largest.  ~0 when clean;
        a distance matrix that is not realisable in R^3 goes negative.
    """
    X = np.asarray(dist_rows, dtype=np.float64)
    if X.ndim == 1:
        X = X[None, :]
    m, n_pairs = X.shape

    n_atoms = int(round((1.0 + np.sqrt(1.0 + 8.0 * n_pairs)) / 2.0))
    if n_atoms * (n_atoms - 1) // 2 != n_pairs:
        raise ValueError(
            f"{n_pairs} columns is not N(N-1)/2 for integer N; "
            "these are not complete pairwise distances"
        )

    iu = np.triu_indices(n_atoms, k=1)
    D = np.zeros((m, n_atoms, n_atoms))
    D[:, iu[0], iu[1]] = X
    D += D.transpose(0, 2, 1)

    D2 = D * D
    B = -0.5 * (
        D2
        - D2.mean(axis=2, keepdims=True)
        - D2.mean(axis=1, keepdims=True)
        + D2.mean(axis=(1, 2), keepdims=True)
    )

    w = np.linalg.eigvalsh(B)          # ascending, shape (m, n_atoms)
    top3 = w[:, -3:].sum(axis=1)
    resid = np.abs(w[:, :-3]).sum(axis=1) / top3
    negfrac = -w[:, 0] / w[:, -1]
    return resid, negfrac, n_atoms


def check_periodic_wrapping(
    features, reference=None, n_sample=20000, n_blocks=20, tol=1e-4, verbose=True
):
    """Test whether minimum-image wrapping corrupted *features*.

    Parameters
    ----------
    features : (n_frames, n_pairs) array or memmap
        e.g. ``sd.features``.
    reference : (n_pairs,) array, optional
        Distances for one frame known to be unwrapped -- pass
        ``reference_from_pdb(pdbfile)`` to get a positive control that also
        validates the column-ordering assumption.
    n_sample, n_blocks : int
        How many frames to test, and over how many contiguous runs.
    tol : float
        Residual above which a frame is called non-embeddable.  Clean float32
        data sits near 1e-7, so 1e-4 is generous.

    Returns
    -------
    dict with keys ``clean``, ``n_atoms``, ``n_tested``, ``frac_bad``,
    ``resid_median``, ``resid_max``, ``negfrac_max``, ``d_min``, ``d_max``.
    """
    if reference is not None:
        r_ref, neg_ref, _ = _embedding_residuals(reference)
        if r_ref[0] > tol:
            raise ValueError(
                f"Reference frame itself fails the test (residual {r_ref[0]:.2e}). "
                "The column ordering assumed here does not match how the "
                "features were written -- fix that before trusting any result."
            )

    X = _sample_blocks(features, n_sample, n_blocks)
    resid, negfrac, n_atoms = _embedding_residuals(X)

    frac_bad = float((resid > tol).mean())
    out = {
        "clean": frac_bad == 0.0,
        "n_atoms": n_atoms,
        "n_tested": int(X.shape[0]),
        "frac_bad": frac_bad,
        "resid_median": float(np.median(resid)),
        "resid_max": float(resid.max()),
        "negfrac_max": float(negfrac.max()),
        "d_min": float(X.min()),
        "d_max": float(X.max()),
    }

    if verbose:
        print(f"heavy atoms inferred : {out['n_atoms']}  "
              f"({X.shape[1]} pair distances)")
        print(f"frames tested        : {out['n_tested']}")
        print(f"distance range       : {out['d_min']:.4f} .. {out['d_max']:.4f}")
        if reference is not None:
            print(f"reference residual   : {r_ref[0]:.3e}   (positive control)")
        print(f"residual  median/max : {out['resid_median']:.3e} / "
              f"{out['resid_max']:.3e}   (tol {tol:.0e})")
        print(f"most-negative eigval : {out['negfrac_max']:.3e} of largest")
        print(f"frames failing       : {frac_bad:.4%}")
        print()
        if out["clean"]:
            print("CLEAN -- every tested frame embeds in R^3.  No pair was folded,")
            print("so periodic=True changed nothing and these features are usable.")
        else:
            print("CORRUPTED -- some frames do not embed in R^3.  At least one pair")
            print("was minimum-imaged.  Recompute the features with periodic=False.")
    return out


def reference_from_pdb(pdbfile, periodic=False):
    """Heavy-atom pair distances for the single frame in *pdbfile*.

    Loads only the topology file (a couple of kB), not the trajectory.  Use the
    result as ``reference=`` to confirm the column ordering before trusting a
    verdict.
    """
    import itertools
    import mdtraj as md

    traj = md.load(str(pdbfile))
    heavy = traj.topology.select("resname != HOH && type != H")
    pairs = np.array(list(itertools.combinations(heavy, 2)))
    return md.compute_distances(traj, pairs, periodic=periodic)[0]
