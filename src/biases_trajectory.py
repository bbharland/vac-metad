"""
biases_trajectory
=================

Compute the Metadynamics bias potential over a trajectory, using the deposited Gaussians as the data.  The bias V(s_k) is evaluated right before the new Gaussian is laid down.

        biases[i] = V_{i-1}(s_i) = sum_{j<i} G_j(s_i)

Can be computed three ways with matching numbers:

    serial        20:24     36 ns simulation, 300,000 kernels   (float64)
    threadpool     4:52     4.2x speedup with 6 workers         (float32)
    cupy           0:37     33.1 speedup with 512 blocks        (float32)

Imports are loaded lazily:

import cupy as cp
from threadpoolctl import threadpool_limits
from concurrent.futures import ThreadPoolExecutor
"""


import numpy as np

try:  # progress bar is optional
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover

    def tqdm(iterable, **kwargs):
        return iterable


from src.kernel import gaussian2d


def biases_trajectory(
    gaussians, dispatch="serial", num_frames=None, use_tqdm=True, workers=6, block=512
):
    """Compute the Metadynamics bias potential over a trajectory of Gaussians.

        biases[i] = V_{i-1}(s_i) = sum_{j<i} G_j(s_i)

    is the bias experienced as the system arrived at s_i but before the i-th
    Gaussian was deposited.  Dispatches to one of three equivalent backends.

    Parameters
    ----------
    gaussians : gaussians.Gaussians
        The set of Gaussians produced over the metadynamics simulation.
        Iterating yields (height, center, width); ``gaussians.centers`` gives the
        (N, 2) array of CV frames.
    dispatch : {"serial", "threadpool", "cupy"}
        Which backend to use:
            "serial"      numpy, single core (reference).
            "threadpool"  numpy across ``workers`` threads. ~1.5x on a 6-core
                          laptop -- bandwidth/thermal bound, not a big win.
            "cupy"        GPU, tiled. ~33x on the reference machine.

    Returns
    -------
    biases : ndarray, shape (num_frames,)
        float64. (cupy path computes the kernel in float32, accumulates and
        returns float64; see module Notes on precision.)

    Raises
    ------
    ValueError
        If ``dispatch`` is not one of the three supported strings.
    """
    match dispatch:
        case "serial":
            return biases_trajectory_serial(
                gaussians, num_frames=num_frames, use_tqdm=use_tqdm
            )
        case "threadpool":
            # Pin the math backend to one thread per worker: otherwise each of
            # the `workers` threads can spawn its own BLAS thread pool, giving
            # workers x cores threads on `cores` hardware -- everything pegs and
            # nothing speeds up.
            from threadpoolctl import threadpool_limits

            with threadpool_limits(limits=1):
                return biases_trajectory_threadpool(gaussians, workers=workers)
        case "cupy":
            return biases_trajectory_cupy(gaussians, block=block)
        case _:
            raise ValueError(
                f"{dispatch=} not implemented; "
                "choose 'serial', 'threadpool', or 'cupy'."
            )


def biases_trajectory_serial(gaussians, num_frames=None, use_tqdm=True):
    """Serial numpy.  If num_frames is not None, only compute that many frames found in 'gaussians'.  use_tqdm=True gives tqdm progress bar.
    """
    if num_frames is None:
        num_frames = len(gaussians)

    enum_params = enumerate(gaussians[: num_frames - 1])
    if use_tqdm:
        enum_params = tqdm(enum_params, total=num_frames - 1)

    cvs = gaussians.centers
    biases = np.zeros(num_frames)

    for i, (h, c, w) in enum_params:
        frames = cvs[i + 1 : num_frames]
        biases[i + 1 :] += gaussian2d(frames, h, c, w)

    return biases


def biases_trajectory_threadpool(gaussians, workers=6):
    """ """
    from concurrent.futures import ThreadPoolExecutor

    num_frames = len(gaussians)
    num_kernels = num_frames - 1
    cvs = gaussians.centers
    params = gaussians[:num_kernels]  # materialize (h, c, w) once

    def partial(worker):
        out = np.zeros(num_frames)
        for i in range(worker, num_kernels, workers):  # round-robin -> balanced load
            h, c, w = params[i]
            out[i + 1 :] += gaussian2d(cvs[i + 1 :], h, c, w, dtype=np.float32)
        return out

    if workers <= 1:
        return partial(0)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        return np.sum(list(ex.map(partial, range(workers))), axis=0)


def biases_trajectory_cupy(gaussians, block=512):
    """GPU lower-triangular all-pairs Gaussian sum.

    Tiles the N x N triangular problem over blocks of ``block`` source
    Gaussians. For each block, every source is evaluated against all target
    frames k >= i0 at once; the diagonal block is masked so only k > i
    contributes, and the column sums accumulate into ``biases``. The kernel is
    evaluated in float32 (speed + half the memory) but the triangular sum is
    accumulated in float64 to avoid precision drift over up to ~N terms.

    On the reference machine (~300k Gaussians) this ran in ~37 s vs ~1224 s
    serial (~33x), matching the serial result to np.allclose(rtol=1e-4).

    Parameters
    ----------
    block : int, optional
        Source-block size. Memory/speed knob: the early tiles are ``block x N``,
        so this caps peak VRAM. Lower it on OutOfMemoryError; raise it (1024,
        2048, 4096) for fewer kernel launches if you have headroom. Default 512.

    Notes
    -----
    Requires cupy (imported lazily here so the rest of the module stays usable
    without a GPU). For an honest timing, synchronize the device before and
    after the call -- cupy queues work asynchronously.
    """
    import cupy as cp

    N = len(gaussians)

    # --- pull aligned (height, center, width) arrays, then push to the GPU ---
    C = np.asarray(gaussians.centers, dtype=np.float32)[:N]  # (N, 2)
    if hasattr(gaussians, "heights") and hasattr(gaussians, "widths"):
        H = np.asarray(gaussians.heights, dtype=np.float32)[:N]  # (N,)
        W = np.asarray(gaussians.widths, dtype=np.float32)[:N]  # (N, 2)
    else:  # robust fallback: pull scalars once
        H = np.empty(N, np.float32)
        W = np.empty((N, 2), np.float32)
        for i in range(N):
            h, c, w = gaussians[i]
            H[i] = h
            W[i] = np.asarray(w).reshape(2)

    Cx = cp.asarray(C[:, 0])
    Cy = cp.asarray(C[:, 1])
    Wx = cp.asarray(W[:, 0])
    Wy = cp.asarray(W[:, 1])
    Hg = cp.asarray(H)
    biases = cp.zeros(N, dtype=cp.float64)  # accumulate in f64

    for i0 in range(0, N - 1, block):
        i1 = min(i0 + block, N - 1)  # sources i0..i1-1
        B = i1 - i0
        # targets: global k in [i0, N); column c -> global k = i0 + c
        dx = (Cx[i0:][None, :] - Cx[i0:i1, None]) / Wx[i0:i1, None]  # (B, N-i0)
        dy = (Cy[i0:][None, :] - Cy[i0:i1, None]) / Wy[i0:i1, None]
        G = Hg[i0:i1, None] * cp.exp(-0.5 * (dx * dx + dy * dy))  # (B, N-i0)
        G[:, :B] = cp.triu(G[:, :B], k=1)  # keep k > i in the diagonal block
        biases[i0:] += G.sum(axis=0, dtype=cp.float64)

    return cp.asnumpy(biases)
