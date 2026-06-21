import numpy as np
import multiprocessing as mp


def _compute_row(x_i, y, func):
    """Evaluate ``func(x_i, y_j)`` for every ``y_j`` in ``y`` (one grid row)."""
    return np.array([func(x_i, y_j) for y_j in y])


def grid_from_arrays_by_row(x, y, func, processes):
    """Build the grid one row at a time, distributing rows across workers."""
    row_args = [(x_i, y, func) for x_i in x]
    with mp.Pool(processes=processes) as pool:
        rows = pool.starmap(_compute_row, row_args)
    return np.vstack(rows)


def grid_from_arrays(x, y, func, processes=None, by_row=True):
    """
    Return ``z`` with shape ``(len(x), len(y))`` where ``z[i, j] = func(x[i], y[j])``.

    Parameters
    ----------
    x, y : array-like
        The two axes of the grid.
    func : callable
        Scalar function of two scalars. When ``processes`` is not None this must
        be picklable (i.e. a module-level function or a ``functools.partial`` of
        one -- not a lambda or a nested closure).
    processes : int or None
        If None, evaluate serially in this process.
        If int, evaluate with a multiprocessing pool of that many workers.
    by_row : bool
        Only relevant when ``processes`` is not None. If True, distribute whole
        rows to workers; if False, distribute individual points.
    """
    if processes is None:
        z = np.empty((len(x), len(y)), dtype=float)
        for i, x_ in enumerate(x):
            for j, y_ in enumerate(y):
                z[i, j] = func(x_, y_)
        return z

    if by_row:
        return grid_from_arrays_by_row(x, y, func, processes)
    else:
        xy_args = [(x_, y_) for x_ in x for y_ in y]
        with mp.Pool(processes=processes) as pool:
            vals = pool.starmap(func, xy_args)
        return np.array(vals, dtype=float).reshape(len(x), len(y))
