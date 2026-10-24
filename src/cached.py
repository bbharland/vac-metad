import functools
from pathlib import Path

import numpy as np

from .dataclass import DataClass
from .util import (
    save_pickle,
    load_pickle,
)

# ---- usage ----

# Define/wrap function

#   @cached(wd, "acfs.pickle")          # wd is dir or hasattr .working_dir
#   def compute_acfs(x, y=2):
#       ...
#       return acfs

# or, for imported function
#   compute_acfs = cached(sd, "acfs.pickle")(compute_acfs)

# Call function, calculate= added by the decorator; not passed to function

#   compute_acfs(...)                   # None (default): load if the file
#                                       #   exists, else compute and save
#   compute_acfs(..., calculate=True)   # always compute and overwrite the
#                                       #   file (use after changing inputs)
#   compute_acfs(..., calculate=False)  # always load; FileNotFoundError if
#                                       #   the file isn't there

# Caution: the cache key is only the file name. Calling again with different
# arguments silently loads the OLD result unless you pass calculate=True.

# Formats (chosen by the file extension):
#   .pickle / .pkl : any picklable object
#   .npy           : a single numpy array
#   .npz           : a DataClass (None, float or ndarray fields)


def cached(where, filename):
    """Cache a function's return value in where / filename; format set by extension.

    where: object with .working_dir (e.g. SimulationData) or a directory path.
    filename: has extension
      .pickle / .pkl : any picklable object
      .npy           : a single numpy array (ndarray)
      .npz           : a DataClass (None, float or ndarray fields)

    Wrap function to include kwarg:
      calculate=None: load if the file exists, else compute and save.
      calculate=True: always compute and save.
      calculate=False: always load.
    """
    file = _resolve_dir(where) / filename      # checked once, at decoration
    suffix = file.suffix
    if suffix not in _FORMATS:
        raise ValueError(
            f"cached: unsupported extension {suffix!r} in {filename!r}; "
            f"expected one of {sorted(_FORMATS)}")
    save, load = _FORMATS[suffix]

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, calculate=None, **kwargs):
            if calculate is None:
                calculate = not file.exists()
            if calculate:
                result = func(*args, **kwargs)
                save(file, result)
            else:
                result = load(file)
            return result
        return wrapper
    return decorator


def _resolve_dir(where):
    """Working directory from a SimulationData-like object or a path."""
    if hasattr(where, "working_dir"):
        wd = Path(where.working_dir)
    else:
        wd = Path(where)
    if not wd.is_dir():
        raise NotADirectoryError(f"cached: {wd} is not an existing directory")
    return wd


# ---- format handlers: save(file, obj), load(file) ----

def _save_npy(file, obj):
    if not isinstance(obj, np.ndarray):
        raise TypeError(f".npy caching expects an ndarray, got {type(obj).__name__}")
    np.save(file, obj)


def _load_npy(file):
    return np.load(file, allow_pickle=False)


def _save_npz(file, obj):
    if not isinstance(obj, DataClass):
        raise TypeError(f".npz caching expects a DataClass, got {type(obj).__name__}")
    obj.savez(file)


_FORMATS = {
    ".pickle": (save_pickle, load_pickle),
    ".pkl":    (save_pickle, load_pickle),
    ".npy":    (_save_npy, _load_npy),
    ".npz":    (_save_npz, DataClass.from_npz),
}
