import numpy as np
from pathlib import Path


def working_dir(wd):
    """Return a Path object for the working directory specified by wd.
    If it does not exist, create it.
    """
    if not isinstance(wd, (str, Path)):
        raise TypeError(f"wd must be Path or str. Found {type(wd)}")
    if isinstance(wd, str):
        wd = Path(wd)
    wd.mkdir(parents=True, exist_ok=True)
    return wd


def numpy_ndarray(wd, label, arr=None):
    """Convenience function for saving/loading numpy arrays from .npy files.

    Usage:
        x = numpy_ndarray(sd, 'x')
            Load 'x' from 'x.npy' and return it.
        x = numpy_ndarray(sd, 'x', x)
            Save 'x' to 'x.npy' and return it.
    """
    file = working_dir(wd) / f"{label}.npy"

    if arr is None:
        if not file.exists():
            raise FileNotFoundError(f"File {file} does not exist")
        return np.load(file)

    if isinstance(arr, (list, tuple, np.ndarray, np.generic)):
        arr = np.array(arr)
        np.save(file, arr)
        return arr

    raise TypeError(
        f"arr must be None, array-like or numpy scalar. Found {type(arr)}"
    )


def data_class(wd, label, obj=None):
    """Load or save a DataClass object and return it.

    Usage:
        data_class(wd, 'x')
            Load DataClass from 'x.npz' and return it.
        data_class(wd, 'x', x)
            Save DataClass x in 'x.npz' and return it.

    Parameters
    ----------
    wd : str or Path
        Determines the working_dir for file
    label : str
        File name = 'label.npz'
    obj : None, DataClass or dict[str: array]
        If None, load DataClass from file.
        If dict, convert to DataClass, save and return it.
        If DataClass, save and return it.

    Returns
    -------
    DataClass
    """
    file = working_dir(wd) / f"{label}.npz"

    if obj is None:
        if not file.exists():
            raise FileNotFoundError(f"File {file} does not exist")
        return DataClass.from_npz(file)

    if isinstance(obj, dict):
        obj = DataClass.from_dict(obj)
        obj.savez(file)
        return obj

    if isinstance(obj, DataClass):
        obj.savez(file)
        return obj

    raise TypeError(f"obj must be None, a dict or DataClass. Found {type(obj)}")


class DataClass:
    """A simple object to replace a dict.
    Intended to store floats, arrays, and None.
    """

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            k, v = self._enforce_type(k, v)
            setattr(self, k, v)

    def _enforce_type(self, key, val):
        if isinstance(val, (int, np.integer)):
            val = float(val)
        elif isinstance(val, np.floating):
            val = float(val)

        if not (val is None or isinstance(val, (float, np.ndarray))):
            raise TypeError(
                f"type({key}) = {type(val)}. '{self.__class__.__name__}' supports None, float or ndarray."
            )
        return key, val

    @classmethod
    def from_dict(cls, d):
        """Create from a dictionary of attribute names and values."""
        return cls(**d)

    @classmethod
    def from_strenum(cls, e, d=None):
        """Attribute names from StrEnum e, initially set to None.
        Optional values can be supplied from dict d.
        """
        dc = cls.from_dict({k: None for k in e.__members__.keys()})
        if d is not None:
            for name, val in d.items():
                if not hasattr(dc, name):
                    raise AttributeError(
                        f"Trying to assign attribute {name} from dict, not found in StrEnum {e}"
                    )
                setattr(dc, name, val)
        return dc

    @classmethod
    def from_npz(cls, file):
        """Load attributes from a .npz archive."""
        d = {}
        with np.load(file, allow_pickle=False) as z:
            for key in z.files:
                val = z[key]

                if isinstance(val, np.ndarray) and val.shape == ():
                    try:
                        val = val.item()
                    except Exception:
                        val = float(val)

                d[key] = val
        return cls.from_dict(d)

    def savez(self, file):
        np.savez(file, **self.__dict__)

    @property
    def attrs(self):
        """Tuple of all attribute names."""
        return tuple(self.__dict__.keys())

    def __add__(self, other):
        """Merge two DataClass objects.

        Shared keys:
            * one has attr=None: keep the non-None value
            * both are None: keep None
            * neither is None: raise AttributeError
        """
        if other is None:
            return DataClass(**self.__dict__)

        if not isinstance(other, DataClass):
            return NotImplemented

        ds = dict(self.__dict__)
        do = dict(other.__dict__)

        for key in ds.keys() & do.keys():
            if ds[key] is None and do[key] is None:
                continue
            elif ds[key] is None:
                ds.pop(key)
            elif do[key] is None:
                do.pop(key)
            else:
                raise AttributeError(
                    f"adding two DataClasses with shared non-None attr {key}"
                )

        return DataClass(**(ds | do))

    def __radd__(self, other):
        if other is None or other == 0:
            return DataClass(**self.__dict__)
        if isinstance(other, DataClass):
            return other.__add__(self)
        return NotImplemented

    def __eq__(self, other):
        if not isinstance(other, DataClass):
            return NotImplemented

        if self.__dict__.keys() != other.__dict__.keys():
            return False

        for k in self.__dict__:
            a = getattr(self, k)
            b = getattr(other, k)

            if a is b:
                continue

            if a is None or b is None:
                if a is not None or b is not None:
                    return False
                continue

            if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
                if not np.array_equal(a, b, equal_nan=True):
                    return False
                continue

            try:
                if np.isnan(a) and np.isnan(b):
                    continue
            except TypeError:
                pass

            if a != b:
                return False

        return True

    def __repr__(self):
        parts = []
        for k, v in self.__dict__.items():
            if v is None:
                parts.append(f"{k}: None")
            elif isinstance(v, np.ndarray):
                parts.append(f"{k}: ndarray{v.shape}")
            elif isinstance(v, (float, np.floating)):
                parts.append(f"{k}: {float(v):.2f}")
            else:
                parts.append(f"{k}: {type(v).__name__}")
        return ", ".join(parts)
