import numpy as np
import torch
from pathlib import Path
import datetime

from .util import load_pickle, save_pickle


# Each handler is a (loader, saver) pair where:
#     loader(file) -> object
#     saver(file, obj) -> None

def _load_npy(file):
    return np.load(file)


def _save_npy(file, obj):
    np.save(file, obj)


def _load_pt(file):
    # weights_only=False is needed until torch objects are saved in a
    # weights_only-safe form; map_location="cpu" sidesteps the torch/CUDA
    # device issue. Revisit both once those migrations land.
    return torch.load(file, map_location="cpu", weights_only=False)


def _save_pt(file, obj):
    torch.save(obj, file)


_HANDLERS = {
    ".pickle": (load_pickle, save_pickle),
    ".npy": (_load_npy, _save_npy),
    ".pt": (_load_pt, _save_pt),
}


class DataHandles:
    """Base class providing lazily-loaded handles for saved data.

    Subclasses must define two class attributes:

    data_filenames : list[str]
        Filenames whose stems become the data labels/attributes.  For example
        ``"features.npy"`` is reachable as ``self.features``.

    other_filenames : dict[str, str]
        ``label -> filename`` for files referenced by an explicit label rather
        than by stem (e.g. trajectory inputs).  These are *not* loaded as
        attributes; they live in ``self.files`` for path lookup.

    Accessing ``self.<label>`` reads the file on first use and caches the
    result.  If the file does not exist the attribute evaluates to ``None`` and
    is re-checked on the next access (so it picks up a file written later).

    Save new objects with :meth:`save_and_assign_objects`, which writes them and
    refreshes the cached attributes.
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        for required in ("data_filenames", "other_filenames"):
            if not hasattr(cls, required):
                raise TypeError(
                    f"{cls.__name__} must define class attribute {required!r}"
                )

    def __init__(self, working_dir):
        """Parameters
        ----------
        working_dir : str or Path
            Directory where data files are read from and written to.  Created if
            it does not exist.
        """
        self.working_dir = Path(working_dir)
        self.working_dir.mkdir(parents=True, exist_ok=True)
        self.files = self._build_files()

    def _build_files(self):
        """Map every label to its absolute path.

        Returns
        -------
        dict[str, Path]
            ``label -> path`` for both ``data_filenames`` (keyed by stem) and
            ``other_filenames`` (keyed by their explicit label).

        Raises
        ------
        ValueError
            If two filenames collapse to the same label.
        """
        files = {}

        for filename in self.data_filenames:
            label = Path(filename).stem
            if label in files:
                raise ValueError(f"Duplicate data label {label!r} from {filename!r}")
            files[label] = self.working_dir / filename

        for label, filename in self.other_filenames.items():
            if label in files:
                raise ValueError(f"Label {label!r} collides with a data filename")
            files[label] = self.working_dir / filename

        return files

    # -- lazy attribute access ---------------------------------------------

    def __getattr__(self, name):
        """Load ``self.files[name]`` on first access and cache it.

        Only invoked when normal attribute lookup fails.  Names starting with
        an underscore, and any lookup before ``self.files`` exists, raise
        ``AttributeError`` so that copy/pickle machinery behaves normally.
        """
        if name.startswith("_") or "files" not in self.__dict__:
            raise AttributeError(name)

        files = self.__dict__["files"]
        if name not in files:
            raise AttributeError(name)

        obj = self._load(files[name])
        if obj is not None:
            # Cache so future access bypasses __getattr__.  A missing file
            # (None) is left uncached so it is re-read if written later.
            setattr(self, name, obj)
        return obj

    # -- (de)serialisation --------------------------------------------------

    @staticmethod
    def _load(file):
        """Load the object at *file*, or ``None`` if it does not exist.

        A load failure (corrupt file, unpickling error, device mismatch, ...)
        is reported and treated as ``None`` rather than raising.
        """
        if not file.exists():
            return None
        try:
            loader, _ = _HANDLERS[file.suffix]
        except KeyError:
            raise TypeError(f"No handler for extension {file.suffix!r}") from None
        try:
            return loader(file)
        except Exception as exc:
            print(f"Could not load {file} ({type(exc).__name__}: {exc}); using None")
            return None

    @staticmethod
    def _save(file, obj):
        try:
            _, saver = _HANDLERS[file.suffix]
        except KeyError:
            raise TypeError(f"No handler for extension {file.suffix!r}") from None
        saver(file, obj)

    def _save_object(self, label, obj):
        """Write *obj* to the file for *label* and cache it as an attribute."""
        self._save(self.files[label], obj)
        setattr(self, label, obj)

    def save_and_assign_objects(self, labels_objects):
        """Save and cache every ``label -> object`` pair in *labels_objects*."""
        for label, obj in labels_objects.items():
            self._save_object(label, obj)

    def reload(self, *labels):
        """Drop cached values so the next access re-reads from disk.

        With no arguments, drop every cached data attribute.
        """
        for label in labels or tuple(self.files):
            self.__dict__.pop(label, None)

    def __str__(self):
        existing_data_fields = []
        missing_data_fields = []
        max_len = max((len(f) for f in type(self).data_filenames), default=0)

        for filename in type(self).data_filenames:
            file = self.working_dir / filename
            if file.exists():
                m_time = file.stat().st_mtime
                mtime = datetime.datetime.fromtimestamp(m_time)

                name_str = file.name + (max_len - len(filename)) * " "
                date_str = mtime.strftime("%d-%b-%Y")
                existing_data_fields.append(f"   {name_str}   {date_str}\n")
            else:
                missing_data_fields.append(f"   {file.name}\n")

        fields = []
        line = f"{type(self).__name__} at {self.working_dir} containing data from:\n"
        fields.append(line)
        fields.append(f'{(len(line) - 1) * "-"}\n')
        fields.extend(existing_data_fields)
        fields.append("\n")
        line = "Files that do not yet exist:\n"
        fields.append(line)
        fields.append(f'{(len(line) - 1) * "-"}\n')
        fields.extend(missing_data_fields)
        return "".join(fields)
