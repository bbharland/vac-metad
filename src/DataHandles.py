import numpy as np
from pathlib import Path
import datetime
import torch

from .util import save_pickle, load_pickle


class DataHandles:
    """Base class for providing handles for computed & saved data.

    Child classes must define:

        data_filenames : list[str]
            List of filenames for all data (arrays, torch objects, or pickled objects) for which handles (assigned variables) may be provided.

        other_filenames : dict[str label: str filename]
            Labels/filenames that are not assigned, just added to self.files dict.

    Two use cases:
        1. assign_labels = None

            Create an attribute with name 'filename.stem', for each filename is in data_filenames.  Each attribute is assigned to either:

            (a) the object in its file
            (b) None, in case the file doesn't exist

        2. assign_labels = ['label1', 'label2', ...]

            Create only attributes 'label1', 'label2', ...

    Once instantiated, further new attributes can be assigned with
        save_and_assign_objects({'label': object, ...})
    """

    def __init__(self, working_dir, assign_labels=None):
        """Parameters
        ----------
        working_dir : str or Path
            Location for data files to be written
        assign_label : list(str)
            If None: assign all objects in ChildClass.data_filenames
            Else: assign only objects in list of labels
        """
        self.working_dir = Path(working_dir)
        self.working_dir.mkdir(parents=True, exist_ok=True)
        self.files = self._files_dict(self.working_dir)

        if assign_labels is None:
            labels = [Path(filename).stem for filename in type(self).data_filenames]
        else:
            labels = assign_labels

        for label in labels:
            self._assign_object(self.files[label])

    def _files_dict(self, working_dir):
        """Returns
        -------
        files : dict[str: Path]
                The dictionary containing all label: file pairs contained in
                (child).data_filenames : list [str]
                (child).other_filenames : dict [str: str]
        """
        files = {}

        for filename in type(self).data_filenames:
            file = working_dir / filename
            files[file.stem] = file

        for label, filename in type(self).other_filenames.items():
            files[label] = working_dir / filename

        return files

    def _assign_object(self, file):
        """If the file containing the object exists, load it and assign it to an attribute.  If it does not, assign the attribute to None."""
        if file.exists():
            match file.suffix:
                case ".pickle":
                    try:
                        obj = load_pickle(file)
                    except Exception as e:
                        # Specific exceptions some day?
                        #   * cPickle.UnpicklingError: no top-level class def
                        #   * AttributeError: openmm unit error
                        #   * RuntimeError: torch/CUDA device error
                        print(
                            f"Error loading {file}:\n"
                            f"   {type(e).__name__}: {e}\n"
                            f"Setting {file.stem} to None"
                        )
                        obj = None
                case ".npy":
                    obj = np.load(file)
                case ".pt":
                    # In a future PyTorch release, weights_only=True becomes the
                    # default, as it is safer (it won't allow unpickling arbitrary
                    # Python objects).
                    # TODO: migrate to saving torch objects with weights_only=True
                    # so this load no longer needs weights_only=False.
                    # TODO: remove map_location=torch.device("cpu") once the
                    # torch/cuda issue is resolved.
                    obj = torch.load(
                        file, map_location=torch.device("cpu"), weights_only=False
                    )
                case _:
                    raise TypeError(f"Don't recognize extension {file.suffix}")
        else:
            obj = None
        setattr(self, file.stem, obj)

    def assign_objects(self, labels):
        for label in labels:
            self._assign_object(self.files[label])

    def _save_and_assign_object(self, file, obj):
        """Parameters
        ----------
        file : Path object
        obj : object to be saved to location specified by 'file'
        """
        match file.suffix:
            case ".pickle":
                save_pickle(file, obj)
            case ".npy":
                np.save(file, obj)
            case ".pt":
                torch.save(obj, file)
            case _:
                raise TypeError(f"Don't recognize extension {file.suffix}")
        # self._assign_object(file) # TODO: is this better for some reason?
        setattr(self, file.stem, obj)

    def save_and_assign_objects(self, labels_objects):
        """Save and assign all objects in dictionary,
        labels_objects : dict (str) -> (obj)
        """
        for label, obj in labels_objects.items():
            self._save_and_assign_object(self.files[label], obj)

    def __str__(self):
        existing_data_fields = []
        missing_data_fields = []
        max_len = len(max(type(self).data_filenames, key=len))

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
