import numpy as np
import torch
import openmm as mm
import openmm.unit as unit
import mdtraj as md
from dataclasses import is_dataclass
import pickle
from pathlib import PosixPath
import sys
import time

# --------------------------------------------------------------------------- #
#  Torch Utility/Convenience Functions
# --------------------------------------------------------------------------- #


def to_torch(a, device=None):
    """Send array to torch.Tensor.  If device=None, it will go to cpu (always?)  The intention is for a single utility function to handle this everywhere.

    torch.tensor():
        Takes array-like (np.ndarrays, lists, ...)
        Makes a copy of the data unless they reside on same device and have
        corresponding data types.
    """
    return torch.tensor(a, dtype=torch.float32, device=device)


def get_module_device(net):
    return next(net.parameters()).device


# --------------------------------------------------------------------------- #
#  OpenMM Utility/Convenience Functions
# --------------------------------------------------------------------------- #


def print_platform(simulation):
    platform = simulation.context.getPlatform().getName()
    print(f"Simulation platform: {platform}")


def write_pdb(simulation, filename):
    positions = simulation.context.getState(getPositions=True).getPositions()
    with open(filename, "w") as f:
        mm.app.PDBFile.writeFile(simulation.topology, positions, f)


def create_system(forcefield, topology):
    return forcefield.createSystem(
        topology,
        nonbondedMethod=mm.app.PME,
        nonbondedCutoff=0.9 * unit.nanometer,
        constraints=mm.app.HBonds,
    )


def state_data_reporter(filename, report_interval):
    return mm.app.StateDataReporter(
        filename,
        report_interval,
        kineticEnergy=True,
        potentialEnergy=True,
        temperature=True,
    )


def hdf5_reporter(filename, report_interval):
    return md.reporters.HDF5Reporter(filename, report_interval)


def assign_force_groups(system):
    """Each force must be in a separate group before the simulation is created.

    Taken from http://docs.openmm.org/7.1.0/api-python/generated/simtk.openmm.openmm.Context.html#simtk.openmm.openmm.Context.getState
    """
    for i, f in enumerate(system.getForces()):
        f.setForceGroup(i)


def get_force_group_name(system, force_group_id):
    return system.getForces()[force_group_id].getName()


def check_force_group(system, force_group_id):
    assert (
        get_force_group_name(system, force_group_id) == "TorchForce"
    ), "Force group id assigned to something other than TorchForce!"


def get_energy_dict(system, simulation):
    """Return dict: energy_name => energy (in kJ/mol)

    From: https://github.com/openmm/openmm-cookbook/blob/main/notebooks/cookbook/Analyzing%20Energy%20Contributions.ipynb
    """
    energy = {}
    for i, f in enumerate(system.getForces()):
        state = simulation.context.getState(getEnergy=True, groups={i})
        energy[f.getName()] = state.getPotentialEnergy()._value
    return energy


def bias_from_module(simulation, module, device=None):
    """Return the value of the bias potential for the current state of the simulation context, in kJ/mol.  Calculate it using the force module.

    Parameters
    ----------
    simulation : openmm.app.simulation.Simulation
    module : src.force.ForceModule
    """
    if device is None:
        device = get_module_device(module.net)
    r = simulation.context.getState(getPositions=True).getPositions()
    w = module(torch.tensor(r._value).to(device=device))
    return w.cpu().item()


def bias_from_context_units(simulation, force_group_id):
    """Return the value of the bias potential for the current state of the simulation context, in units.  Extract it from the simulation force group.

    Parameters
    ----------
    simulation : openmm.app.simulation.Simulation
    force_group_id : int
    """
    groups = {force_group_id}
    state = simulation.context.getState(getEnergy=True, groups=groups)
    return state.getPotentialEnergy()


def bias_from_context(simulation, force_group_id):
    return bias_from_context_units(simulation, force_group_id)._value


def kT_in_kJ_per_mol(temperature):
    assert unit.is_quantity(temperature), "temperature must be a unit in K"
    kT = unit.MOLAR_GAS_CONSTANT_R * temperature
    return kT.in_units_of(unit.kilojoule / unit.mole)._value


# --------------------------------------------------------------------------- #
#  Misc.
# --------------------------------------------------------------------------- #


def save_pickle(filename, obj):
    with open(filename, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def pretty_print(obj, gap=4):
    """Print obj's attributes (or dataclass fields), one per line, value-aligned."""
    if is_dataclass(obj):
        items = {name: getattr(obj, name) for name in obj.__dataclass_fields__}
    else:
        items = vars(obj)

    maxlen = max(len(key) for key in items)
    for key, val in items.items():
        spaces = " " * (maxlen - len(key) + gap - 2)
        print(key, spaces, val)


def arr_str(a, precision=3, abbreviate=False, abbreviate_rows=False, oneline=False):
    """How to display arrays in __str__ methods.

    Parameters:
    a : array with shape (n,) or (m, n)

    precision : int
        Precision of array elements in string
    abbreviate : Bool
        If True, include only first and last elements
    abbreviate_rows : Bool
        In case of 2d array, like abbreviate but for rows
    """

    def row_str(a, precision, abbreviate):
        if abbreviate:
            row = f"{a[0]:.{precision}f}...{a[-1]:.{precision}f}"
        else:
            row = " ".join([f"{x:.{precision}f}" for x in a])
        return "[" + row + "]"

    endchar = "" if oneline else "\n"
    if len(a.shape) == 1:
        return row_str(a, precision, abbreviate)
    elif len(a.shape) == 2:
        if abbreviate_rows:
            return (
                "["
                + row_str(a[0], precision, abbreviate)
                + endchar
                + "..."
                + endchar
                + row_str(a[-1], precision, abbreviate)
                + "]"
            )
        else:
            return (
                "[" + endchar.join([row_str(r, precision, abbreviate) for r in a]) + "]"
            )


def sample_array_rows(a, sample_size):
    """Parameters:
    -----------
    a : array, shape (num_rows, num_dim)
        The array to be sampled.
    sample_size : int
        The number of random samples to be returned.
    """
    assert (
        len(a) >= sample_size
    ), f"ERROR: can't sample {sample_size} rows from array with shape {a.shape}"
    return a[np.random.choice(len(a), sample_size, replace=False)]


def sizeof(obj, units="kB", loud=False):
    assert units in ("kB", "MB", "GB"), f"Can't do unit {units}"
    BYTE_TO_KB = 1 / 1024

    def size_units(size_kb, units):
        if units == "kB":
            return size_kb
        elif units == "MB":
            return size_kb / 1024
        elif units == "GB":
            return size_kb / 1024**2

    total_size_kb = 0
    for key, val in vars(obj).items():
        kb = sys.getsizeof(val) * BYTE_TO_KB
        if loud:
            print(f"size of {key} = {size_units(kb, units)} {units}")
        total_size_kb += kb

    return size_units(total_size_kb, units)


def check_dataclass_field_types(obj):
    """Found on https://stackoverflow.com/questions/58992252/how-to-enforce-dataclass-fields-types
    Check that arguments passed match type hints.
    Parameters: dc_obj = instance of dataclass object
    """
    for name, field_type in obj.__annotations__.items():
        obj_attr = getattr(obj, name)
        if not isinstance(obj_attr, field_type):
            raise TypeError(
                f"The field `{name}` was assigned by `{type(obj_attr)}` instead of `{field_type}`"
            )


def range_steps_dataset(step, num_steps_data=5):
    """Return an iterator over the list of steps to be included in the dataset (in addition to step 0).  Count up to (but not including) 'step'.

    Parameters
    ---------
    step : int
        The current step you want to train s^n => p^n(s) => w_n(s)
    num_steps_data : int
        The maximum number of previous steps to be included in this data.
    """
    start = max(1, step - num_steps_data)
    return range(start, step)


def print_status_file(file, num_chars=None):
    """If 'file' exists, print when it was last modified.  Otherwise, say not found.

    Parameters
    ----------
    file : Path
    num_chars : int
        The total number of character reserved for the file name.  The date modified appears after this many characters.  If None, use 4 spaces after the file name.
    """
    assert isinstance(file, PosixPath), "'file' needs to be of type 'Path'"
    if num_chars is None:
        whitespace = " " * 4
    else:
        whitespace = " " * (num_chars - len(str(file)))

    if file.exists():
        timestamp = time.ctime(file.stat().st_mtime)
        print(f"file {file} exists.{whitespace} Modified {timestamp}")
    else:
        print(f"file {file} not found.")


def print_status_files(files):
    """For each Path object in list, 'files', print whether file exists and, if it does, when it was last modified."""
    num_chars = len(str(max(files, key=lambda file: len(str(file))))) + 4
    for file in files:
        print_status_file(file, num_chars=num_chars)


def unpack_obj(obj, obj_type="surf"):
    if obj_type == "pdf":
        return obj.x, obj.y, obj.p
    elif obj_type == "bias":
        return obj.x, obj.y, obj.v
    elif obj_type == "fes":
        return obj.x, obj.y, obj.f
    elif obj_type == "surf":
        return obj.x, obj.y, obj.z
    else:
        raise ValueError(f"Unknown obj_type: {obj_type!r}")
