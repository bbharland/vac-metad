import numpy as np
from pathlib import Path
import openmm.unit as unit

from .util import kT_in_kJ_per_mol
from .DefaultMixin import DefaultMixin


def param_unbiased_reference():
    """3 us simulation, June 12-17, 2026"""
    return SimulationParameters(working_dir="data/unbiased")


def param_unbiased_weighted():
    """3 us simulation, June 12-17, 2026, but using Weighted functions.

    In case the point is to get lucky while training:
    SRV unweighted:  lambda_1 = 1252 ps
    SRV weighted:    lambda_1 = 1579 ps
    """
    return SimulationParameters(working_dir="data/verify-weighted")


def param_unbiased_reference_350():
    """350 ns simulation, ca Apr. 24, 2024"""
    return SimulationParameters(working_dir="data/unbiased-350")


def param_datarich_metad():
    """22 ns WTMetaD simulation, extended on Apr. 21, 2025"""
    return SimulationParametersMetaD(working_dir="data/datarich-metad")


def param_datarich_opes():
    """22 ns OPES simulation, extended on Apr. 21, 2025"""
    return SimulationParametersOPES(working_dir="data/datarich-opes")


class SimulationParameters(DefaultMixin):
    """Hold parameters for OpenMM simulation
    * defaults defined in top of init
    * desired values are replaced by 'replace_defaults'
    * some derivative parameters are then computed from these
    """

    def __init__(self, **kwargs):
        self.working_dir = "data"
        self.pdb_file = "data/ala2_solv.pdb"
        self.simulation_time = 1 * unit.nanosecond
        self.lagtime = 1 * unit.picosecond
        self.temperature = 300 * unit.kelvin
        self.timestep = 0.002 * unit.picosecond
        self.friction_coeff = 1 / unit.picosecond

        self.num_features = 45
        self.num_eigvecs = 6
        self.num_cvs = 2
        self.loss_method = "vamp2"
        self.learning_rate = 5e-3
        self.frac_test = 0.1

        # attributes below this point are computed from the attributes above
        # replace defaults from DefaultMixin here
        self.replace_defaults(kwargs)

        # kT as a float, units of kJ/mol
        self.kT = kT_in_kJ_per_mol(self.temperature)

        # take one frame per lagtime.
        num_frames = self.simulation_time / self.lagtime

        # simulation must be a perfect multiple of the lagtime
        assert np.isclose(
            num_frames, round(num_frames)
        ), "Simulations must be a whole number of frames.  Check simulation_time and lagtime"
        self.num_frames = round(num_frames)

        self.timesteps_per_frame = round(self.lagtime / self.timestep)

        # assuming we always report once per frame
        self.report_interval = self.timesteps_per_frame

        self.ns_per_frame = self.lagtime.value_in_unit(unit.nanosecond)

        # string for labeling simulation output
        self.ns = f"{self.ns_per_frame * self.num_frames:.1f}"

        self.working_dir = Path(self.working_dir)
        self.working_dir.mkdir(parents=True, exist_ok=True)


class SimulationParametersMetaD(SimulationParameters):
    """SimulationParameters with Well-Tempered Metadynamics bias settings.

    Note: because we're using this behaviour:
        super().__init__(**kwargs) --> DefaultMixin.__init__(**kwargs)
    any field that does not appear in SimulationParameters.__init__ kwargs will give an error.

    Workaround is to use 'pop' with default values, which matches the desired behaviour.
    """

    def __init__(self, **kwargs):
        # MetaD-specific overrides, extracted before replace_defaults runs
        self.bias_factor = kwargs.pop("tau_G", 5.0)
        self.tau_G = kwargs.pop("tau_G", 120 * unit.femtoseconds)
        self.height = kwargs.pop("height", 1.20 * unit.kilojoule_per_mole)
        self.width = kwargs.pop("width", np.array([0.1, 0.1]))

        # replace_defaults applied to remaining parameters
        super().__init__(**kwargs)

        # post-init
        self.num_gaussians = round(self.simulation_time / self.tau_G)
        self.steps_per_gaussian = round(self.tau_G / self.timestep)


class SimulationParametersOPES(SimulationParameters):
    """SimulationParameters with OPES bias settings."""

    def __init__(self, **kwargs):
        self.bias_factor = kwargs.pop("bias_factor", 15.0)
        self.dist_regularization = kwargs.pop("dist_regularization", 1.0e-7)

        super().__init__(**kwargs)
