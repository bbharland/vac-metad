import numpy as np
from pathlib import Path
import openmm.unit as unit
import mdtraj as md

from .DataHandles import DataHandles
from .param import SimulationParameters
from .calcs import (
    feature_grid_over_dihedrals,
    psi_grid_from_feature_grid,
    cvs_grid_from_feature_grid,
)


def simulation_data(p, subdir=None, assign_labels=None, loud=True, num_points=100):
    """Working directory is 'base_dir/subdir'
        1. basedir determined by 'p'
        2. subdir can be
            - ignored       None
            - subdir        subdir is a string
            - step_n        subdir is an integer, n

    Parameters
    ----------
    p : SimulationParameters => basedir = Path(p.working_dir)
                        Path => basedir = p
                         str => basedir = Path(p)

    subdir : None => don't use a subdirectory
              int => the step number for the subdirectory
              str => the name of the subdirectory

    assign_labels : list(str)
        When not None, only assign these labels when instantiating (to save memory)

    loud : Bool
        When True, create working_dir if it does not exist and print a bunch of checks to screen.
        When False, just invoke SimulationData constructor

    num_points : int => sd.theta_grid has num_points points
                None => sd.theta_grid not assigned
    """
    lagframes = 1

    if isinstance(p, SimulationParameters):
        basedir = Path(p.working_dir)
        lagtime = p.lagtime.in_units_of(unit.picosecond)._value
    elif isinstance(p, Path):
        basedir = p
        lagtime = 1
    elif isinstance(p, str):
        basedir = Path(p)
        lagtime = 1
    else:
        raise TypeError(f"Type {type(p)} not known.")

    if subdir is None:
        working_dir = basedir
    elif type(subdir) is int:
        working_dir = basedir / f"step_{subdir}"
    elif type(subdir) is str:
        working_dir = basedir / subdir
    else:
        raise TypeError(f"Can't handle 'subdir' type {type(subdir)}")

    if not working_dir.exists():
        print(f"Creating new directory {working_dir}")
        working_dir.mkdir()

    sd = SimulationData(working_dir, lagtime, lagframes, assign_labels=assign_labels)

    # Always assign 'theta_grid' (unless num_point is explicitly set to None).  This makes code easier to reason about
    if num_points is not None:
        if not sd.files["theta_grid"].exists():
            sd.save_theta_grid(num_points=num_points)
        else:
            sd.assign_objects(["theta_grid"])

    if loud:
        dihedrals_exist = sd.files["dihedrals"].exists()
        features_exist = sd.files["features"].exists()

        if dihedrals_exist and features_exist:
            if hasattr(sd, "features") and sd.features is not None:
                print(
                    f"Loaded SimulationData object from {working_dir} "
                    f"with {len(sd.features)} frames"
                )
            elif hasattr(sd, "dihedrals") and sd.dihedrals is not None:
                print(
                    f"Loaded SimulationData object from {working_dir} "
                    f"with {len(sd.dihedrals)} frames"
                )
            else:
                print(f"Loaded SimulationData object from {working_dir}")
        elif not dihedrals_exist and not features_exist:
            print(f"Creating new SimulationData object with no data at {working_dir}")
        else:
            missing = "dihedrals" if features_exist else "features"
            print(
                f"WARNING: {working_dir} has one of dihedrals/features but "
                f"not the other (missing {missing})"
            )

    # make sure SRV net is on correct device
    if sd.files["srv"].exists() and hasattr(sd, "srv"):
        sd.srv.net.to(device=sd.srv.device)

    return sd


class SimulationData(DataHandles):
    """The sequence is:
        1. Get dihedrals, features, biases weights out of a trajectory
            sd.save_feature_data(recalculate=True, pdbfile=p.pdb_file)
            sd.save_and_assign_objects({
                'biases': np.array(simulation_biases),
                'weights': np.array(simulation_weights)
                })

        2. Train network with features
            sd.save_and_assign_objects({'vampnet': vn})

        3. Save eigenfunctions and CVs for trajectory
            sd.save_eigen_data(srv)

        4. Project trajectory eigenfunctions/CVs onto dihedral grid
            sd.save_grid_data(srv)
            sd.save_and_assign_objects({'bias_shift': bias_shift})

    KDE/FES/bias-grid data (formerly save_kde_fes_bias_data) has moved to
    data_attic.py pending the grid-calculations redesign -- see claude.txt.
    """

    data_filenames = [
        # ---------------------- numpy arrays -------------------------
        "widths.npy",
        #                       current step data
        "dihedrals.npy",
        "features.npy",
        "heavy_atoms.npy",
        "weights.npy",
        "biases.npy",
        "eigvals.npy",
        "timescales.npy",
        "bias_shift.npy",
        #                       full dataset
        "psi.npy",
        "cvs.npy",
        #                       grids
        "s1.npy",
        "s2.npy",
        "grid.pickle",
        "bias.pickle",
        "fes.pickle",
        "theta_grid.npy",
        "feature_grid.npy",
        "psi_grid.npy",
        "cvs_grid.npy",
        #                       MSM states
        "states.npy",
        "states_core.npy",
        #                       converging distributions
        # 'num_frames_conv.npy',  'pgrids_conv.npy',
        # ---------------------- pytorch models -----------------------
        # 'force_module.pt' needs to be opened with 'torch.jit.load'
        "srv_net.pt",
        # ---------------------- objects ------------------------------
        "vampnet.pickle",
        "srv.pickle",
        "kde.pickle",
        "kde_c.pickle",
        "final_positions.pickle",
        "metad.pickle",
        "metad_kde.pickle",
    ]
    other_filenames = {
        "h5file": "traj.h5",
        "dcdfile": "traj.dcd",
        "outfile": "traj.out",
    }

    def __init__(self, working_dir, lagtime, lagframes, assign_labels=None):
        """Parameters
        ----------
        working_dir : str
            Location for data files to be written
        lagtime : float
            Time separating transitions (tau, in ps)
        lagframes : int
            Number of simulation frames that make up this lagtime
        assign_label : list(str)
            If None: assign all objects in SimulationData.data_filenames
            Else: assign all obects in list of labels
        """
        super().__init__(working_dir, assign_labels=assign_labels)
        self.lagtime = lagtime
        self.lagframes = lagframes

    def save_theta_grid(self, num_points=100):
        """Save the grid over which dihedral space is discretized.  This doesn't really fit anywhere else.

        Note that 'theta_grid' is assigned always (when 'simlation_data' function is used for construction)!
        """
        theta_grid = np.linspace(-np.pi, np.pi, num_points)
        self.save_and_assign_objects({"theta_grid": theta_grid})

    def save_feature_data(self, periodic=True, recalculate=False, pdbfile=None):
        """Save files if they don't exist (or recalculate is set to True):

            dihedrals.npy : ndarray with shape (num_frames, 2)
            features.npy : ndarray with shape (num_frames, num_features)

        Files are produced by reading existing dcd or h5 files.  If the dcd file exists, that will be read.  Otherwise, h5 file is used.

        Parameters
        ----------
        pdbfile : str
            If anything is provided here, prefer using dcd file.  Otherwise h5 file will be used.
        """
        if pdbfile and self.files["dcdfile"].exists():
            traj = md.load_dcd(self.files["dcdfile"], top=pdbfile)
        elif self.files["h5file"].exists():
            traj = md.load(self.files["h5file"])
        else:
            raise FileNotFoundError("Need either h5 file or dcd + pdb file")

        file = self.files["dihedrals"]
        if not file.exists() or recalculate:
            print("writing", file)
            dihedrals = np.hstack([md.compute_phi(traj)[1], md.compute_psi(traj)[1]])
            self._save_and_assign_object(file, dihedrals)
        else:
            print("not recalculating", file)

        file = self.files["features"]
        if not file.exists() or recalculate:
            print("writing", file)
            features = self.heavy_atom_distances(traj, periodic=periodic)
            self._save_and_assign_object(file, features)
        else:
            print("not recalculating", file)

    def _heavy_atom_indices(self, traj):
        return traj.topology.select("resname != HOH && type != H")

    def heavy_atom_distances(self, traj, periodic=True):
        """Return : ndarray with shape (num_frames, num_pairs)
        The 45 atom-pair distances for an mdtraj Trajectory
        """
        heavy_atoms = self._heavy_atom_indices(traj)
        atom_pairs = np.array(
            [(i, j) for i in heavy_atoms for j in heavy_atoms if i < j]
        )
        return md.compute_distances(traj, atom_pairs, periodic=periodic)

    def heavy_atom_positions(self, traj):
        """Save heavy-atom Cartesian coordinates.

        Parameters
        ----------
        traj : mdtraj.Trajectory

        Saves
        -----
        heavy_atoms.npy : ndarray, shape (num_heavy_atoms * 3, num_frames)
            Heavy-atom xyz coordinates, one column per frame.
        """
        heavy_atoms = self._heavy_atom_indices(traj)
        xyz = traj.xyz[:, heavy_atoms, :]
        positions = xyz.reshape(xyz.shape[0], -1).T
        self.save_and_assign_objects({"heavy_atoms": positions})
        return positions

    def save_eigen_data(self, srv, features=None, num_cvs=2):
        """Transform/save vampnet eigenfunction data from a single simulation.
        * call once SRV has been fitted
        * SRV not intended to be used after this (sends srv.net to CPU)
        """
        if features is None:
            features = self.features
        psi = srv(features)

        labels_objects = {
            "srv": srv,
            "eigvals": srv.eigvals,
            "timescales": srv.timescales(),
            "psi": psi,
            "cvs": psi[:, :num_cvs],
            "srv_net": srv.srv_net(),  # sends srv.net to the CPU
        }
        srv.net.to(device=srv.device)  # put it back
        self.save_and_assign_objects(labels_objects)

    def save_grid_data(self, srv):
        """Project eigenfunctions onto the dihedral grid.

        Call after save_eigen_data (or with an SRV whose net is already on
        srv.device, as save_eigen_data leaves it).
        """
        feature_grid = feature_grid_over_dihedrals(
            self.features, self.dihedrals, self.theta_grid
        )
        psi_grid = psi_grid_from_feature_grid(feature_grid, srv)
        cvs_grid = cvs_grid_from_feature_grid(feature_grid, srv)

        self.save_and_assign_objects(
            {
                "feature_grid": feature_grid,
                "psi_grid": psi_grid,
                "cvs_grid": cvs_grid,
            }
        )
