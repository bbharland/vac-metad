"""Convenient handles for one simulation's extracted, learned, and projected data."""

import itertools
from pathlib import Path

import numpy as np
import mdtraj as md
from openmm import unit

# Project-specific imports — adjust these to match your package layout.
from .param import SimulationParameters
from .DataHandles import DataHandles


def simulation_data(p, subdir=None, lagframes=1):
    """Build a :class:`SimulationData` for ``basedir/subdir``.

    Parameters
    ----------
    p : SimulationParameters or Path or str
        Determines the base directory.  A ``SimulationParameters`` also supplies
        the lagtime; a bare path/string defaults the lagtime to 1.

    subdir : None, int, or str
        - ``None``: use the base directory directly.
        - int ``n``: use the ``step_n`` subdirectory.
        - str: use the named subdirectory.

    lagframes : int
        Number of simulation frames corresponding to one lagtime.

    Returns
    -------
    SimulationData
    """
    if isinstance(p, SimulationParameters):
        basedir = Path(p.working_dir)
        lagtime = p.lagtime.value_in_unit(unit.picosecond)
    elif isinstance(p, (str, Path)):
        basedir = Path(p)
        lagtime = 1
    else:
        raise TypeError(f"Cannot build simulation_data from p of type {type(p).__name__}")

    if subdir is None:
        working_dir = basedir
    elif isinstance(subdir, str):
        working_dir = basedir / subdir
    elif isinstance(subdir, (int, np.integer)):
        working_dir = basedir / f"step_{subdir}"
    else:
        raise TypeError(f"Cannot handle subdir of type {type(subdir).__name__}")

    return SimulationData(working_dir, lagtime, lagframes)


def simulation_data_test(working_dir, sd, labels):
    """Create and return a new SimulationData object at 'working_dir' for testing purposes.

    It will match variables from passed SimulationData object 'sd':
        * lagtime
        * items from list 'labels' will be symlinked
    """

    # create new working_dir and symlinked files
    working_dir.mkdir(parents=True, exist_ok=True)
    for label in labels:
        src = sd.files[label].resolve()
        dst = working_dir / src.name
        dst.unlink(missing_ok=True)
        dst.symlink_to(src)

    # create new SimulationData object and return it
    sd2 = simulation_data(working_dir)
    sd2.lagtime = sd.lagtime
    return sd2


class SimulationData(DataHandles):
    """Handles for one simulation's extracted, learned, and projected data.

    Typical sequence:
        1. Extract ``dihedrals`` and ``features`` from a trajectory
           (:meth:`save_feature_data`), usually from a ``.h5`` file.
        2. Train an SRV on the features, then save eigenfunctions/CVs
           (:meth:`save_eigen_data`).
        3. Project eigenfunctions onto a dihedral grid
           (:meth:`save_grid_data`).

    TODO: extend to an EnhancedSimulationData that also stores bias-potential
    values and frame weights — without adding those calculations here.
    """

    data_filenames = [
        "final_positions.pickle",
        # Extracted simulation data
        "dihedrals.npy",
        "features.npy",
        "psi.npy",
        "cvs.npy",
        # SRV
        "vampnet.pickle",
        "srv.pickle",
        "srv_net.pt",
        "eigvals.npy",
        "timescales.npy",
        # Data projected onto dihedral grids
        "theta_grid.npy",
        "feature_grid.npy",
        "psi_grid.npy",
        # MSM states
        "states.npy",
        "states_core.npy",
        # Enhanced sampling
        "weights.npy",
        "biases.npy"
    ]
    other_filenames = {
        "h5file": "traj.h5",
        "dcdfile": "traj.dcd",
        "outfile": "traj.out",
    }

    def __init__(self, working_dir, lagtime, lagframes):
        """Parameters
        ----------
        working_dir : str or Path
            Directory for this simulation's data files.
        lagtime : float
            Time separating transitions (tau, in ps).
        lagframes : int
            Number of simulation frames making up one lagtime.
        """
        super().__init__(working_dir)
        self.lagtime = lagtime
        self.lagframes = lagframes

    def save_feature_data(self, periodic=True, recalculate=False, pdbfile=None):
        """Compute and save ``dihedrals`` and ``features`` if missing.

        Writes (when absent, or when *recalculate* is True):

            dihedrals.npy : (num_frames, 2)
            features.npy  : (num_frames, num_features)

        The trajectory is only loaded if at least one file needs writing.  A
        ``.dcd`` file is used when *pdbfile* is given and the dcd exists;
        otherwise the ``.h5`` file is read.

        Parameters
        ----------
        periodic : bool
            Whether heavy-atom distances respect periodic boundaries.
        recalculate : bool
            Recompute and overwrite even if the files already exist.
        pdbfile : str, optional
            Topology for reading the ``.dcd`` trajectory.  If omitted, the
            ``.h5`` file is used instead.
        """
        need_dihedrals = recalculate or not self.files["dihedrals"].exists()
        need_features = recalculate or not self.files["features"].exists()
        if not (need_dihedrals or need_features):
            print("dihedrals and features already present in", self.working_dir)
            return

        traj = self._load_trajectory(pdbfile)

        if need_dihedrals:
            print("writing", self.files["dihedrals"])
            dihedrals = np.hstack([md.compute_phi(traj)[1], md.compute_psi(traj)[1]])
            self._save_object("dihedrals", dihedrals)

        if need_features:
            print("writing", self.files["features"])
            features = self.heavy_atom_distances(traj, periodic=periodic)
            self._save_object("features", features)

    def _load_trajectory(self, pdbfile=None):
        """Load this simulation's trajectory, preferring dcd+pdb when available."""
        if pdbfile and self.files["dcdfile"].exists():
            return md.load_dcd(self.files["dcdfile"], top=pdbfile)
        if self.files["h5file"].exists():
            return md.load(self.files["h5file"])
        raise FileNotFoundError("Need either an h5 file, or a dcd file plus pdbfile")

    @staticmethod
    def heavy_atom_distances(traj, periodic=True):
        """Pairwise distances between all heavy (non-water, non-H) atoms.

        Returns
        -------
        np.ndarray
            Shape ``(num_frames, num_pairs)``.  For alanine dipeptide this is
            the 45 heavy-atom-pair distances.
        """
        heavy_atoms = traj.topology.select("resname != HOH && type != H")
        atom_pairs = np.array(list(itertools.combinations(heavy_atoms, 2)))
        return md.compute_distances(traj, atom_pairs, periodic=periodic)

    def save_eigen_data(self, srv, features=None, num_cvs=2):
        """Transform and save SRV eigenfunction data for this simulation.

        Call once the SRV has been fitted.  The SRV is not intended to be used
        afterwards, since ``srv.srv_net()`` moves ``srv.net`` to the CPU (it is
        restored to ``srv.device`` before the objects are written).

        Parameters
        ----------
        srv : fitted SRV
        features : np.ndarray, optional
            Defaults to ``self.features``.
        num_cvs : int
            Number of leading eigenfunctions to keep as collective variables.
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
            "srv_net": srv.srv_net(),  # moves srv.net to the CPU
        }
        srv.net.to(device=srv.device)  # restore before pickling srv
        self.save_and_assign_objects(labels_objects)

    def save_grid_data(self, srv, num_points):
        """Project eigenfunctions onto a ``num_points`` dihedral grid.

        Call after :meth:`save_eigen_data` (or with an SRV whose ``net`` is on
        ``srv.device``, which :meth:`save_eigen_data` leaves it as).
        """
        theta_grid = np.linspace(-np.pi, np.pi, num_points)

        feature_grid = feature_grid_over_dihedrals(
            self.features, self.dihedrals, theta_grid
        )
        psi_grid = psi_grid_from_feature_grid(feature_grid, srv)

        self.save_and_assign_objects(
            {
                "theta_grid": theta_grid,
                "feature_grid": feature_grid,
                "psi_grid": psi_grid,
            }
        )


def feature_grid_over_dihedrals(features, dihedrals, theta_grid):
    """Map one feature vector to each occupied dihedral grid point.

    Returns
    -------
    np.ndarray
        Shape ``(num_points, num_points, num_features)``; grid points with no
        sampled frame are NaN.
    """

    def grid_index(theta, dtheta):
        return round((theta + np.pi) / dtheta)

    num_points = len(theta_grid)
    dtheta = theta_grid[1] - theta_grid[0]
    num_features = features.shape[1]
    feature_grid = np.full((num_points, num_points, num_features), np.nan)

    for dihedral, x in zip(dihedrals, features):
        i = grid_index(dihedral[0], dtheta)
        j = grid_index(dihedral[1], dtheta)
        if np.isnan(feature_grid[i, j, 0]):
            feature_grid[i, j, :] = x
    return feature_grid


def psi_grid_from_feature_grid(feature_grid, srv):
    """Evaluate the SRV eigenfunctions across the occupied grid points.

    The network is applied once to all occupied points (rather than one point
    at a time), and the number of eigenfunctions is taken from the SRV output.

    Returns
    -------
    np.ndarray
        Shape ``(num_points, num_points, num_eigfuncs)``; empty grid points
        are NaN.
    """
    num_points = feature_grid.shape[0]
    filled = ~np.isnan(feature_grid[:, :, 0])  # (num_points, num_points)
    psi = np.asarray(srv(feature_grid[filled]))  # (num_filled, num_eigfuncs)

    psi_grid = np.full((num_points, num_points, psi.shape[1]), np.nan)
    psi_grid[filled] = psi
    return psi_grid
