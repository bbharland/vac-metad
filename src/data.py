import numpy as np
from pathlib import Path
import openmm.unit as unit
import mdtraj as md

from .DataHandles import DataHandles
from .param import SimulationParameters
from .hist2d import histogram2d
from .opes import bias_potential
from .force import force_module
from .calcs import (
    feature_grid_over_dihedrals,
    psi_grid_from_feature_grid,
    cvs_grid_from_feature_grid,
    calc_kde_grid,
    free_energy_surface,
    calc_bias_grid,
    calc_bias_grid_fm,
    calc_fes_from_bias
)


def simulation_data(p, subdir=None, assign_labels=None, loud=True, num_points=100):
    """Working directory is 'base_dir/subdir'
        1. basedir determined by 'p'
        2. subdir can be
            - ignored       None
            - subdir        subdir is a string
            - step_n        subdir is an integer, n

    Parameters:
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

    TODO: the lagtime and lagframes should come from the sim parameters object, no?

    TODO: in 'loud' section, don't check both dihedrals and features this way.
    Better to check that they are either both there or neither.  Complain if there is just one.

    TODO:  comment on the fact that 'theta_grid' get assigned no matter what if it is found not to exist
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
        raise TypeError(f'Type {type(p)} not known.')

    if subdir is None:
        working_dir = basedir
    elif type(subdir) is int:
        working_dir = basedir / f'step_{subdir}'
    elif type(subdir) is str:
        working_dir = basedir / subdir
    else:
        raise TypeError(f"Can't handle 'subdir' type {type(subdir)}")

    if not working_dir.exists():
        print(f'Creating new directory {working_dir}')
        working_dir.mkdir()

    sd = SimulationData(working_dir, lagtime, lagframes, assign_labels=assign_labels)

    # Always assign 'theta_grid' (unless num_point is explicitly set to None).  This makes code easier to reason about
    if num_points is not None:
        if not sd.files['theta_grid'].exists():
            sd.save_theta_grid(num_points=num_points)
        else:
            sd.assign_objects(['theta_grid'])

    if loud:
        if sd.files['features'].exists():
            if hasattr(sd, 'features') and sd.features is not None:
                print(f'Loaded SimulationData object from {working_dir} with {len(sd.features)} frames')
            elif hasattr(sd, 'dihedrals') and sd.dihedrals is not None:
                print(f'Loaded SimulationData object from {working_dir} with {len(sd.dihedrals)} frames')
            else:
                print("Simulation exists, but not setting 'features' or 'features' attributes")
        else:
            print(f'Creating new SimulationData object with no data at {working_dir}')

    # TODO: UNCOMMENT THIS!
    # # make sure SRV net is on correct device
    # if sd.files['srv'].exists() and hasattr(sd, 'srv'):
    #     sd.srv.net.to(device=sd.srv.device)

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
            sd.save_kde_data(kde, kde_c, s1, s2)
            sd.save_and_assign_objects({'bias_shift': bias_shift})
    """
    data_filenames = [
        #---------------------- numpy arrays -------------------------
        'widths.npy',
        #                       current step data
        'dihedrals.npy',        'features.npy',      'weights.npy',
        'biases.npy',           'eigvals.npy',       'timescales.npy',
        'bias_shift.npy',
        #                       full dataset
        'psi.npy',              'cvs.npy',
        #                       grids
        's1.npy',               's2.npy',            'grid.pickle',
        'bias.pickle',          'fes.pickle',        'theta_grid.npy',
        'feature_grid.npy',     'psi_grid.npy',      'cvs_grid.npy',
        #                       MSM states
        'states.npy',           'states_core.npy',
        #                       converging distributions
        # 'num_frames_conv.npy',  'pgrids_conv.npy',
        #---------------------- pytorch models -----------------------
        # 'force_module.pt' needs to be opened with 'torch.jit.load'
        # 'srv_net.pt',    # TODO: NEED TO UNCOMMENT THIS LINE!
        #---------------------- objects ------------------------------
        # 'vampnet.pickle',       'srv.pickle', # TODO: UNCOMMENT THIS LINE!
        'kde.pickle',
        'kde_c.pickle',         'final_positions.pickle',
        'metad.pickle',         'metad_kde.pickle'
    ]
    other_filenames = {'h5file': 'traj.h5', 'dcdfile': 'traj.dcd',
                       'outfile': 'traj.out'}

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
        # print(f'{type(self)} init: {assign_labels = }')
        super().__init__(working_dir, assign_labels=assign_labels)
        self.lagtime = lagtime
        self.lagframes = lagframes

    def save_theta_grid(self, num_points=100):
        """Save the grid over which dihedral space is discretized.  This doesn't really fit anywhere else.

        Note that 'theta_grid' is assigned always (when 'simlation_data' function is used for construction)!
        """
        theta_grid = np.linspace(-np.pi, np.pi, num_points)
        self.save_and_assign_objects({'theta_grid': theta_grid})

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
        if pdbfile and self.files['dcdfile'].exists():
            traj = md.load_dcd(self.files['dcdfile'], top=pdbfile)
        elif self.files['h5file'].exists():
            traj = md.load(self.files['h5file'])
        else:
            raise FileNotFoundError('Need either h5 file or dcd + pdb file')

        file = self.files['dihedrals']
        if not file.exists() or recalculate:
            print('writing', file)
            dihedrals = np.hstack([md.compute_phi(traj)[1],
                                   md.compute_psi(traj)[1]])
            self._save_and_assign_object(file, dihedrals)
        else:
            print('not recalculating', file)

        file = self.files['features']
        if not file.exists() or recalculate:
            print('writing', file)
            features = self.heavy_atom_distances(traj, periodic=periodic)
            self._save_and_assign_object(file, features)
        else:
            print('not recalculating', file)

    def heavy_atom_distances(self, traj, periodic=True):
        """Return : ndarray with shape (num_frames, num_pairs)
            The 45 atom-pair distances for an mdtraj Trajectory
        """
        heavy_atoms = traj.topology.select('resname != HOH && type != H')
        atom_pairs = np.array([(i, j)
                               for i in heavy_atoms
                               for j in heavy_atoms if i < j])
        return md.compute_distances(traj, atom_pairs, periodic=periodic)

    def save_eigen_data(self, srv, features=None, num_cvs=2):
        """Transform/save data from a single simulation.
            * call once SRV has been fitted
            * SRV not intended to be used after this (sends srv.net to CPU)
        """
        if features is None:
            features = self.features
        psi = srv(features)

        feature_grid = feature_grid_over_dihedrals(
            self.features, self.dihedrals, self.theta_grid
        )
        psi_grid = psi_grid_from_feature_grid(feature_grid, srv)
        cvs_grid = cvs_grid_from_feature_grid(feature_grid, srv)

        labels_objects = {
            'srv': srv,
            'eigvals': srv.eigvals,
            'timescales': srv.timescales(),
            'psi': psi,
            'cvs': psi[:, :num_cvs],
            'feature_grid': feature_grid,
            'psi_grid': psi_grid,
            'cvs_grid': cvs_grid,
            'srv_net': srv.srv_net(), # sends srv.net to the CPU
        }
        srv.net.to(device=srv.device) # put it back
        self.save_and_assign_objects(labels_objects)

    def save_kde_fes_bias_data(self, p, s1, s2, widths, kde, kde_c):
        """Saves data in variables:

        s1 : array, shape (len_s1,)
        s2 : array, shape (len_s1,)
        widths : array, shape (2,)

        kde : KDE          Uncompressed, normalized
        kde_c : KDE        Compresssed & normalized

        grid : dict
            'hist' : The histogram of P(s)
            'kde' : The (compressed) KDE estimate for P(s)

        fes : dict
            'hist' : F(s) computed using histogram P(s)
            'kde' : F(s) computed using KDE P(s)

        bias : dict
            'bp' : w(s) computed using opes.BiasPotential
            'fm' : w(s) computed using force.ForceModule
            'fes' : F(s) computed using w(s) from BiasPotential
        """
        self.save_and_assign_objects({'s1': s1, 's2': s2, 'widths': widths,
                                      'kde': kde, 'kde_c': kde_c})

        temperature = p.temperature
        epsilon = p.dist_regularization
        bias_factor = p.bias_factor

        grid = {
            'hist': histogram2d(s1, s2, self.cvs),
            'kde': calc_kde_grid(s1, s2, kde_c)
        }
        fes = {
            'hist': free_energy_surface(grid['hist'], temperature, epsilon),
            'kde': free_energy_surface(grid['kde'], temperature, epsilon)
        }
        self.save_and_assign_objects({'grid': grid,  'fes': fes})

        bp = bias_potential(p, kde_c)
        fm = force_module(bp, kde_c, self.srv_net.cpu())
        gr = calc_bias_grid(s1, s2, bp)
        bias = {
            'bp': gr,
            'fm': calc_bias_grid_fm(s1, s2, fm),
            'fes': calc_fes_from_bias(gr, bias_factor)
        }
        self.save_and_assign_objects({'bias': bias})

    # def save_kde_fes_data(self, s1, s2, widths, kde, kde_c, temperature, epsilon):
    #     from .hist2d import histogram2d
    #     from .calcs import calc_kde_grid, free_energy_surface

    #     grid = {
    #         'hist': histogram2d(s1, s2, self.cvs),
    #         'kde': calc_kde_grid(s1, s2, kde_c)
    #     }
    #     fes = {
    #         'hist': free_energy_surface(grid['hist'], temperature, epsilon),
    #         'kde': free_energy_surface(grid['kde'], temperature, epsilon)
    #     }
    #     labels_objects = {
    #         's1': s1, 's2': s2, 'widths': widths,
    #         'kde': kde, 'kde_c': kde_c,
    #         'grid': grid, 'fes': fes,
    #     }
    #     self.save_and_assign_objects(labels_objects)

    # def dataset_srv_data(self, srv, collected_data, num_cvs=2, pad=0.07):
    #     """Replacing 'save_eigen_data': call once SRV has been fitted

    #     Save:
    #         eigvals.npy
    #         timescales.npy

    #         dihedrals_full.npy  # --- full dataset ---
    #         weights_full.npy
    #         psi.npy
    #         cvs.npy

    #         srv.pickle          # --- objects ---
    #         srv_net.pt

    #     Parameters
    #     ----------
    #     srv : vampnet.SRV
    #     collected_data : dict(list(np.ndarray))
    #         Data collection from '.util.prepare_data'
    #     pad : float : for 'ranges_data'

    #     Return
    #     ------
    #     ranges : list(list(float))
    #         Full set of ranges from psi.npy
    #     cvs_list : list(np.ndarray)
    #         List of transformed data (CVs) for plotting
    #     """
    #     from .hist2d import ranges_data

    #     # transform feature data and be done with SRV
    #     # - srv.net will be sent to cpu in srv.srv_net()
    #     psi_list = [srv(x) for x in collected_data['features']]
    #     cvs_list = [psi[:, :num_cvs] for psi in psi_list]
    #     psi = np.vstack(psi_list)

    #     labels_objects = {
    #         'srv': srv,
    #         'eigvals': srv.eigvals,
    #         'timescales': srv.timescales(),
    #         'psi': psi,
    #         'cvs': psi[:, :num_cvs],
    #         'srv_net': srv.srv_net(),
    #         'dihedrals_full': np.vstack(collected_data['dihedrals']),
    #         'weights_full': np.hstack(collected_data['weights']),
    #     }
    #     self.save_and_assign_objects(labels_objects)
    #     return ranges_data(psi, pad=pad), cvs_list, psi_list


# def symlink_to_sd_labels(sd, labels, dst_dir):
#     """Create a symlink in 'dst_dir' pointing to each file [sd.files[label] for label in labels]
#     """
#     if isinstance(dst_dir, str):
#         dst_dir = Path(dst_dir)
#     elif not isinstance(dst_dir, Path):
#         raise TypeError(f'dst_dir is unrecognized type {type(dst_dir)}')

#     for label in labels:
#         src = Path.cwd() / sd.files[label] # src must be absolute path
#         dst = dst_dir / src.name          # dst must include file name
#         if dst.exists():
#             print(f'File {dst} exists.  Not creating new symlink.')
#         else:
#             dst.symlink_to(src)
