import numpy as np
import torch
import openmm as mm
import openmm.unit as unit
from openmm.app import PDBFile
import mdtraj as md

from .gaussians import Gaussians
from .util import to_torch


class SimulationCVS:
    """Used to compute CV's from
        (a) pdb files
        (b) simulation.context.getState

    This is required by Metadynamics.add_gaussian(cvs)
    """
    def __init__(self, pdb_filename):
        self.pdb_filename = pdb_filename

        traj = md.load(pdb_filename)
        self.ala_atoms = traj.topology.select('resname != HOH')

        heavy_atoms = traj.topology.select('resname != HOH && type != H')
        self.atom_pairs = np.array([(i, j)
                                    for i in heavy_atoms
                                    for j in heavy_atoms if i < j])

    def pdb(self):
        return PDBFile(self.pdb_filename)

    def pdb_cvs(self, net):
        traj = md.load(self.pdb_filename)
        xyz = traj[0].xyz.squeeze()
        return self.xyz_to_cvs(net, xyz)

    def sim_cvs(self, net, simulation):
        state = simulation.context.getState(getPositions=True)
        xyz = np.array(state.getPositions()._value)
        return self.xyz_to_cvs(net, xyz)

    def xyz_to_cvs(self, net, xyz):
        positions = to_torch(xyz[self.ala_atoms])
        x = self.featurize(positions)
        return net(x).numpy().flatten()

    def featurize(self, positions):
        return torch.stack([
            torch.sqrt(torch.sum(torch.square(positions[j] - positions[i])))
            for (i, j) in self.atom_pairs
        ]).unsqueeze(0)


def metadynamics(temperature, bias_factor, height, width):
    """Deal with units and return a Metadynamics object.

    Parameters
    ----------
    temperature : openmm.unit.quantity.Quantity
        Temperature the simulation is run at, in K.
    bias_factor : float
        For WTMetaD, deltaT = T * (bias_factor - 1)
    height : openmm.unit.quantity.Quantity
        Standard height of biasing Gaussians (will convert to kJ/mol).
    width : np.ndarray with shape (num_cvs,)
        The Gaussian widths along the direction of each CV.
    """
    assert unit.is_quantity(temperature), "temperature must be a unit in K"
    assert bias_factor > 1, f"{bias_factor = } must be greater than 1"
    assert unit.is_quantity(height), "height must be a unit"
    assert (
        isinstance(width, np.ndarray) and width.ndim == 1
    ), "width must be 1d np.ndarray"
    deltaT = temperature * (bias_factor - 1)
    betap = 1 / (unit.MOLAR_GAS_CONSTANT_R * deltaT)
    betap = betap / (unit.mole / unit.kilojoule)
    height = height / (unit.kilojoule / unit.mole)
    return Metadynamics(betap, height, width)


class Metadynamics:
    """Well-tempered metadynamics bias, built on a Gaussians kernel sum.

    Holds a growing Gaussians object (the source of truth for the deposited
    kernels) plus the policy that governs deposition.  Evaluation of the bias
    potential delegates to Gaussians; this class owns only the well-tempered
    height rule, the torch bridge, and compression bookkeeping.

    Reference: Valsson, Tiwary, and Parrinello, "Enhancing Important
    Fluctuations: Rare Events and Metadynamics from a Conceptual Viewpoint",
    Annu. Rev. Phys. Chem. 67, 159 (2016).
    https://www.annualreviews.org/doi/abs/10.1146/annurev-physchem-040215-112229
    """

    def __init__(self, betap, height, width):
        """
        betap : float
            Bias potential tempering factor.
        height : float
            Base height for Gaussians (in kJ/mol).
        width : np.ndarray with shape (num_cvs,)
            The Gaussian widths along the direction of each CV.
        """
        self._betap = betap
        self._height = height
        self._width = np.asarray(width, dtype=np.float32)
        num_cvs = len(self._width)
        # Start empty.  An empty Gaussians evaluates to 0.0, so add_gaussian's
        # first call (bias = 0) and bias_potential both work with no guard.
        self.gaussians = Gaussians(
            np.empty(0, dtype=np.float32),
            np.empty((0, num_cvs), dtype=np.float32),
            np.empty((0, num_cvs), dtype=np.float32),
        )
        # index into self.gaussians where the current step's deposits begin
        self._step_start = 0

    def __len__(self):
        return len(self.gaussians)

    def reset(self, gaussians):
        """Replace the kernel set and mark the start of a new step.

        Typically called at the start of a step with the compressed state from
        the previous step, e.g. ``metad.reset(load_gaussians(prev/'current.npz'))``.
        Everything deposited after this call is the current step's contribution
        (see ``step_deposits``).
        """
        assert gaussians.dim == len(self._width), (
            f"CV-dimension mismatch: gaussians.dim={gaussians.dim}, "
            f"expected {len(self._width)}"
        )
        self.gaussians = gaussians
        self._step_start = len(gaussians)

    @property
    def step_deposits(self):
        """The Gaussians deposited since the last reset (this step only,
        uncompressed).  Save this for the per-step log.
        """
        return self.gaussians[self._step_start:]

    # Read-only views into the held kernels.  To replace the kernels wholesale,
    # assign to ``self.gaussians`` (e.g. the value returned by ``compressed``).
    @property
    def heights(self):
        return self.gaussians.heights

    @property
    def centers(self):
        return self.gaussians.centers

    @property
    def widths(self):
        return self.gaussians.widths

    def add_gaussian(self, sn):
        """Deposit a new Gaussian centered at sn : ndarray, shape (num_cvs,).

        Well-tempered height: h = h0 * exp(-betap * V_bias(sn)), evaluated against the bias *before* this kernel is added.
        """
        sn = np.asarray(sn, dtype=np.float32)
        height = np.float32(
            self._height * np.exp(-self._betap * self.bias_potential(sn))
        )
        new = Gaussians(
            np.array([height], dtype=np.float32),
            sn.reshape(1, -1),
            self._width.reshape(1, -1),
        )
        self.gaussians = self.gaussians + new

    def bias_potential(self, s, num_gaussians=None):
        """Return V_bias(s) in kJ/mol.

        If num_gaussians is not None, evaluate using only the first that many deposited kernels (useful for replaying deposition history).
        """
        if num_gaussians is None:
            return self.gaussians(s)
        return self.gaussians[:num_gaussians](s)

    def compressed(self, dist_threshold=1.0, loud=True):
        """Return a new Metadynamics with nearby kernels merged.

        Same policy params, with the kernel set replaced by the compressed
        Gaussians.  Replaces the old standalone kde_compression(metad) helper.
        """
        new = Metadynamics(self._betap, self._height, self._width)
        new.gaussians = self.gaussians.compressed(
            dist_threshold=dist_threshold, loud=loud
        )
        return new

    def force_module(self, net):
        """Factory for the openmm-torch ForceModule.

        Parameters
        ----------
        net : torch.nn.Sequential
            Network mapping features -> collective variables, in inference mode
            (all parameters requires_grad = False).

        Returns
        -------
        ForceModule
            The module to be compiled and added to the simulation.
        """
        assert len(self) > 0, "Can't deal with empty metad object"

        dtype = torch.float32
        width = torch.tensor(self._width, dtype=dtype).unsqueeze(0)
        heights = torch.tensor(self.heights, dtype=dtype)
        centers = torch.tensor(self.centers, dtype=dtype)
        widths = torch.tensor(self.widths, dtype=dtype)
        return ForceModule(net, heights, centers, width, widths)


class ForceModule(torch.nn.Module):
    """Dialanine module for metadynamics in openmm with latent variables.
        * heavy-atom distance featurization (45 features)
        * two latent variables (num_cvs = 2)

        Autodiff only cares about input (positions) and output (bias_potential).  Everything else must be a buffer.

        TODO: Figure out a way to turn off autodiff during 'add_gaussian()'?  Until then, Metadynamics object has to handle this.

        TODO: Figure out a way to initialize with no Gaussians?

        Claude: One thing worth flagging beyond the immediate fix: growing self.heights/centers/widths inside forward is the fragile part of this design (it's also what your own TODO: turn off autodiff during add_gaussian is circling). The reshape patch makes it correct, but if this keeps biting you, the sturdier pattern is to deposit on the Python Metadynamics side and rebuild/reload the force module's buffers from it, rather than mutating buffers through scripted global parameters. Not necessary today — just where I'd look if the deposition path stays brittle.
    """
    def __init__(self, net, heights, centers, width, widths):
        """
        Parameters
        ----------
        net : torch.nn.modules.container.Sequential
            The network that transforms features -> collective variables.  This must be in inference mode for autodiff to work properly.

        heights : torch.Tensor with shape (num_gaussians,)
            Heights of Gaussians in kJ/mol

        centers : torch.Tensor with shape (num_gaussians, num_cvs)
            Locations of Gaussians

        width : torch.Tensor with shape (1, num_cvs)
            Width along each CV for newly deposited Gaussians.

        widths : torch.Tensor with shape (num_gaussians, num_cvs)
            These are identical for all Gaussians until a KDE compression is done.
        """
        super().__init__()
        self.net = net
        self.register_buffer('heights', heights)
        self.register_buffer('centers', centers)
        self.register_buffer('width', width)
        self.register_buffer('widths', widths)

    def forward(self, positions, add_gaussian, height, center1, center2):
        """Calculate the bias potential energy from positions.

        Parameters
        ----------
        positions : torch.Tensor with shape (nparticles, 3)
           positions[i,k] is the position (in nanometers) of spatial dimension k of particle i

        add_gaussian : torch.Scalar, cast to bool by torch.jit
            Global parameters are handled by the TorchForce swig object:
                torch_force.addGlobalParameter('add_gaussian', False)
            and modified from the context:
                simulation.context.setParameter('add_gaussian', True)

        If add_gaussian = True, use values below for new Gaussian
        height : torch.Scalar = torch.Tensor with shape ()
        center1 : torch.Scalar = first component of center, s1
        center2 : torch.Scalar = second component, s2

        Returns
        -------
        potential : torch.Scalar
           The potential energy (in kJ/mol)
        """
        # if add_gaussian:
        #     device = self.heights.device
        #     h = height.to(device=device)
        #     c = torch.stack([center1, center2]).to(device=device)
        #     self.heights = torch.cat([self.heights, h.unsqueeze(0)], dim=0)
        #     self.centers = torch.cat([self.centers, c.unsqueeze(0)], dim=0)
        #     self.widths = torch.cat([self.widths, self.width], dim=0)
        if add_gaussian:
            device = self.heights.device
            h = height.reshape(1).to(device=device)
            c = torch.stack([center1, center2]).reshape(1, -1).to(device=device)
            self.heights = torch.cat([self.heights, h], dim=0)
            self.centers = torch.cat([self.centers, c], dim=0)
            self.widths  = torch.cat([self.widths, self.width], dim=0)

        x = self.featurize(positions)
        s = self.net(x)
        return self.bias_potential(s)

    def featurize(self, positions):
        """Return x : torch.Tensor with shape (1, num_features)
            This is the input for the network
        """
        return torch.stack([
            torch.sqrt(torch.sum(torch.square(positions[j] - positions[i])))
            for (i, j) in ((1, 4), (1, 5), (1, 6), (1, 8), (1, 10),
                           (1, 14), (1, 15), (1, 16), (1, 18), (4, 5),
                           (4, 6), (4, 8), (4, 10), (4, 14), (4, 15),
                           (4, 16), (4, 18), (5, 6), (5, 8), (5, 10),
                           (5, 14), (5, 15), (5, 16), (5, 18), (6, 8),
                           (6, 10), (6, 14), (6, 15), (6, 16), (6, 18),
                           (8, 10), (8, 14), (8, 15), (8, 16), (8, 18),
                           (10, 14), (10, 15), (10, 16), (10, 18), (14, 15),
                           (14, 16), (14, 18), (15, 16), (15, 18), (16, 18))
            ]).unsqueeze(0)

    def bias_potential(self, s):
        """
        Parameters
        ----------
        s : torch.Tensor with shape (1, num_cvs)
           The values of the collective variables

        Returns
        -------
        potential : torch.Scalar
           The bias potential energy (in kJ/mol)
        """
        norm_sqs = torch.sum(torch.square((s - self.centers) / self.widths), 1)
        return torch.sum(self.heights * torch.exp(-0.5 * norm_sqs))

