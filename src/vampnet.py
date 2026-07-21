"""VAMPNet / SRV implementation.

Revised version of ``vampnet.py``.  Summary of the changes (see accompanying
notes for the full rationale):

* **No SciPy dependency.**  All linear algebra now goes through ``torch.linalg``.
  Double precision is obtained by casting the *final*, one-off SRV eigenvalue
  problem to ``float64`` rather than by detouring through NumPy/SciPy.  This
  keeps a single code path for the whitening + eigendecomposition used both
  during training (``float32``) and at fit time (``float64``).

* **``SRV.srv_net`` no longer mutates ``self.net`` as a side effect.**  This
  was the cause of the cross-device ``RuntimeError`` recorded in the old TODO:
  ``nn.Sequential(*self.net, eig_layer).to('cpu')`` shares the original child
  modules by reference and moved them to CPU in place, while ``self.device``
  still pointed at ``cuda:0``.  We now deep-copy the feature network first.

* **Bug fixes:** a syntax error in ``cov_matrices_weighted`` and a latent shape
  bug in the (unweighted) ``srv_net`` (the stored mean was 2-D, now flattened).

* **De-duplication:** ``SRV`` and ``WeightedSRV`` share one eigensolver.
"""
import copy
import os

import numpy as np
import torch
import torch.nn as nn

from .dataset import (
    TimeLaggedDataset,
    WeightedTimeLaggedDataset,
)
from .util import (
    to_torch,
    torch_device,
    module_device,
)

# Ridge added to covariance matrices before inversion, keeping near-singular
# C0 safely positive definite.  Used by both the training estimate and the
# final SRV fit so the two are consistent.  Pass epsilon=0.0 to recover the
# old (unregularised) SciPy behaviour exactly.
EPSILON = 1e-6


# --------------------------------------------------------------------------- #
# Linear algebra (torch only, dtype/device agnostic)                          #
# --------------------------------------------------------------------------- #
def sym_eig(a: torch.Tensor, epsilon: float = EPSILON):
    """Eigendecomposition of a symmetric matrix with a small ridge for stability.

    Inherits the dtype/device of ``a`` (so a float64 input gives a float64
    decomposition), which is why the ridge identity is built from ``a``.
    """
    ar = a + epsilon * torch.eye(a.shape[0], dtype=a.dtype, device=a.device)
    eigvals, eigvecs = torch.linalg.eigh(ar)
    return eigvals, eigvecs


def sym_inverse(a: torch.Tensor, epsilon: float = EPSILON, return_sqrt: bool = False):
    """Inverse (``C^-1``) or inverse square root (``C^-1/2``) of a symmetric PD matrix.

    Using the eigendecomposition directly (rather than ``inv(sqrtm(.))``) is the
    natural and more stable route for a symmetric positive-definite matrix.
    """
    eigvals, eigvecs = sym_eig(a, epsilon=epsilon)
    if return_sqrt:
        diag = torch.diag(torch.sqrt(1.0 / eigvals))
    else:
        diag = torch.diag(1.0 / eigvals)
    return eigvecs @ diag @ eigvecs.t()


def cov_matrices(x: torch.Tensor, y: torch.Tensor):
    """Unbiased instantaneous/time-lagged covariances, symmetrised over x<->y."""
    xmean = torch.mean(x, dim=0, keepdim=True)
    ymean = torch.mean(y, dim=0, keepdim=True)
    x = x - xmean
    y = y - ymean

    n = x.shape[0]
    c00 = (x.t() @ x) / (n - 1)
    c11 = (y.t() @ y) / (n - 1)
    c01 = (x.t() @ y) / (n - 1)

    mean = 0.5 * (xmean + ymean)
    c0 = 0.5 * (c00 + c11)      # average x, y variance
    c1 = 0.5 * (c01 + c01.t())  # add reverse transitions
    return mean, c0, c1


def cov_matrices_weighted(x: torch.Tensor, xweights: torch.Tensor,
                          y: torch.Tensor, yweights: torch.Tensor):
    """Weighted covariances using the (biased) weighted estimator.

    The correct unbiased estimator is more involved and described at
    https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Weighted_sample_covariance
    """
    def wmean(w, a):
        # (N,1) * (N,F) -> sum over samples -> (F,)
        return torch.sum(w.reshape(-1, 1) * a, dim=0) / torch.sum(w)

    def wcov(w, a, b):
        # A^T W B / sum(w);  (w * a.t()) is (F,N), @ b is (F,F)
        return (w * a.t()) @ b / torch.sum(w)

    xmean = wmean(xweights, x)
    ymean = wmean(yweights, y)
    x = x - xmean
    y = y - ymean

    c00 = wcov(xweights, x, x)
    c01 = wcov(xweights, x, y)
    c11 = wcov(yweights, y, y)

    mean = 0.5 * (xmean + ymean)
    c0 = 0.5 * (c00 + c11)      # average x, y variance
    c1 = 0.5 * (c01 + c01.t())  # add reverse transitions
    return mean, c0, c1


def koopman_matrix(x: torch.Tensor, y: torch.Tensor, epsilon: float = EPSILON):
    """Minibatch estimate of the Koopman matrix during training.

    Subtracting the means of x, y projects out the equilibrium eigenfunction.
    """
    _, c0, c1 = cov_matrices(x, y)
    inv_sqrt_c0 = sym_inverse(c0, epsilon=epsilon, return_sqrt=True)
    return inv_sqrt_c0 @ c1 @ inv_sqrt_c0


def koopman_matrix_weighted(x: torch.Tensor, xweights: torch.Tensor,
                            y: torch.Tensor, yweights: torch.Tensor,
                            epsilon: float = EPSILON):
    """Weighted minibatch estimate of the Koopman matrix during training."""
    _, c0, c1 = cov_matrices_weighted(x, xweights, y, yweights)
    inv_sqrt_c0 = sym_inverse(c0, epsilon=epsilon, return_sqrt=True)
    return inv_sqrt_c0 @ c1 @ inv_sqrt_c0


# --------------------------------------------------------------------------- #
# Factory                                                                     #
# --------------------------------------------------------------------------- #
def vampnet(p, weighted=True):
    device = torch_device()
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    elif device.type == 'cpu':
        torch.set_num_threads(_default_num_threads())
    else:
        raise ValueError(
            f"Unsupported device {device!r}; expected a 'cpu' or 'cuda' device"
        )

    net = nn.Sequential(
        nn.BatchNorm1d(p.num_features),
        nn.Linear(p.num_features, 100), nn.ELU(),
        nn.Linear(100, 100), nn.ELU(),
        nn.Linear(100, 30), nn.ELU(),
        nn.Linear(30, p.num_eigvecs), nn.Tanh()
    )
    if weighted:
        return WeightedVAMPNet(net, device, p.learning_rate, p.loss_method)
    else:
        return VAMPNet(net, device, p.learning_rate, p.loss_method)


def _default_num_threads():
    """Number of CPU threads to use for intra-op parallelism.

    Prefer the affinity mask over os.cpu_count(): on clusters, containers, and
    cgroup-limited jobs it reflects the cores actually allocated to this process,
    not the whole machine. Falls back to cpu_count() where affinity isn't exposed.
    """
    try:
        return len(os.sched_getaffinity(0))   # Linux; respects cpuset/taskset pinning
    except AttributeError:                     # macOS/Windows
        return os.cpu_count() or 1


# --------------------------------------------------------------------------- #
# Training                                                                     #
# --------------------------------------------------------------------------- #
class VAMPNet:
    """Optimize the objective function of the Koopman matrix K.

    vamp1: loss = -(1 + tr K),    where the 1 is the Perron eigenvalue
    vamp2: loss = -(1 + tr KK')

    Pytorch notes:
    --------------
    1. Train, evaluation modes
        net.train() -> net.training = True
        net.eval() -> net.training = False
    Evaluation mode: ignores dropouts, batchnorm taken from saved statistics and not computed on the fly.

    2. Disabling autograd: when you don't need/want to track gradients on parameters
        context manager:    with torch.no_grad():
        set to inference:   for p in net.parameters():
                                p.requires_grad = False
    """
    def __init__(self, net, device, learning_rate, loss_method):
        assert loss_method in ('vamp1', 'vamp2'), (
            f'Invalid loss method {loss_method}'
        )
        self.net = net.to(device=device).float()
        self.device = device
        self.optim = torch.optim.Adam(params=self.net.parameters(),
                                      lr=learning_rate)
        self.loss_method = loss_method

        self._train_scores = []
        self._test_scores = []

    @property
    def train_scores(self):
        return np.array(self._train_scores)

    @property
    def test_scores(self):
        return np.array(self._test_scores)

    def fit(self, data_loader_train, data_loader_test, num_epochs=1, progress=None):
        for epoch in progress(range(num_epochs), desc="VAMPnet epoch",
                              total=num_epochs, leave=False):
            # training
            self.net.train()
            for x, y in data_loader_train:
                self.optim.zero_grad()
                loss = self.loss(
                    self.net(x.to(device=self.device)),
                    self.net(y.to(device=self.device))
                )
                loss.backward()
                self.optim.step()
                self._train_scores.append([epoch + 1, (-loss).item()])

            # validation
            self.net.eval()
            with torch.no_grad():
                for x, y in data_loader_test:
                    loss = self.loss(
                        self.net(x.to(device=self.device)),
                        self.net(y.to(device=self.device))
                    )
                    self._test_scores.append([epoch + 1, (-loss).item()])

    def loss(self, x: torch.Tensor, y: torch.Tensor):
        koopman = koopman_matrix(x, y)
        if self.loss_method == 'vamp1':
            vamp_score = torch.linalg.norm(koopman, ord='nuc')
        else:
            vamp_score = torch.square(torch.linalg.norm(koopman, ord='fro'))
        return -(1 + vamp_score)


class WeightedVAMPNet(VAMPNet):
    def __init__(self, net, device, learning_rate, loss_method):
        super().__init__(net, device, learning_rate, loss_method)

    def fit(self, data_loader_train, data_loader_test, num_epochs=1, progress=None):
        for epoch in progress(range(num_epochs), desc="VAMPnet epoch",
                              total=num_epochs, leave=False):
            # training
            self.net.train()
            for x, wx, y, wy in data_loader_train:
                self.optim.zero_grad()
                loss = self.loss(self.net(x.to(device=self.device)),
                                 wx.to(device=self.device),
                                 self.net(y.to(device=self.device)),
                                 wy.to(device=self.device))
                loss.backward()
                self.optim.step()
                self._train_scores.append([epoch + 1, (-loss).item()])

            # validation
            self.net.eval()
            with torch.no_grad():
                for x, wx, y, wy in data_loader_test:
                    loss = self.loss(self.net(x.to(device=self.device)),
                                     wx.to(device=self.device),
                                     self.net(y.to(device=self.device)),
                                     wy.to(device=self.device))
                    self._test_scores.append([epoch + 1, (-loss).item()])

    def loss(self,
             x: torch.Tensor, xweights: torch.Tensor,
             y: torch.Tensor, yweights: torch.Tensor):
        koopman = koopman_matrix_weighted(x, xweights, y, yweights)
        if self.loss_method == 'vamp1':
            vamp_score = torch.linalg.norm(koopman, ord='nuc')
        else:
            vamp_score = torch.square(torch.linalg.norm(koopman, ord='fro'))
        return -(1 + vamp_score)


# --------------------------------------------------------------------------- #
# SRV (slow reaction-coordinate / eigenfunction extraction)                   #
# --------------------------------------------------------------------------- #
class SRV:
    """Wrap a trained VAMPNet and solve the final eigenvalue problem.

    The feature network is held in eval mode with autograd disabled.  The final whitening + eigendecomposition is done in ``float64`` (torch) for accuracy; the resulting ``mean`` / ``transform_matrix`` / ``eigvals`` are stored as float64 NumPy arrays.

    Eigenfunctions:
        SRV.__call__(features) -> ndarray (n_samples, num_eigvecs)
        srv_net()              -> torch module mapping features -> CVs (CPU)
    """

    def __init__(self, net, lagtime):
        """Parameters
        ----------
        net : torch.nn.Sequential
            The trained ``VAMPNet.net``.
        lagtime : float
            Lag time (tau), e.g. in ps.
        """
        self.net = net.eval()
        for p in self.net.parameters():
            p.requires_grad = False

        self.lagtime = lagtime
        self.num_eigvecs = self.net[-2].out_features
        self.device = module_device(self.net)
        self.mean = None
        self.transform_matrix = None
        self.eigvals = None

    def timescales(self):
        return -self.lagtime / np.log(self.eigvals)

    # -- internals --------------------------------------------------------- #
    def _transform_features(self, features, batch_size=100_000):
        """Run features through the float32 network, return a float64 CPU tensor.

        Features are transformed one batch of rows at a time, so a memmap-backed (or otherwise large) input is never moved to the device in full: each batch is copied to ``self.device``, passed through the network, and collected back on the CPU as float64.  The concatenated output is ``(n_samples, num_eigvecs)`` -- small even for a large input -- so only the input side needs streaming.

        The result is identical to a single whole-array call: the network is in eval mode (``BatchNorm1d`` uses its stored running statistics; every other layer is pointwise), so processing rows in batches changes no row's output.
        """
        chunks = []
        with torch.no_grad():
            for start in range(0, len(features), batch_size):
                # .copy() materializes a writeable batch from the (read-only) memmap
                # slice, avoiding torch's non-writable-tensor warning -- see the
                # matching note in dataset.py.  Only one batch (~18 MB) at a time.
                batch = features[start : start + batch_size].copy()
                z = self.net(to_torch(batch, device=self.device))
                chunks.append(z.double().cpu())
        return torch.cat(chunks, dim=0)

    def _solve(
        self,
        mean: torch.Tensor,
        c0: torch.Tensor,
        c1: torch.Tensor,
        epsilon: float = EPSILON,
    ):
        """Whitening + symmetric eigenproblem (float64, torch). Stores results."""
        inv_sqrt_c0 = sym_inverse(c0, epsilon=epsilon, return_sqrt=True)
        koopman = inv_sqrt_c0 @ c1 @ inv_sqrt_c0  # symmetric by construction
        eigvals, eigvecs = torch.linalg.eigh(koopman)  # ascending
        eigvals = torch.flip(eigvals, dims=(0,))  # -> descending
        eigvecs = torch.flip(eigvecs, dims=(1,))
        transform_matrix = inv_sqrt_c0 @ eigvecs

        self.mean = mean.reshape(-1).cpu().numpy().astype(np.float64)
        self.eigvals = eigvals.cpu().numpy().astype(np.float64)
        self.transform_matrix = transform_matrix.cpu().numpy().astype(np.float64)

    # -- public ------------------------------------------------------------ #
    def fit(self, dataset, epsilon=EPSILON):
        assert isinstance(dataset, TimeLaggedDataset), f'ERROR with {type(dataset) = }'
        if isinstance(dataset, TrajectoryDataset):
            # x and y are offset views of one trajectory: transform it once,
            # then slice.  Exact because the eval-mode net is row-wise.
            z = self._transform_features(dataset.trajectory)
            lag = dataset.lagframes
            x, y = z[:-lag], z[lag:]
        else:
            x = self._transform_features(dataset.x)
            y = self._transform_features(dataset.y)
        mean, c0, c1 = cov_matrices(x, y)
        self._solve(mean, c0, c1, epsilon=epsilon)
        return self

    def srv_net(self, num_cvs=2):
        """Return a CPU network with the SRV transform appended as a linear layer.

        A deep copy of the feature network is used so that moving the returned
        module to CPU does NOT mutate ``self.net`` (which may live on the GPU).
        """
        W = to_torch(self.transform_matrix[:, :num_cvs])
        b = -to_torch((self.mean @ self.transform_matrix)[:num_cvs])

        # torch convention for a linear layer: y = x W' + b
        eig_layer = nn.Linear(self.num_eigvecs, num_cvs)
        eig_layer.weight = nn.Parameter(W.t())
        eig_layer.bias = nn.Parameter(b)

        feature_net = copy.deepcopy(self.net)  # isolate from self.net
        net = nn.Sequential(*feature_net, eig_layer)
        for p in net.parameters():
            p.requires_grad = False
        net.eval()  # use BatchNorm running stats
        return net.to(device=torch.device("cpu")).float()

    def __call__(self, features):
        z = self._transform_features(features).numpy()
        return (z - self.mean) @ self.transform_matrix


class WeightedSRV(SRV):
    def __init__(self, net, lagtime):
        super().__init__(net, lagtime)

    def fit(self, dataset, epsilon: float = EPSILON):
        assert isinstance(dataset, WeightedTimeLaggedDataset), (
            f'ERROR with {type(dataset) = }'
        )
        x = self._transform_features(dataset.x)
        y = self._transform_features(dataset.y)
        xweights = to_torch(dataset.xweights, device=self.device).double().cpu()
        yweights = to_torch(dataset.yweights, device=self.device).double().cpu()
        mean, c0, c1 = cov_matrices_weighted(x, xweights, y, yweights)
        self._solve(mean, c0, c1, epsilon=epsilon)
        return self
