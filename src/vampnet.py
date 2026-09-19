"""VAMPNet / SRV implementation.

* **No SciPy dependency.**  All linear algebra now goes through ``torch.linalg``.

* **Float precision policy.**  The network runs in ``float32``; promotion to
  ``float64`` happens inside ``cov_matrices`` / ``cov_matrices_weighted``, at
  the boundary where per-frame values become a sum over frames.  That is where
  precision is actually lost: summing ~1e6 float32 terms costs ~1e-5 relative
  error in ``c0``, which the near-singular whitening then amplifies by the
  condition number.  This is the weak link in resolving near-singular
  eigenvalues, not the precision of the 6x6 Koopman matrix into eigh.

* **Rank-aware whitening.**  ``sym_eig`` takes a ``mode`` deciding what happens
  to c0 eigenvalues at or below ``EPSILON``: ``trunc`` discards them (accepting
  rank reduction), ``clamp`` floors them, ``regularize`` reproduces the old
  ridge behaviour.  All three are nan-proof, unlike the original
  ``c0 + epsilon*I`` which could invert a negative eigenvalue.  Training uses
  ``clamp`` (the per-minibatch rank must not jump); the one-off SRV solve uses
  ``trunc`` (no continuity requirement, and truncation caps the whitening
  amplification at 1/sqrt(lambda_min) instead of 1/sqrt(epsilon)).
  ``SRV.rank`` reports how many modes are real.

* **Weighted/unweighted share one implementation.**  ``WeightedVAMPNet``
  overrides only ``_batch_loss`` and ``loss``; ``WeightedSRV`` only
  ``_dataset_class`` and ``_covariances``.  Everything else -- the training
  loop, the transform, the solve -- lives in the base class.
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
    torch_device,
    module_device,
)
from .progress import progress


# Threshold for deciding a covariance direction is unresolvable.  This is a
# *statistical* threshold, not a numerical one: it asks whether a direction
# carries variance above sampling noise, so it does not scale with machine
# epsilon and needs no dtype-dependent tuning.  The network's output layer is
# Tanh, so features are bounded in [-1, 1] and C0's eigenvalues are O(1) at
# most -- which is what makes an absolute threshold interpretable here.
EPSILON = 1e-6


# How sym_eig handles eigenvalues at or below EPSILON.  See sym_eig.
SYM_MODES = ("trunc", "clamp", "regularize")


# VAMP scores of the Koopman matrix K.  The leading 1 in each loss is the
# Perron (equilibrium) eigenvalue, restored after mean-subtraction inside
# cov_matrices removes the trivial mode.
#
#   vamp1: loss = -(1 + sum_i |lambda_i|)   the nuclear norm of K
#   vamp2: loss = -(1 + tr KK')             the squared Frobenius norm
#
# Note vamp1 is the nuclear norm, NOT tr K: the two differ whenever any
# eigenvalue is negative, which is routine with unresolvable modes present.
SUPPORTED_LOSS_METHODS = ("vamp1", "vamp2")


def sym_eig(a: torch.Tensor, epsilon: float = EPSILON, mode: str = "trunc"):
    """Eigendecomposition of a symmetric matrix, with unresolvable directions handled.

    Inherits the dtype/device of ``a``, which is why any identity is built from ``a``.

    ``torch.linalg.eigh`` on a near-singular C0 routinely returns small
    *negative* eigenvalues -- they are not physical, just the numerical residue
    of directions whose true variance is ~0.  Inverting those is what produced
    nan in the old ``a + epsilon*I`` formulation: adding the ridge before
    ``eigh`` gives ``lambda_i + epsilon``, which is still negative whenever
    ``lambda_i < -epsilon``, and ``sqrt(1/negative)`` is nan.  Every mode below
    is nan-proof by construction.

    mode
        'trunc'      discard eigenvalues <= epsilon, accepting rank reduction.
                     The returned eigvecs are ``(n, k)`` with ``k <= n``, so the
                     caller's reconstruction stays ``(n, n)`` but has rank k.
                     Asks the honest question -- "is this direction resolvable?"
                     -- and answers it by dropping the direction rather than
                     inventing a value for it.
        'clamp'      floor eigenvalues at epsilon.  Keeps full rank and a fixed
                     shape; the floored directions are whitened by
                     1/sqrt(epsilon) and so still contribute amplified noise,
                     but continuously and without rank jumps.
        'regularize' the old behaviour made safe: add epsilon*I before eigh,
                     then take absolute values.  Retained for comparison
                     against previous results; not recommended for new work,
                     since flipping the sign of a negative eigenvalue treats
                     numerical residue as though it were signal.

    Returns
    -------
    eigvals : (k,) ascending, all > 0
    eigvecs : (n, k) columns matching eigvals
    """
    if mode not in SYM_MODES:
        raise ValueError(f"Unknown mode {mode!r}; expected one of {SYM_MODES}")

    if mode == "regularize":
        a = a + epsilon * torch.eye(a.shape[0], dtype=a.dtype, device=a.device)

    eigvals, eigvecs = torch.linalg.eigh(a)

    if mode == "trunc":
        keep = eigvals > epsilon
        if not torch.any(keep):
            raise RuntimeError(
                f"No eigenvalue exceeds epsilon={epsilon:g}; the covariance is "
                f"entirely unresolvable (max eigenvalue {eigvals.max():.3e}). "
                f"Check that the network is producing non-constant output."
            )
        eigvals, eigvecs = eigvals[keep], eigvecs[:, keep]
    elif mode == "clamp":
        eigvals = torch.clamp(eigvals, min=epsilon)
    else:
        eigvals = torch.abs(eigvals)

    return eigvals, eigvecs


def sym_inverse(
    a: torch.Tensor,
    epsilon: float = EPSILON,
    return_sqrt: bool = False,
    mode: str = "trunc",
):
    """Inverse (``C^-1``) or inverse square root (``C^-1/2``) of a symmetric PSD matrix.

    Using the eigendecomposition directly (rather than ``inv(sqrtm(.))``) is the
    natural and more stable route for a symmetric positive-semidefinite matrix,
    and it is what lets ``sym_eig`` drop or floor unresolvable directions.

    Under ``mode='trunc'`` the result is the Moore-Penrose pseudo-inverse
    restricted to the resolvable subspace: shape ``(n, n)``, rank k.  Whitening
    with it projects out the null directions instead of amplifying them by
    ``1/sqrt(epsilon)``.
    """
    eigvals, eigvecs = sym_eig(a, epsilon=epsilon, mode=mode)
    if return_sqrt:
        diag = torch.diag(torch.rsqrt(eigvals))
    else:
        diag = torch.diag(torch.reciprocal(eigvals))
    return eigvecs @ diag @ eigvecs.t()


def cov_matrices(x: torch.Tensor, y: torch.Tensor):
    """Unbiased instantaneous/time-lagged covariances, symmetrised over x<->y.

    Inputs are float32.  Promotion to float64 is to avoid float-accumulation error in the covariance matrices.

    Implications for VAMPNet, SRVs:
        * VAMPNet training is done on GPU over minibatches (VRAM ok)
        * SRV fitting is done over CPU, full dataset (RAM ok)
    """
    xmean = torch.mean(x, dim=0, keepdim=True, dtype=torch.float64)
    ymean = torch.mean(y, dim=0, keepdim=True, dtype=torch.float64)
    x = x.double() - xmean
    y = y.double() - ymean

    n = x.shape[0]
    c00 = (x.t() @ x) / (n - 1)
    c11 = (y.t() @ y) / (n - 1)
    c01 = (x.t() @ y) / (n - 1)

    mean = 0.5 * (xmean + ymean)
    c0 = 0.5 * (c00 + c11)  # average x, y variance
    c1 = 0.5 * (c01 + c01.t())  # add reverse transitions
    return mean, c0, c1


def cov_matrices_weighted(
    x: torch.Tensor, xweights: torch.Tensor, y: torch.Tensor, yweights: torch.Tensor
):
    """Weighted covariances using the (biased) weighted estimator.

    The correct unbiased estimator is more involved and described at
    https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Weighted_sample_covariance

    See 'cov_matrices' for reasoning about float32 -> float64
    """

    def wmean(w, a):
        # (N,1) * (N,F) -> sum over samples -> (F,)
        return torch.sum(w.reshape(-1, 1) * a, dim=0, dtype=torch.float64) / torch.sum(
            w
        )

    def wcov(w, a, b):
        # A^T W B / sum(w);  (w * a.t()) is (F,N), @ b is (F,F)
        return (w * a.t()) @ b / torch.sum(w)

    xweights = xweights.double().reshape(-1)
    yweights = yweights.double().reshape(-1)
    xmean = wmean(xweights, x)
    ymean = wmean(yweights, y)
    x = x.double() - xmean
    y = y.double() - ymean

    c00 = wcov(xweights, x, x)
    c01 = wcov(xweights, x, y)
    c11 = wcov(yweights, y, y)

    mean = 0.5 * (xmean + ymean)
    c0 = 0.5 * (c00 + c11)  # average x, y variance
    c1 = 0.5 * (c01 + c01.t())  # add reverse transitions
    return mean, c0, c1


def koopman_matrix(
    x: torch.Tensor, y: torch.Tensor, epsilon: float = EPSILON, mode: str = "clamp"
):
    """Minibatch estimate of the Koopman matrix during training.

    Subtracting the means of x, y projects out the equilibrium eigenfunction.

    ``mode='clamp'`` rather than the ``sym_eig`` default of ``'trunc'``: the
    resolvable-rank mask is recomputed per minibatch, so truncation would let
    the rank change from batch to batch and make the loss jump discontinuously
    whenever a mode crosses the threshold.  Clamping keeps the rank fixed and
    the objective continuous -- floored directions simply receive no gradient.
    """
    _, c0, c1 = cov_matrices(x, y)
    inv_sqrt_c0 = sym_inverse(c0, epsilon=epsilon, return_sqrt=True, mode=mode)
    return inv_sqrt_c0 @ c1 @ inv_sqrt_c0


def koopman_matrix_weighted(
    x: torch.Tensor,
    xweights: torch.Tensor,
    y: torch.Tensor,
    yweights: torch.Tensor,
    epsilon: float = EPSILON,
    mode: str = "clamp",
):
    """Weighted minibatch estimate of the Koopman matrix during training.

    See :func:`koopman_matrix` for why training clamps rather than truncates.
    """
    _, c0, c1 = cov_matrices_weighted(x, xweights, y, yweights)
    inv_sqrt_c0 = sym_inverse(c0, epsilon=epsilon, return_sqrt=True, mode=mode)
    return inv_sqrt_c0 @ c1 @ inv_sqrt_c0


# Factory
def vampnet(p, weighted=True):
    device = torch_device()
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    elif device.type == "cpu":
        torch.set_num_threads(_default_num_threads())
    else:
        raise ValueError(
            f"Unsupported device {device!r}; expected a 'cpu' or 'cuda' device"
        )

    net = nn.Sequential(
        nn.BatchNorm1d(p.num_features),
        nn.Linear(p.num_features, 100),
        nn.ELU(),
        nn.Linear(100, 100),
        nn.ELU(),
        nn.Linear(100, 30),
        nn.ELU(),
        nn.Linear(30, p.num_eigvecs),
        nn.Tanh(),
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
        return len(os.sched_getaffinity(0))  # Linux; respects cpuset/taskset pinning
    except AttributeError:  # macOS/Windows
        return os.cpu_count() or 1


class VAMPNet:
    """Optimize the objective function of the Koopman matrix K.
    See ``SUPPORTED_LOSS_METHODS`` for the two loss definitions.

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
        if loss_method not in SUPPORTED_LOSS_METHODS:
            raise ValueError(
                f"Invalid loss method {loss_method!r}; "
                f"expected one of {SUPPORTED_LOSS_METHODS}"
            )

        self.net = net.to(device=device).float()
        self.device = device
        self.optim = torch.optim.Adam(params=self.net.parameters(), lr=learning_rate)
        self.loss_method = loss_method

        self._train_scores = []
        self._test_scores = []

    @property
    def train_scores(self):
        return np.array(self._train_scores)

    @property
    def test_scores(self):
        return np.array(self._test_scores)

    def fit(self, data_loader_train, data_loader_test, num_epochs=1, usetqdm=True):
        for epoch in progress(
            range(num_epochs),
            usetqdm,
            desc="VAMPnet epoch",
            total=num_epochs,
            leave=False,
        ):
            # training
            self.net.train()
            for batch in data_loader_train:
                self.optim.zero_grad()
                loss = self._batch_loss(batch)
                loss.backward()
                self.optim.step()
                self._train_scores.append([epoch + 1, (-loss).item()])

            # validation
            self.net.eval()
            with torch.no_grad():
                for batch in data_loader_test:
                    loss = self._batch_loss(batch)
                    self._test_scores.append([epoch + 1, (-loss).item()])

    def loss(self, x: torch.Tensor, y: torch.Tensor):
        # the 1 restores the trivial (equilibrium) mode that mean-subtraction
        # inside cov_matrices removes
        return -(1 + self._score(koopman_matrix(x, y)))

    def _batch_loss(self, batch):
        """Unpack one dataloader batch, move it to the device, and score it.

        The only part of the training loop that differs between weighted and
        unweighted, which is why ``fit`` lives entirely in this class.
        """
        x, y = batch
        return self.loss(
            self.net(x.to(device=self.device)),
            self.net(y.to(device=self.device)),
        )

    def _score(self, koopman):
        """VAMP score of an already-built Koopman matrix.

        vamp1 is the nuclear norm sum_i |lambda_i|; vamp2 the squared Frobenius
        norm tr KK'.  Shared by both loss methods -- only the matrix differs.
        """
        if self.loss_method == "vamp1":
            return torch.linalg.norm(koopman, ord="nuc")
        return torch.square(torch.linalg.norm(koopman, ord="fro"))


class WeightedVAMPNet(VAMPNet):

    def _batch_loss(self, batch):
        x, wx, y, wy = batch
        return self.loss(
            self.net(x.to(device=self.device)),
            wx.to(device=self.device),
            self.net(y.to(device=self.device)),
            wy.to(device=self.device),
        )

    def loss(
        self,
        x: torch.Tensor,
        xweights: torch.Tensor,
        y: torch.Tensor,
        yweights: torch.Tensor,
    ):
        return -(1 + self._score(koopman_matrix_weighted(x, xweights, y, yweights)))


class SRV:
    """Wrap a trained VAMPNet and solve the final eigenvalue problem.

    The feature network is held in eval mode with autograd disabled.  Features are transformed in ``float32`` and promoted to ``float64`` inside the covariance functions, so the whitening + eigendecomposition and the stored ``mean`` / ``transform_matrix`` / ``eigvals`` are all float64.

    Eigenfunctions:
        SRV.__call__(features) -> ndarray (n_samples, num_eigvecs)
        srv_net()              -> torch module mapping features -> CVs (CPU)
    """
    # Dataset type accepted by fit(); WeightedSRV narrows it.
    _dataset_class = TimeLaggedDataset

    def __init__(self, net, lagtime):
        """Parameters
        ----------
        net : torch.nn.Sequential
            The trained ``VAMPNet.net``.
        lagtime : float
            The lag time tau (e.g. in ps) -- the familiar MSM/VAC object, NOT
            the frame spacing.  It is ``frametime * lagframes``; see
            :class:`SimulationData.__init__` for the complete definition.

            Consistency requirement: the eigenvalues solved for in :meth:`fit`
            are estimated at a lag of ``dataset.lagframes`` frames (the ``x``/
            ``y`` offset), so for :meth:`timescales` to be correct this
            ``lagtime`` MUST equal ``dataset.lagframes * frametime``.  Passing a
            SimulationData's ``.lagtime`` built from the same ``lagframes``
            satisfies this automatically; supplying the frame spacing here
            instead would under-report every timescale by a factor of lagframes.
        """
        self.net = net.eval()
        for p in self.net.parameters():
            p.requires_grad = False

        self.lagtime = lagtime
        if not isinstance(self.net[-2], nn.Linear):
            raise TypeError(
                f"Expected net[-2] to be the output Linear layer, got "
                f"{type(self.net[-2]).__name__}.  SRV assumes the network ends "
                f"with Linear -> Tanh (or other activation function)."
            )
        self.num_eigvecs = self.net[-2].out_features
        self.device = module_device(self.net)

        self.mean = None
        self.transform_matrix = None
        self.eigvals = None
        self.rank = None  # resolvable rank of c0; < num_eigvecs after truncation

    def timescales(self):
        """Implied (relaxation) timescales t_i from the SRV eigenvalues.

        Each eigenvalue decays over one lag time tau (== ``self.lagtime``) as
        ``lambda_i = exp(-tau / t_i)``, hence

            t_i = -tau / ln(lambda_i).

        The lag here is tau -- the same lag at which the eigenvalues were
        estimated (``dataset.lagframes`` frames apart) -- not the frame spacing;
        see :class:`SimulationData.__init__`.

        The literature form uses ``ln|lambda_i|``.  The absolute value is
        deliberately omitted here so that non-physical eigenvalues (noise modes
        with lambda_i <= 0, or >= 1) surface as nan / negative rather than as
        spurious finite timescales -- a useful "this mode is noise" flag.  Swap
        in ``np.log(np.abs(self.eigvals))`` if you prefer the literature form.
        """
        return -self.lagtime / np.log(self.eigvals)

    def _transform_features(self, features, batch_size=100_000, usetqdm=True):
        """Run features through the float32 network, return a float32 CPU tensor.

        Batched so a memmap-backed (or otherwise large) input is never moved to the device in full.  Exact: the net is in eval mode (``BatchNorm1d`` uses stored running statistics, everything else is pointwise), so batching changes no row's output -- which is what lets ``fit`` transform a trajectory once and slice it.

        Promotion to float64 is deferred to the covariance functions; see the
        module docstring.
        """
        chunks = []
        num_chunks = -(-len(features) // batch_size)  # ceiling division

        with torch.no_grad():
            for start in progress(
                range(0, len(features), batch_size),
                usetqdm,
                total=num_chunks,
                desc="transforming",
            ):
                # torch.tensor always copies, so a read-only memmap slice is
                # fine here -- no intermediate .copy() needed.
                batch = features[start : start + batch_size]
                z = self.net(torch.tensor(batch, dtype=torch.float32, device=self.device))
                chunks.append(z.cpu())

        return torch.cat(chunks, dim=0)

    def _solve(
        self,
        mean: torch.Tensor,
        c0: torch.Tensor,
        c1: torch.Tensor,
        epsilon: float = EPSILON,
        mode: str = "trunc",
    ):
        """Whitening + symmetric eigenproblem (float64, torch). Stores results.

        ``mode='trunc'`` here, unlike the ``'clamp'`` used during training: this
        is a single one-off solve, so there is no continuity requirement, and
        discarding unresolvable directions is preferable to whitening them by
        1/sqrt(epsilon).  Truncation reduces the *rank* of the whitening, not
        its shape, so ``eigvals`` and ``transform_matrix`` keep their full

        ``num_eigvecs`` dimensions; the discarded modes return eigenvalues at the numerical-zero level (~1e-24), which :meth:`timescales` turns into nan (negative) or small positive values (positive) -- neither meaningful. Use :attr:`rank` to tell how many modes are real; do not rely on the timescale values themselves to flag the dead ones.
        """
        inv_sqrt_c0 = sym_inverse(c0, epsilon=epsilon, return_sqrt=True, mode=mode)
        koopman = inv_sqrt_c0 @ c1 @ inv_sqrt_c0  # symmetric by construction
        eigvals, eigvecs = torch.linalg.eigh(koopman)  # ascending
        eigvals = torch.flip(eigvals, dims=(0,))  # -> descending
        eigvecs = torch.flip(eigvecs, dims=(1,))

        # eigh fixes no sign convention, so psi_i can come back negated between
        # otherwise identical fits.  Pin it: force each eigenvector's
        # largest-magnitude component positive.
        imax = torch.argmax(eigvecs.abs(), dim=0)
        signs = torch.sign(eigvecs[imax, torch.arange(eigvecs.shape[1])])
        eigvecs = eigvecs * signs

        transform_matrix = inv_sqrt_c0 @ eigvecs

        self.rank = int((torch.linalg.eigvalsh(c0) > epsilon).sum())
        self.mean = mean.reshape(-1).cpu().numpy()
        self.eigvals = eigvals.cpu().numpy()
        self.transform_matrix = transform_matrix.cpu().numpy()

    def _transform_pairs(self, dataset, usetqdm=True):
        """Transform a dataset's x/y features to float32 CPU tensors.

        For a trajectory-backed dataset (anything carrying ``.trajectory`` and
        ``.lagframes``), x and y are offset views of one array, so the
        trajectory is transformed once and then sliced -- roughly half the
        forward passes.  Exact because the eval-mode net is row-wise; see
        :meth:`_transform_features`.
        """
        if hasattr(dataset, "trajectory"):
            z = self._transform_features(dataset.trajectory, usetqdm=usetqdm)
            return z[: -dataset.lagframes], z[dataset.lagframes :]
        return (
            self._transform_features(dataset.x, usetqdm=usetqdm),
            self._transform_features(dataset.y, usetqdm=usetqdm),
        )

    def _covariances(self, dataset, x, y):
        """Covariances for this estimator; WeightedSRV adds the weights."""
        return cov_matrices(x, y)

    def fit(self, dataset, epsilon=EPSILON, mode="trunc", usetqdm=True):
        if not isinstance(dataset, self._dataset_class):
            raise TypeError(
                f"dataset must be a {self._dataset_class.__name__} (or its "
                f"trajectory subclass), got {type(dataset).__name__}"
            )
        x, y = self._transform_pairs(dataset, usetqdm=usetqdm)
        mean, c0, c1 = self._covariances(dataset, x, y)
        self._solve(mean, c0, c1, epsilon=epsilon, mode=mode)
        return self

    def srv_net(self, num_cvs=2):
        """Return a CPU network with the SRV transform appended as a linear layer.

        A deep copy of the feature network is used so that moving the returned
        module to CPU does NOT mutate ``self.net`` (which may live on the GPU).
        """
        if num_cvs < 1:
            raise ValueError(f"num_cvs must be at least 1, got {num_cvs}")

        if self.transform_matrix is None:
            raise RuntimeError("srv_net() requires a fitted SRV; call fit() first.")

        if self.rank is not None and num_cvs > self.rank:
            raise ValueError(
                f"num_cvs={num_cvs} exceeds the resolvable rank of c0 "
                f"({self.rank}).  CVs beyond the rank are numerical noise "
                f"(~1e-24); biasing along them is meaningless."
            )

        # drop back to float32
        W = torch.tensor(self.transform_matrix[:, :num_cvs], dtype=torch.float32)
        b = -torch.tensor((self.mean @ self.transform_matrix)[:num_cvs], dtype=torch.float32)

        # torch convention for a linear layer: y = x W' + b
        eig_layer = nn.Linear(self.num_eigvecs, num_cvs)
        # torch deals with transpose operation by swapping stride.  Contiguous call makes a copy laid out with expected shape, (num_cvs, num_eigvecs)
        eig_layer.weight = nn.Parameter(W.t().contiguous())
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
    _dataset_class = WeightedTimeLaggedDataset

    def _covariances(self, dataset, x, y):
        # No dtype: preserve the arrays' float64.  Narrowing reweighting factors
        # to float32 quantises them by ~6e-8 relative, which propagates to ~2e-7
        # in c0 -- the same order as the accumulation error the float64 policy
        # exists to remove.
        xweights = torch.tensor(dataset.xweights, dtype=None)
        yweights = torch.tensor(dataset.yweights, dtype=None)
        return cov_matrices_weighted(x, xweights, y, yweights)
