import numpy as np
import torch
from torch.utils.data import (
    DataLoader,
    random_split
)


def data_loaders(dataset, frac_test, batch_size=10_000,
                 num_workers=0, prefetch_factor=None):
    """Return DataLoaders for training, validation.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        User-defined types defined below.
    frac_test : float
        Fraction of dataset to be used for testing (~0.1)
    batch_size : int
        Size of training minibatches
    num_workers : int
        Number of subprocesses for the training loader's data loading.  ``0``
        loads in the main process (synchronously, between GPU steps).  With
        memmap-backed features, a small positive value (e.g. 2) lets workers
        issue concurrent random reads, raising NVMe queue depth and hiding
        first-touch latency behind compute.  Workers share the read-only
        memmap through the OS page cache, so they add process overhead
        (~150-250 MB each) rather than duplicating the array.  The validation
        loader is left at 0 (single batch, nothing to overlap).
    prefetch_factor : int or None
        Batches pre-loaded per worker (total prefetched = ``num_workers *
        prefetch_factor``).  Ignored, and must be ``None``, when
        ``num_workers`` is 0.
    """
    num_transitions = len(dataset)
    num_test = round(num_transitions * frac_test)
    num_train = num_transitions - num_test
    data_train, data_test = random_split(dataset, [num_train, num_test])

    train_loader = DataLoader(
        data_train, batch_size=batch_size, shuffle=True,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )
    test_loader = DataLoader(
        data_test, batch_size=len(data_test), shuffle=False
    )
    return train_loader, test_loader


class TimeLaggedDataset(torch.utils.data.Dataset):
    """Data defined as a set of transitions {x_i, y_i}.  They do not have to appear in any particular order.  The lag time is defined elsewhere.
    """
    def __init__(self, x, y):
        """Parameters
        ---------
        x : ndarray with shape (num_frames - lagframes, num_features)
        y : ndarray with shape (num_frames - lagframes, num_features)
        """
        assert len(x) == len(y), (
            f"Length mistmatch: {len(x)=} != {len(y)=}"
        )
        self.x = x
        self.y = y

    def astype(self, dtype):
        """Return a copy cast to *dtype*.

        Reads ``x``/``y`` fully into RAM, so calling this on a memmap-backed
        dataset defeats the memory-mapping.  With data already stored as
        float32 on disk this is normally unnecessary.
        """
        return TimeLaggedDataset(self.x.astype(dtype), self.y.astype(dtype))

    def __getitem__(self, item):
        # .copy() returns writeable arrays.  Rows of a memmap-backed x/y are
        # read-only views, which makes torch.from_numpy (called by the default
        # collate) warn about non-writable tensors; copying the single row here
        # avoids that and is cheap for in-RAM arrays too.
        return self.x[item].copy(), self.y[item].copy()

    def __getitems__(self, indices):
        # Batched fetch: DataLoader calls this once per batch (with the whole
        # index list) instead of __getitem__ once per index, when present.
        # A single fancy-index read is dramatically faster than the equivalent
        # Python-level loop over __getitem__ -- ~50x in local testing -- since
        # it replaces thousands of individual calls (each a separate random
        # access when x/y are memmap-backed) with one vectorized read. Fancy
        # indexing also returns an independent, writeable array on its own,
        # so no .copy() is needed here.
        x, y = self.x[indices], self.y[indices]
        return list(zip(x, y))

    def __len__(self):
        return len(self.x)

    def __add__(self, other):
        # np.vstack materializes both operands fully in RAM; to combine many
        # per-step runs without that peak, concatenate on disk with
        # gather_feature_data instead.
        x = np.vstack((self.x, other.x))
        y = np.vstack((self.y, other.y))
        return TimeLaggedDataset(x, y)


class TrajectoryDataset(TimeLaggedDataset):
    """Data defined as a ordered sequence of states {x_t}.  The lag time is defined elsewhere.
    """
    def __init__(self, trajectory, lagframes=1):
        """Parameters
        ---------
        trajectory : ndarray with shape (num_frames, num_features)
            Feature data from simulation
        lagframes : int
            Number of simulation frames separating transition
        """
        assert lagframes > 0, (
            "lagframes must be positive"
        )
        assert len(trajectory) > lagframes, (
            "Not enough data to for lagtime"
        )
        self.lagframes = lagframes
        self.trajectory = trajectory
        super().__init__(trajectory[:-lagframes], trajectory[lagframes:])


class WeightedTimeLaggedDataset(torch.utils.data.Dataset):
    """As TimeLaggedDataset, but each state in {x_i, y_i} has an associated weights.
    """
    def __init__(self, x, xweights, y, yweights):
        """Parameters
        ---------
        x, y : arrays with shape (num_frames - lagframes, num_features)
        xweights, yweights : arrays with shape (num_frames - lagframes,)
        """
        assert x.shape == y.shape, (
            f'Shape mismatch: {x.shape} != {y.shape}'
        )
        assert len(xweights) == len(x), (
            f'Size mismatch: {len(xweights)} != {len(x)}'
        )
        assert len(yweights) == len(x), (
            f'Size mismatch: {len(yweights)} != {len(x)}'
        )
        self.x = x
        self.xweights = xweights
        self.y = y
        self.yweights = yweights

    def astype(self, dtype):
        return WeightedTimeLaggedDataset(self.x.astype(dtype),
                                         self.xweights.astype(dtype),
                                         self.y.astype(dtype),
                                         self.yweights.astype(dtype))

    def __getitem__(self, item):
        # Copy the (2-D) feature rows so memmap-backed data is writeable; the
        # weights come back as scalar value-copies from integer indexing and
        # need no copy.  See TimeLaggedDataset.__getitem__.
        return (self.x[item].copy(), self.xweights[item],
                self.y[item].copy(), self.yweights[item])

    def __getitems__(self, indices):
        # Batched fetch -- see TimeLaggedDataset.__getitems__ for why this
        # matters for memmap-backed x/y (avoids one Python call + one random
        # access per sample in favor of a single vectorized read per batch).
        x, xw = self.x[indices], self.xweights[indices]
        y, yw = self.y[indices], self.yweights[indices]
        return list(zip(x, xw, y, yw))

    def __len__(self):
        return len(self.x)

    def __add__(self, other):
        x = np.vstack((self.x, other.x))
        xweights = np.hstack([self.xweights, other.xweights])
        y = np.vstack((self.y, other.y))
        yweights = np.hstack([self.yweights, other.yweights])
        return WeightedTimeLaggedDataset(x, xweights, y, yweights)


class WeightedTrajectoryDataset(WeightedTimeLaggedDataset):
    """As TrajectoryDataset, but with weights.  Implementing __add__ is a bad idea.  Use TimeLaggedDataset for this.
    """
    def __init__(self, trajectory, weights, lagframes=1):
        """Parameters
        ---------
        trajectory : ndarray with shape (num_frames, num_features)
            Feature data from simulation
        weights : ndarray with shape (num_frames,)
            Weights corresponding to each frame
        lagframes : int
            Number of simulation (.h5) frames separating transition
        """
        assert lagframes > 0, (
            'lagframes must be positive'
        )
        assert len(trajectory) > lagframes, (
            'Not enough data to for lagtime'
        )
        assert len(weights) == len(trajectory), (
            f'Length mismatch: {len(weights)} != {len(trajectory)}'
        )
        self.lagframes = lagframes
        self.weights = weights
        self.trajectory = trajectory

        super().__init__(trajectory[:-lagframes], weights[:-lagframes],
                         trajectory[lagframes:], weights[lagframes:])
