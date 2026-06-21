import numpy as np
import multiprocessing as mp

from kernel import gaussian2d_grid_over_patch


class BiasShiftExactGrid:
    """Do efficient computation of exact bias shift, computed over a grid with the PDF from a long, unbiased simulation.  It progresses by adding the Gaussians of a metadynamics simulation one-by-one.

    The integrand is stored on a grid:

        igrid = p * np.exp(-v / kT)

    and update using the metadynamics bias potential update:

        V_t(s) = V_{t-1}(s) + G_t(s)

    Save computation by updating only a patch of V_t by a number of widths of the Gaussian being added.  In fact it is the 'bias factor' integrad which is being updated and the change in the bias shift depends only upon the change in the sum of the integrand stored on a grid.

    The main method is:
        bias_shift.add_gaussian(h, c, w)
    which adds Gaussian to V_{t-1}(s) by updating the integrand, igrid and adds the new bias shift value to the list, self._ct
    """
    def __init__(self, x, y, p, kT, nsigma=4):
        """Parameters
        ----------
        x : array, shape (nx,)
        y : array, shape (ny,)
        p : array, shape (nx, ny)
            The PDF for the unbiased ensemble, p(s) or p_0(s)
        kT : float
            Thermal energy in kJ/mol
        nsigma : float
            How many widths away from Gaussian centers do we extend patch?
        """
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.da = (x[1] - x[0]) * (y[1] - y[0])
        self.kT = kT
        self.nsigma = nsigma

        self.igrid = np.array(p, copy=True) # integrand on grid with v = 0
        self.isum = np.sum(self.igrid)      # must keep track of running sum
        self._ct = [0.]

    @property
    def ct(self):
        return np.array(self._ct)

    def add_gaussian(self, height, center, width):
        """Add a single Gaussian to the metadynamics potential and compute the new value of the bias factor.  Append it to list self._ct.

        Track sum over integrand on grid and update only patch, P:
            S_t = S_{t-1} + sum_{i,j in P} (I_t[i,j] - I_{t-1}[i,j])

            * store I_t in igrid, S_t in isum.
            * update igrid with exp(-beta G_t(s)) over patch

        Parameters
        ----------
        height, center, width : Gaussian to add
        """
        xslice, yslice, patch = self.gaussian_grid_patch(height, center, width)

        old = self.igrid[xslice, yslice]       # store igrid patch
        new = old * np.exp(-patch / self.kT)   # updated igrid patch
        self.isum += np.sum(new - old)         # update integrand sum
        self.igrid[xslice, yslice] = new       # update integrand grid
        self._ct.append(                       # update list of c_t
            -self.kT * np.log(self.da * self.isum)
        )

    def gaussian_grid_patch(self, height, center, width):
        """Returns
        -------
        xslice : slice over x-index for Gaussian patch
        yslice : slice over y-index for Gaussian patch
        patch : array, shape (nx_patch, ny_patch)

        Delegates to kernel.gaussian2d_grid_over_patch (single-Gaussian math
        now lives in kernel.py).
        """
        return gaussian2d_grid_over_patch(
            self.x, self.y, height, center, width, self.nsigma
        )


class BiasShiftExactMC:
    """Efficient computation of exact bias shift using a Monte Carlo dataset.

    The expectation is approximated as

        <exp(-V_t / kT)> ~= (1/N) * sum_k exp(-V_t(s_k) / kT)

    where s_k are sampled from the unbiased ensemble.

    We store the integrand values over the dataset:

        eweights[k] = exp(-V_t(s_k) / kT)

    and update incrementally when a new Gaussian G_t is added:

        V_t(s_k) = V_{t-1}(s_k) + G_t(s_k)

    so that

        eweights_new[k] = eweights_old[k] * exp(-G_t(s_k) / kT)

    To save computation, each Gaussian is evaluated only on points within an
    nsigma-width rectangular cutoff around its center.
    """
    def __init__(self, points, kT, nsigma=4.0):
        """Parameters
        ----------
        points : array, shape (N, 2)
            Monte Carlo dataset of CV points.
        kT : float
            Thermal energy.
        nsigma : float, optional
            Cutoff in units of Gaussian width, by default 4.0
        """
        self.points = np.asarray(points)
        if self.points.ndim != 2 or self.points.shape[1] != 2:
            raise ValueError(f"points must have shape (N, 2). Got {self.points.shape}")

        self.kT = float(kT)
        self.nsigma = float(nsigma)
        self.N = len(self.points)

        self.eweights = np.ones(self.N, dtype=float)
        self.mean_eweights = 1.0
        self._ct = [0.0]

    @property
    def ct(self):
        return np.array(self._ct)

    def add_gaussian(self, height, center, width):
        """Add one Gaussian and append the new bias shift.

        Parameters
        ----------
        height : float
        center : array, shape (2,)
        width : array, shape (2,)

        Returns
        -------
        ct_new : float
            Updated bias shift
        """
        mask, gvals = self.gaussian_points_patch(height, center, width)
        return self._apply_patch(mask, gvals)

    def _apply_patch(self, mask, gvals):
        """Apply a precomputed MC patch in the main process."""
        if np.any(mask):
            old = self.eweights[mask]
            new = old * np.exp(-gvals / self.kT)
            self.mean_eweights += np.sum(new - old) / self.N
            self.eweights[mask] = new

        ct_new = -self.kT * np.log(self.mean_eweights)
        self._ct.append(ct_new)
        return ct_new

    def gaussian_points_patch(self, height, center, width):
        """Evaluate one Gaussian only on dataset points inside the nsigma cutoff.

        Parameters
        ----------
        height : float
        center : array, shape (2,)
        width : array, shape (2,)

        Returns
        -------
        mask : array, shape (N,), dtype=bool
            True where the point is inside the rectangular cutoff
        gvals : array, shape (num_selected,)
            Gaussian values on the selected points
        """
        center = np.asarray(center)
        width = np.asarray(width)

        dx = np.abs(self.points[:, 0] - center[0])
        dy = np.abs(self.points[:, 1] - center[1])

        mask = (dx <= self.nsigma * width[0]) & (dy <= self.nsigma * width[1])

        if not np.any(mask):
            return mask, np.empty(0, dtype=float)

        selected = self.points[mask]
        gvals = self.gaussian_points(selected, height, center, width)
        return mask, gvals

    @staticmethod
    def gaussian_points(points, height, center, width):
        """Evaluate a single Gaussian over a set of points.

        Parameters
        ----------
        points : array, shape (num_points, 2)
        height : float
        center : array, shape (2,)
        width : array, shape (2,)
            cov = diag(width**2)

        Returns
        -------
        gvals : array, shape (num_points,)
        """
        z = (points - center[None, :]) / width[None, :]
        return height * np.exp(-0.5 * np.sum(z * z, axis=1))

    def add_gaussians_parallel(self, gaussians, processes=None, chunksize=1):
        """Compute Gaussian patches in worker processes, apply sequentially."""
        if processes is None:
            processes = mp.cpu_count()

        tasks = [(self.points, self.nsigma, h, c, w) for h, c, w in gaussians]

        with mp.Pool(processes=processes) as pool:
            for mask, gvals in pool.imap(
                _mc_gaussian_patch_worker, tasks, chunksize=chunksize
            ):
                self._apply_patch(mask, gvals)


def _mc_gaussian_patch_worker(args):
    """Worker: compute mask and Gaussian values for one Gaussian."""
    points, nsigma, height, center, width = args

    center = np.asarray(center)
    width = np.asarray(width)

    dx = np.abs(points[:, 0] - center[0])
    dy = np.abs(points[:, 1] - center[1])

    mask = (dx <= nsigma * width[0]) & (dy <= nsigma * width[1])

    if not np.any(mask):
        return mask, np.empty(0, dtype=float)

    selected = points[mask]
    z = (selected - center[None, :]) / width[None, :]
    gvals = height * np.exp(-0.5 * np.sum(z * z, axis=1))
    return mask, gvals














class BiasShiftExactMCSerial:
    def __init__(self, points, kT, nsigma=4.0):

        self.points = np.asarray(points)
        if self.points.ndim != 2 or self.points.shape[1] != 2:
            raise ValueError(f"points must have shape (N, 2). Got {self.points.shape}")

        self.kT = float(kT)
        self.nsigma = float(nsigma)
        self.N = len(self.points)

        # Store exp(-V/kT); initially V=0 so this is all ones
        self.eweights = np.ones(self.N, dtype=float)

        # Running mean of eweights
        self.mean_eweights = 1.0

        # Bias shift history
        self._ct = [0.0]

    @property
    def ct(self):
        return np.array(self._ct)

    def add_gaussian(self, height, center, width):
        mask, gvals = self.gaussian_points_patch(height, center, width)

        if np.any(mask):
            old = self.eweights[mask]
            new = old * np.exp(-gvals / self.kT)

            # Update running mean efficiently:
            # mean_new = mean_old + (sum(new-old) / N)
            self.mean_eweights += np.sum(new - old) / self.N
            self.eweights[mask] = new

        ct_new = -self.kT * np.log(self.mean_eweights)
        self._ct.append(ct_new)
        return ct_new

    def gaussian_points_patch(self, height, center, width):
        center = np.asarray(center)
        width = np.asarray(width)

        dx = np.abs(self.points[:, 0] - center[0])
        dy = np.abs(self.points[:, 1] - center[1])

        mask = (dx <= self.nsigma * width[0]) & (dy <= self.nsigma * width[1])

        if not np.any(mask):
            return mask, np.empty(0, dtype=float)

        selected = self.points[mask]
        gvals = self.gaussian_points(selected, height, center, width)
        return mask, gvals

    @staticmethod
    def gaussian_points(points, height, center, width):
        z = (points - center[None, :]) / width[None, :]
        return height * np.exp(-0.5 * np.sum(z * z, axis=1))
