import numpy as np


class KDE:
    """Parrinellow's KDE representation of p(s).  See
    Rethinking Metadynamics: From Bias Potentials to Probability Distributions, J. Phys. Chem. Lett. 2020, 11, 2731

    NB:  p(s) is represented by sums of Gaussians.  When these KDEs are then represented as grids over (s1, s2), they will have analytical norms ~ 1, but not numberical.  This is especially the case where Silverman's rule of thumb is used since this gives large widths for eigenfunction data.

    Therefore, be careful with functions that check numerical norms (hist2d.grid_norm)
    """
    def __init__(self, centers, widths, heights, wsum=None):
        self.centers = centers
        self.widths = widths
        self.heights = heights
        self.wsum = wsum

    @classmethod
    def from_npz(cls, file):
        assert file.suffix == '.npz', (
            f'Incorrect file extension {file.suffix} must be .npz'
        )
        a = np.load(file)
        return cls(centers=a['centers'], widths=a['widths'],
                   heights=a['heights'], wsum=float(a['wsum']))

    def savez(self, file):
        np.savez(file, centers=self.centers, widths=self.widths,
                 heights=self.heights, wsum=self.wsum)

    def save_pickle(self, file):
        save_pickle(file, self)

    def __len__(self):
        """Just in case we encounter an "empty" instance
        """
        if self.heights is None:
            return 0
        else:
            return len(self.heights)

    def __call__(self, s):
        """Return P(s) = sum_k G(s - s_k) where s must be a single point.

        Parameters:
        -----------
        s : array, shape (num_cvs,)
        """
        assert s.shape == self.centers[0].shape, (
            f'ERROR: using kde.__call__ with {s.shape = }'
        )
        norm_sqs = np.sum(np.square((s - self.centers) / self.widths), axis=1)
        return np.sum(self.heights * np.exp(-0.5 * norm_sqs))

    def __eq__(self, other):
        return np.allclose(self.centers, other.centers) \
            and np.allclose(self.widths,  other.widths) \
            and np.allclose(self.heights,  other.heights) \
            and np.isclose(self.wsum, other.wsum)

    def __add__(self, other):
        """This is worked out in test-5k-kde-adding.ipynb.  See notes there.

        Algorithm, starting with P_n(s)
        -------------------------------
        1. Generate set of new frames and weights
        2. Create a KDE object (delta P_n(s)) for new frames, ignoring P_n(s)
        3. [Start __add__.]  Compute scaling factors for the heights of each KDE
        4. Concatenate data and return KDE for P_{n+1}(s)
        """
        # if not isinstance(other, KDE):  #  will this ever happen??
        #     return NotImplemented
        self_is_addable = (self.wsum is not None)
        other_is_addable = (other.wsum is not None)
        assert self_is_addable and other_is_addable, (
            "Cannot add KDE objects unless they both have wsums not None"
        )
        wsum = self.wsum + other.wsum
        sheights = self.wsum / wsum * self.heights
        oheights = other.wsum / wsum * other.heights

        return KDE(centers=np.vstack([self.centers, other.centers]),
                   widths=np.vstack([self.widths, other.widths]),
                   heights=np.concatenate([sheights, oheights]),
                   wsum=wsum)

    def __radd__(self, other):
        """Support:
            None + KDE = KDE      (used when iterating over SimulationFrames)
            0 + KDE = KDE         (enables sum())

        * when x.__add__(y) returns NotImplemented, Python will then attempt to call y.__radd__(x)
        * the isinstance(other, KDE) line will never occur (self.__add__ will be used in that case.)  But it's ok as addition is cumutative.
        """
        if other is None or other == 0:
            return self
        if isinstance(other, KDE):
            return self.__add__(other)
        return NotImplemented

    def renormalize(self):
        """TODO: figure out wheter returning a new KDE is preferable to the fluent-style return self (with its side effects)
        """
        self.heights /= self.norm()
        return self

    def norm(self):
        """Return the analytical norm for the full KDE.
        """
        return (2 * np.pi) ** (len(self.centers[0]) / 2) * sum([
            h * w.prod() for h, w in zip(self.heights, self.widths)
        ])

    def norms_kernels(self):
        """Return an array containing the analytical norm for each kernel.
        """
        return np.array([
            (2 * np.pi) ** (len(self.centers[0]) / 2) * h * w.prod()
            for h, w in zip(self.heights, self.widths)
        ])

    def norm_factor_mc(self):
        """Return the Monte Carlo estimate for Parrinello's norm facctor using
            2020-parrinello-opes-si Eq. (S9), p. S3.
        """
        return np.mean([self(s) for s in self.centers])

    def norm_factor_quad(self, x, y, hist):
        """Return Parrinello's norm factor computed by quadrature over grid defined by x, y.

            Z_n = (1 / N_c) * sum_{ij | hist(s_ij) > 0} KDE(x_i, y_j)

        Parameters:
        -----------
        x, y : arrays with shape (num_pointsx,), (num_pointsy,)
        hist : array, shape (num_pointsx, num_pointsy)
            2d histogram computed from trajectory data frames.
            Simpleer to compute this outside this function and pass in?
        """
        ix, iy = np.where(hist > 0)
        return np.mean([
            self(np.array([x[i], y[j]])) for i, j in zip(ix, iy)
        ])

    def compressed(self, dist_threshold=1, loud=True):
        """Return a compressed KDE object.

        Reason for tqdm: with dist_threshold=1,
            350,000 kernels, widths=(0.02, 0.005)  => 30 minutes
            350,000 kernels, widths=(0.005, 0.005) => 100 minutes

        Reference: Supplemantary Information for Michele Invernizzi, Pablo M. Piaggi, and Michele Parrinello, "Unified Approach to Enhanced Sampling", Phys. Rev. X 10, 041034 (2020), https://journals.aps.org/prx/abstract/10.1103/PhysRevX.10.041034
        """
        gaussians = []
        gauss_params = zip(self.heights, self.centers, self.widths)

        if loud:
            gauss_params = tqdm(gauss_params)

        for h, c, w in gauss_params:
            gn = Gaussian(h, c, w)
            keep_merging = True

            while keep_merging:
                if len(gaussians) == 0:
                    gaussians.append(gn)
                    keep_merging = False
                else:
                    dists = [g.distance(gn.c) for g in gaussians]
                    idx = np.argmin(dists)
                    if dists[idx] > dist_threshold:
                        gaussians.append(gn)
                        keep_merging = False
                    else:
                        gn += gaussians[idx]
                        del gaussians[idx]

        return KDE(centers=np.vstack([g.c for g in gaussians]),
                   widths=np.vstack([g.w for g in gaussians]),
                   heights=np.stack([g.h for g in gaussians]),
                   wsum=self.wsum)



