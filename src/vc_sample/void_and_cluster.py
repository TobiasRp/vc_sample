import numpy as np


def mask_rho_samples(rho, is_sample):
    return rho if is_sample else -np.inf


def mask_rho_points(rho, is_sample):
    return rho if not is_sample else np.inf


mask_rho_samples_func = np.vectorize(mask_rho_samples)
mask_rho_points_func = np.vectorize(mask_rho_points)


class SampleDensity:
    def __init__(
        self, num_points: int, initial_sample_indices: np.array, density_estimator
    ):
        self.density_estimator = density_estimator

        self.is_sample = np.zeros(num_points, dtype=np.bool)
        self.is_sample[initial_sample_indices] = True
        self.rho = self.density_estimator.estimate(self.is_sample)

    def add_largest_void(self) -> int:
        """
        Find and add largest void that is NOT a sample
        """
        largest_void_idx = np.argmin(mask_rho_points_func(self.rho, self.is_sample))

        self.density_estimator.add(self.rho, largest_void_idx)

        assert not self.is_sample[largest_void_idx]
        self.is_sample[largest_void_idx] = True

        return largest_void_idx

    def remove_tightest_cluster(self) -> int:
        """
        Find and remove tightest cluster that IS a sample
        """
        tightest_clust_idx = np.argmax(mask_rho_samples_func(self.rho, self.is_sample))

        self.density_estimator.sub(self.rho, tightest_clust_idx)

        assert self.is_sample[tightest_clust_idx]
        self.is_sample[tightest_clust_idx] = False

        return tightest_clust_idx

    def num_samples(self) -> int:
        return np.sum(self.is_sample)

    @property
    def sample_indices(self) -> np.array:
        return np.where(self.is_sample)


class VoidAndCluster:
    """Performs void-and-cluster sampling to find optimally stratified samples."""

    def __init__(
        self,
        points: np.array,
        density_estimator,
        num_initial_samples: int = 100,
        log_fn=None,
    ):
        self.num_initial_samples = num_initial_samples
        self.num_points = points.shape[0]
        self.points = points

        self.is_sample = np.zeros(self.num_points, dtype=np.bool)

        rank_type = np.int32
        self.rank = np.empty(self.num_points, dtype=rank_type)

        self.INF_RANK = np.iinfo(rank_type).max
        self.rank.fill(self.INF_RANK)

        self.density_estimator = density_estimator
        self.inv_rho_p = 1.0 / density_estimator.estimate()

        self.log_fn = log_fn

    def _initial_sampling(self, num_samples) -> SampleDensity:
        """
        Performs initial (simple) random sampling.
        """
        indices = np.random.choice(
            range(0, self.is_sample.shape[0]), size=num_samples, replace=False
        )
        self.rank[indices] = range(0, num_samples)
        return SampleDensity(self.num_points, indices, self.density_estimator)

    def _swap_rank(self, largest_void, tightest_cluster):
        """
        Swaps the ranks of two indices.
        """
        r_lv = self.rank[largest_void]
        self.rank[largest_void] = self.rank[tightest_cluster]
        self.rank[tightest_cluster] = r_lv

    def _initial_optimization(self, sample_density: SampleDensity):
        """
        Optimizes the samples by finding and exchanging the largest void and tightest cluster
        """
        while True:
            largest_void = sample_density.add_largest_void()
            tightest_cluster = sample_density.remove_tightest_cluster()

            if self.log_fn:
                self.log_fn(
                    "Initial optimization",
                    sample_density.rho,
                    sample_density.sample_indices,
                    largest_void,
                    tightest_cluster,
                )

            self._swap_rank(largest_void, tightest_cluster)

            if largest_void == tightest_cluster:
                break

    def _fill_voids(self, sample_density: SampleDensity, num: int):
        """
        Adds ``num`` samples by iteratively finding and adding the largest void.
        Args:
            sample_density: Density of samples to find the largest void(s)
            num: Number of samples to add
        """
        for i in range(num):
            largest_void = sample_density.add_largest_void()

            assert self.rank[largest_void] == self.INF_RANK
            self.rank[largest_void] = self.num_initial_samples + i

            if self.log_fn:
                self.log_fn(
                    "Void filling",
                    sample_density.rho,
                    sample_density.sample_indices,
                    largest_void,
                    -1,
                )

    def sample(self, size: int):
        """
        Returns ``size`` optimally stratified samples.
        """
        num_initial_samples = (
            self.num_initial_samples if self.num_initial_samples < size else size
        )
        sample_density = self._initial_sampling(num_initial_samples)

        self._initial_optimization(sample_density)
        assert sample_density.num_samples() == num_initial_samples

        self._fill_voids(sample_density, size - self.num_initial_samples)
        assert sample_density.num_samples() == size

        ord = self.ordering(size)
        return self.points[ord]

    def ordering(self, size: int):
        """
        After sampling, this returns a prefix of ``size`` sample indices.
        Args:
            size: Size of the prefix. Cannot be greater than what has previously been sampled.
        Returns:
            Indices of the first ``size`` samples.
        """
        return np.argsort(self.rank)[:size]
