import numpy as np


class VoidAndCluster:
    """Performs void-and-cluster sampling to find optimally stratified samples."""

    def __init__(self, points: np.array, estimate_density_fn, num_initial_samples: int = 100):
        self.num_initial_samples = num_initial_samples
        self.num_points = points.shape[0]
        self.points = points

        self.is_sample = np.zeros(self.num_points, dtype=np.bool)
        self.rank = np.empty(self.num_points, dtype=np.int)
        self.rank = np.inf

        self.estimate_density = estimate_density_fn
        self.inv_rho_p = 1.0 / estimate_density_fn(points)

    def _initial_sampling(self):
        """
        Performs initial (simple) random sampling.
        """
        indices = np.random.choice(range(0, self.is_sample.shape[0]), size=self.num_initial_samples)
        self.is_sample[indices] = True
        self.rank[indices] = range(0, self.num_initial_samples - 1)

    def _swap_rank(self, largest_void, tightest_cluster):
        """
        Swaps the ranks of two indices.
        """
        r_lv = self.rank[largest_void]
        self.rank[largest_void] = self.rank[tightest_cluster]
        self.rank[tightest_cluster] = r_lv

    def _add_largest_void(self, rho_s: np.array) -> int:
        # TODO: Find largest void
        # largest_void = np.argmin(rho)

        # TODO: Update sample density rho_s
        # TODO: Set self.is_sample
        # TODO: Return index
        return 0

    def _remove_tightest_cluster(self, rho_s: np.array) -> int:
        # TODO: tightest_cluster
        # tightest_cluster = np.argmax(rho)

        # TODO: Update sample density rho_s
        # TODO: Set self.is_sample
        # TODO: Return index
        return 0

    def _initial_optimization(self, rho_s):
        """
        Optimizes the samples by finding and exchanging the largest void and tightest cluster
        Args:
            rho_s: Density of samples
        """
        while True:
            largest_void = self._add_largest_void(rho_s)
            tightest_cluster = self._remove_tightest_cluster(rho_s)

            self._swap_rank(largest_void, tightest_cluster)

            if largest_void == tightest_cluster:
                break

    def _fill_voids(self, rho_s, num: int):
        """
        Adds ``num`` samples by iteratively finding and adding the largest void.
        Args:
            rho_s: Density of samples, will be updated
            num: Number of samples to add
        """
        assert num > 0

        for i in range(self.num_initial_samples, num):
            largest_void = self._add_largest_void(rho_s)
            self.rank[largest_void] = i

    def sample(self, size: int):
        """
        Returns ``size`` optimally stratified samples.
        Args:
            size:

        Returns:

        """
        self._initial_sampling()

        rho_s = self.estimate_density(self.points, self.is_sample)

        self._initial_optimization(rho_s)

        samples_left = size - self.num_initial_samples
        if samples_left > 0:
            self._fill_voids(rho_s, samples_left)

        return self.points[self.is_sample][:size]

    def get_ordering(self, size: int):
        """
        After sampling, this returns a prefix of ``size`` sample indices.
        Args:
            size: Size of the prefix. Cannot be greater than what has previously been sampled.
        Returns:
            Indices of the first ``size`` samples.
        """
        return self.rank[:size]
