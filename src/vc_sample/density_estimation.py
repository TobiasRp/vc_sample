import numpy as np
from scipy.spatial import cKDTree


class EpanechnikovKernel:
    """
    The Epanechnikov
    """

    def __call__(self, x: int):
        """
        Evaluates kernel
        """
        if 0 <= x < self.support():
            return 0.75 * (1 - x ** 2)
        else:
            return 0.0

    @staticmethod
    def support():
        """
        The kernel has a support of [0,1).
        """
        return 1.0


class KernelDensityEstimator:
    def __init__(self, points, kernel=EpanechnikovKernel()):
        self.kernel = kernel
        self.points = points
        self.tree = cKDTree(points)

    def estimate(self, mask=None):
        rho = np.zeros(self.points.shape[0], dtype=np.float)

        p_indices = [
            idx for idx in range(self.points.shape[0]) if mask is None or mask[idx]
        ]
        if len(p_indices) == 0:
            return rho

        neighbor_lists = self.tree.query_ball_point(
            [self.points[idx] for idx in p_indices], self.kernel.support(), workers=-1
        )

        for n, neighbors in enumerate(neighbor_lists):
            for i in neighbors:
                p = self.points[i]
                p_idx = self.points[p_indices[n]]

                if mask is None or mask[i]:
                    rho[i] += self.kernel(np.dot(p - p_idx, p - p_idx))

        return rho

    def add(self, rho_s: np.array, idx, mask=None):
        p_idx = self.points[idx]

        neighbors = self.tree.query_ball_point(p_idx, self.kernel.support(), workers=-1)
        for i in neighbors:
            p = self.points[i]
            if mask is None or mask[i]:
                rho_s[i] += self.kernel(np.dot(p - p_idx, p - p_idx))

    def sub(self, rho_s: np.array, idx, mask=None):
        p_idx = self.points[idx]

        neighbors = self.tree.query_ball_point(p_idx, self.kernel.support(), workers=-1)
        for i in neighbors:
            p = self.points[i]
            if mask is None or mask[i]:
                rho_s[i] -= self.kernel(np.dot(p - p_idx, p - p_idx))
