import numpy as np
from scipy.spatial import cKDTree


def epanechnikov(x: float):
    if 0 <= x < 1.0:
        return 0.75 * (1 - x ** 2)
    else:
        return 0.0


def l2norm(vec):
    return np.dot(vec, vec)


class Kernel:
    """
    The Epanechnikov
    """

    def __init__(self, kernel_func, norm=l2norm, scale: float = 1.0):
        """
        Create a new kernel with the given scaling factor.
        Args:
            kernel_func: function
            Kernel function with a support of 1

            scale: float
            Global scaling factor
        """
        self.kernel = kernel_func
        self.norm = norm
        self.scale = scale

    def __call__(self, vec: int):
        """
        Evaluates kernel
        """
        x = self.norm(vec / self.scale)
        return self.kernel(x)

    def support(self):
        return self.scale


class KernelDensityEstimator:
    def __init__(self, points, kernel=Kernel(kernel_func=epanechnikov)):
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
                    rho[i] += self.kernel(p - p_idx)

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
