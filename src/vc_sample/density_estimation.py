import numpy as np
from scipy.spatial import cKDTree


def epanechnikov(x: float):
    if 0 <= x < 1.0:
        return 0.75 * (1 - x ** 2)
    else:
        return 0.0


def gaussian(x: float):
    return 1.0 / (2.0 * np.pi) * np.exp(-0.5 * x ** 2)


def l2norm(vec):
    return np.dot(vec, vec)


def kernel_scale_factor(dimensionality: float, num_points: int, num_samples: int):
    return (num_points / float(num_samples)) ** (1.0 / dimensionality)


class Kernel:
    """A kernel function assigns a weight based on distance between two points."""

    def __init__(self, kernel_func, scale: float, norm=l2norm):
        """Create a new kernel with the given scaling factor.

        Args:
            kernel_func: Kernel function with a support of 1
            scale: Global scaling factor
        """
        self.kernel = kernel_func
        self.norm = norm
        self.scale = scale

    def __call__(self, vec: int):
        """Evaluates the kernel.

        Args:
            vec: Distance vector for which to compute a weight.
        Returns:
            Scalar weight
        """
        x = self.norm(vec / self.scale)
        return self.kernel(x)

    def support(self):
        return self.scale


class KernelDensityEstimator:
    """Estimates density using kernel functions."""

    def __init__(self, points: np.array, kernel):
        self.kernel = kernel
        self.points = points
        self.tree = cKDTree(points)

    def estimate(self, mask: np.array = None):
        """Estimates the density for all points.

        Args:
            mask: Optional parameter to mask points to exclude during the density estimation.
        Returns:
            Array of densities
        """
        rho = np.zeros(self.points.shape[0], dtype=float)

        for i in range(self.points.shape[0]):
            if mask is None or mask[i]:
                self.add(rho, i)

        return rho

    def add(self, rho_s: np.array, idx: int, mask: np.array = None):
        """ For given densities, adds the density of a point.

        Args:
            rho_s: Existing densities to update.
            idx: Index of the point.
            mask: Optional mask to exclude points during the density estimation.

        Returns:
            None
        """
        p_idx = self.points[idx]

        neighbors = self.tree.query_ball_point(p_idx, self.kernel.support(), workers=-1)
        for i in neighbors:
            p = self.points[i]
            if mask is None or mask[i]:
                rho_s[i] += self.kernel(np.dot(p - p_idx, p - p_idx))

    def sub(self, rho_s: np.array, idx: int, mask: np.array = None):
        """ For given densities, subtracts the density of a point.

        Args:
            rho_s: Existing densities to update.
            idx: Index of the point.
            mask: Optional mask to exclude points during the density estimation.

        Returns:
            None
        """
        p_idx = self.points[idx]

        neighbors = self.tree.query_ball_point(p_idx, self.kernel.support(), workers=-1)
        for i in neighbors:
            p = self.points[i]
            if mask is None or mask[i]:
                rho_s[i] -= self.kernel(np.dot(p - p_idx, p - p_idx))