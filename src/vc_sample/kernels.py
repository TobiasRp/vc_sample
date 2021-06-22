from typing import Protocol

import numpy as np


def kernel_scale_factor(
    dimensionality: float, num_points: int, num_samples: int
) -> float:
    """Computes a scaling factor to accound for dimensionality as well
    as the number (or ratio) of samples.
    """
    return (num_points / float(num_samples)) ** (1.0 / dimensionality)


class Kernel(Protocol):
    """A kernel function assigns a weight based on distance between two points."""

    def __call__(self, vec: np.array) -> float:
        """Evaluates the kernel.

        Args:
            vec: Distance vector for which to compute a weight.
        Returns:
            Scalar weight
        """
        ...

    def support(self) -> float:
        """Returns the support of the kernel, i.e.
        after which it will return zero.

        Note that some kernels have infinite support and might return
        something approximate instead.
        """
        ...


class GaussianKernel:
    def __init__(self, sigma: float):
        """Create a new Gaussian kernel.

        Args:
            sigma: Standard deviation
        """
        self.sigma = sigma
        self.norm_factor = 1 / (sigma * np.sqrt(2 * np.pi))

    def __call__(self, vec: np.array) -> float:

        x = np.dot(vec, vec)
        return self.norm_factor * np.exp(-0.5 * (x / self.sigma) ** 2)

    def support(self) -> float:
        return 2.0 * self.sigma
