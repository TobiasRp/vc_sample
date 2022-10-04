from typing import Protocol

import numpy as np
import umap
from scipy.spatial import cKDTree

from vc_sample.kernels import Kernel


class DensityEstimator(Protocol):
    """A DensityEstimator estimates the density for each data point.

    Besides a complete estimation for each data point
    (with or without masking out some data points),
    it enables efficient updates by adding/removing data points from an estimated density.
    """

    def estimate(self, mask: np.array = None) -> np.array:
        ...

    def add(self, rho_s: np.array, idx: int) -> np.array:
        ...

    def sub(self, rho_s: np.array, idx: int) -> np.array:
        ...


class KernelDensityEstimator:
    """Estimates density using kernel functions."""

    def __init__(
        self,
        points: np.array,
        kernel: Kernel,
        divide_data_density: bool = True,
        importance: np.array = None,
    ):
        """
        Creates a new kernel density estimator.

        Args:
            points: Data points.
            kernel: The kernel function to use for density estimation.
            divide_data_density: If true, the density of data points will be divided out.
                                 Sampling from this density will thus keep the density
                                 of the original data points, instead of always sampling
                                 pairwise maximally distant.
            importance: Optional importance (or probabilities) for sampling non-uniformly,
                        by multiplying the estimated density with the _inverse_ importance.
        """
        self.kernel = kernel
        self.points = points
        self.tree = cKDTree(points)

        self.inv_weight = 1.0
        if divide_data_density:
            self.inv_weight = 1.0 / self.estimate()

        if importance is not None:
            self.inv_weight /= importance

    def _weighting_factor(self, idx: int) -> float:
        """For a given idx, returns the inverse weighting to apply to the
        estimated density.
        """
        return self.inv_weight if np.isscalar(self.inv_weight) else self.inv_weight[idx]

    def estimate(self, mask: np.array = None) -> np.array:
        """Estimates the density for all points.

        Args:
            mask: Optional parameter to mask points to exclude during density estimation.
        Returns:
            Array of densities
        """
        rho = np.zeros(self.points.shape[0], dtype=np.float64)

        for i in range(self.points.shape[0]):
            if mask is None or mask[i]:
                self.add(rho, i)

        return rho

    def add(self, rho_s: np.array, idx: int):
        """For given densities, adds the density of the point indexed by ``idx``.

        Args:
            rho_s: Existing densities to update.
            idx: Index of the point.
        """
        p_idx = self.points[idx]

        neighbors = self.tree.query_ball_point(p_idx, self.kernel.support(), workers=-1)
        for i in neighbors:
            p = self.points[i]
            rho_s[i] += self.kernel(
                np.dot(p - p_idx, p - p_idx)
            ) * self._weighting_factor(idx)

    def sub(self, rho_s: np.array, idx: int):
        """For given densities, subtracts the density of a point.

        Args:
            rho_s: Existing densities to update.
            idx: Index of the point.
        """
        p_idx = self.points[idx]

        neighbors = self.tree.query_ball_point(p_idx, self.kernel.support(), workers=-1)
        for i in neighbors:
            p = self.points[i]
            rho_s[i] -= self.kernel(
                np.dot(p - p_idx, p - p_idx)
            ) * self._weighting_factor(idx)


class UMAPDensityEstimator:
    """Estimates density along an embedded manifold in a higher-dimensional space,
    which is represented as a weighted graph. Such a graph is constructed based
    on the UMAP dimensionality reduction technique.
    """

    def __init__(
        self,
        X: np.array,
        n_neighbors: int,
        metric: str = "euclidean",
        random_state=0,
        **kwargs
    ):
        """
        Args:
            X: Data points in a high-dimensional space
            n_neighbors: Number of neighbors to approximate geodesic distance, cf. UMAP.
            metric: The distance metric function or name (string), cf. UMAP.
            kwargs: Additional parameters for UMAP are documented in ``fuzzy_simplicial_set``.
        """
        self.num = X.shape[0]
        self.graph, _, _ = umap.umap_.fuzzy_simplicial_set(
            X,
            n_neighbors=n_neighbors,
            metric=metric,
            random_state=random_state,
            **kwargs
        )

    def estimate(self, mask: np.array = None) -> np.array:
        rho = np.zeros(self.num, dtype=np.float64)
        for i in range(self.num):
            if mask is None or mask[i]:
                self.add(rho, i)
        return rho

    def add(self, rho_s: np.array, idx: int):
        for i in self.graph[idx, :].nonzero():
            rho_s[i] += self.graph[idx, i]

    def sub(self, rho_s: np.array, idx: int):
        for i in self.graph[idx, :].nonzero():
            rho_s[i] -= self.graph[idx, i]
