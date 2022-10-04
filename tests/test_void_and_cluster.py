import pytest

import numpy as np

from vc_sample.density_estimation import KernelDensityEstimator, UMAPDensityEstimator
from vc_sample.kernels import GaussianKernel
from vc_sample.void_and_cluster import VoidAndCluster

from sklearn.datasets import make_circles


@pytest.mark.parametrize("sample_size", [10, 20, 50, 100])
def test_sample_size(sample_size):
    xs = np.random.uniform(-1.0, 1.0, 100)
    ys = np.random.uniform(-1.0, 1.0, 100)
    points = np.stack([xs, ys]).T

    density_estimator = KernelDensityEstimator(points, GaussianKernel(sigma=1.0))
    vc = VoidAndCluster(density_estimator, points.shape[0], num_initial_samples=20)

    sample_indices = vc.sample(size=sample_size)
    samples = points[sample_indices]

    ordering = vc.ordering(size=sample_size)

    for o in ordering:
        assert o < points.shape[0]

    assert len(samples) == len(ordering) == sample_size
    assert np.unique(samples, axis=0).shape[0] == sample_size


def test_umap_circle():
    circles, circles_class = make_circles(n_samples=100, random_state=42)
    estimator = UMAPDensityEstimator(X=circles, n_neighbors=10)

    vc = VoidAndCluster(estimator, circles.shape[0], num_initial_samples=10)

    sample_indices = vc.sample(size=20)
    assert len(sample_indices) == 20

    circles_class_samples = circles_class[sample_indices]
    assert np.unique(circles_class_samples).shape[0] == 2
