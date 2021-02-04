import pytest

import numpy as np

from vc_sample.density_estimation import KernelDensityEstimator
from vc_sample.void_and_cluster import VoidAndCluster


@pytest.mark.parametrize("sample_size", [10, 20, 50, 100])
def test_sample_size(sample_size):
    xs = np.random.uniform(-1.0, 1.0, 100)
    ys = np.random.uniform(-1.0, 1.0, 100)
    points = np.stack([xs, ys]).T

    density_estimator = KernelDensityEstimator(points)
    vc = VoidAndCluster(points, density_estimator, num_initial_samples=20)

    samples = vc.sample(size=sample_size)
    ordering = vc.get_ordering(size=sample_size)

    for o in ordering:
        assert o < sample_size

    assert len(samples) == len(ordering) == sample_size
    assert np.unique(samples, axis=0).shape[0] == sample_size


def test_stratification():
    ps = np.linspace(0.0, 10.0, 100).reshape(-1, 1)

    density_estimator = KernelDensityEstimator(ps)
    vc = VoidAndCluster(ps, density_estimator, num_initial_samples=20)

    sample_size = 50
    nbins = 10

    samples = vc.sample(size=sample_size)

    bins, _ = np.histogram(samples, bins=nbins)
    for b in bins:
        assert (sample_size / nbins) - 1 < b < (sample_size / nbins) + 1
