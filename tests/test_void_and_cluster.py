import numpy as np

from vc_sample.density_estimation import KernelDensityEstimator
from vc_sample.void_and_cluster import VoidAndCluster


def test_sample():
    xs = np.linspace(-1.0, -1.0, 100)
    ys = np.linspace(-1.0, -1.0, 100)
    points = np.stack([xs, ys]).T

    density_estimator = KernelDensityEstimator(points)
    vc = VoidAndCluster(points, density_estimator, num_initial_samples=20)
    vc.sample(size=100)
