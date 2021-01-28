import numpy as np

from vc_sample.density_estimation import EpanechnikovKernel, KernelDensityEstimator


def test_epanechnikov_kernel():
    assert EpanechnikovKernel()(0.5) == 0.5625
    assert EpanechnikovKernel()(-1.0) == 0.0
    assert EpanechnikovKernel()(1.0) == 0.0


def test_kde():
    xs = np.linspace(-1.0, -1.0, 100)
    points = xs.reshape(-1, 1)

    kde = KernelDensityEstimator(points)

    rho = kde.estimate(mask=np.zeros_like(points, dtype=np.bool))
    assert not rho.any()

    rho = kde.estimate()
    rho_50_old = rho[50]
    kde.sub(rho, 50)
    assert rho[50] < rho_50_old

    kde.add(rho, 50)
    assert rho[50] == rho_50_old
