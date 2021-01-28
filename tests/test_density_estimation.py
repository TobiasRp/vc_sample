import numpy as np

from vc_sample.density_estimation import Kernel, KernelDensityEstimator, epanechnikov


def test_epanechnikov_kernel():
    assert epanechnikov(0.5) == 0.5625
    assert epanechnikov(-1.0) == 0.0
    assert epanechnikov(1.0) == 0.0


def test_kernel():
    scale = 3.0
    k = Kernel(epanechnikov, scale=scale)
    assert k.support() == scale
    assert k(np.array([10, 9])) == 0.0
    assert k(1.5) == epanechnikov(0.5 ** 2)


def test_kde1d():
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


def test_kde_border_case():
    xs = np.array([-100, -10, 0, 10, 100], dtype=np.float)
    points = xs.reshape(-1, 1)
    kde = KernelDensityEstimator(points)
    rho = kde.estimate()
    for i, _ in enumerate(xs):
        assert rho[i] == epanechnikov(0.0)
