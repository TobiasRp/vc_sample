import pytest

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


@pytest.fixture
def kde1d():
    xs = np.linspace(-1.0, -1.0, 100)
    points = xs.reshape(-1, 1)
    return KernelDensityEstimator(points, kernel=Kernel(epanechnikov, scale=0.5))


def test_add_sub(kde1d):
    mask = np.zeros(kde1d.points.shape[0], dtype=bool)
    mask[10] = True
    rho1 = kde1d.estimate(mask=mask)

    rho2 = np.zeros(kde1d.points.shape[0], dtype=float)
    kde1d.add(rho2, 10)

    assert np.array_equal(rho1, rho2)


def test_kde1d():
    xs = np.linspace(-1.0, -1.0, 100)
    points = xs.reshape(-1, 1)

    kde = KernelDensityEstimator(points, Kernel(epanechnikov, scale=1.0))

    rho = kde.estimate(mask=np.zeros_like(points, dtype=bool))
    assert not rho.any()

    rho = kde.estimate()
    rho_50_old = rho[50]
    kde.sub(rho, 50)
    assert rho[50] < rho_50_old

    kde.add(rho, 50)
    assert rho[50] == rho_50_old


def test_kde_border_case():
    xs = np.array([-100, -10, 0, 10, 100], dtype=float)
    points = xs.reshape(-1, 1)
    kde = KernelDensityEstimator(points, Kernel(epanechnikov, scale=1.0))
    rho = kde.estimate()
    for i, _ in enumerate(xs):
        assert rho[i] == epanechnikov(0.0)


def test_kde2d():
    x_ = np.linspace(0.0, 1.0, 10)
    y_ = np.linspace(0.0, 1.0, 10)
    x, y = np.meshgrid(x_, y_)
    points = np.stack([x.flatten(), y.flatten()]).T

    kde = KernelDensityEstimator(points, Kernel(epanechnikov, scale=0.1))
    rho = kde.estimate().reshape(10, 10)

    assert np.abs(rho[6, 6] - rho[4, 4]) < 1e-3
