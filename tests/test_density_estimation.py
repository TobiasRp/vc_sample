import pytest

import numpy as np

from vc_sample.density_estimation import KernelDensityEstimator, UMAPDensityEstimator
from vc_sample.kernels import GaussianKernel

from sklearn.datasets import make_circles


@pytest.fixture
def kde1d():
    xs = np.linspace(-1.0, -1.0, 100)
    points = xs.reshape(-1, 1)
    return KernelDensityEstimator(points, kernel=GaussianKernel(sigma=0.5))


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

    kde = KernelDensityEstimator(
        points, GaussianKernel(sigma=1.0), divide_data_density=False
    )

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
    gauss = GaussianKernel(sigma=1.0)
    kde = KernelDensityEstimator(points, gauss, divide_data_density=False)
    rho = kde.estimate()
    for i, _ in enumerate(xs):
        assert rho[i] == gauss(0.0)


def test_kde2d():
    x_ = np.linspace(0.0, 1.0, 10)
    y_ = np.linspace(0.0, 1.0, 10)
    x, y = np.meshgrid(x_, y_)
    points = np.stack([x.flatten(), y.flatten()]).T

    kde = KernelDensityEstimator(
        points, GaussianKernel(sigma=0.1), divide_data_density=False
    )
    rho = kde.estimate().reshape(10, 10)

    assert np.abs(rho[6, 6] - rho[4, 4]) < 1e-3


def test_divide_data_density():
    xs = np.linspace(-1.0, -1.0, 100)
    points = xs.reshape(-1, 1)
    kde = KernelDensityEstimator(
        points, GaussianKernel(sigma=0.1), divide_data_density=True
    )

    estimate = kde.estimate()
    assert ((estimate - 1.0) < 0.001).all()


def test_umap_density_estimator():
    circles, _ = make_circles(n_samples=100, random_state=42)
    estimator = UMAPDensityEstimator(X=circles, n_neighbors=6)
    density = estimator.estimate()
    assert (0.0 < density).all()

    density_copy = np.copy(density)

    indices = [0, 10, 42, 99]
    for idx in indices:
        estimator.add(density, idx)

    assert (density >= density_copy).all()

    for idx in indices:
        estimator.sub(density, idx)

    assert np.isclose(density, density_copy).all()