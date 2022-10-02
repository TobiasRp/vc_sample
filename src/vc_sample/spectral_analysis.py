from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np


def _radial_frequency(x: np.array, y: np.array):
    return np.asarray(np.round(np.sqrt(x ** 2 + y ** 2)), dtype=np.int)


def _create_grid(dft: np.array):
    x, y = np.meshgrid(range(dft.shape[1]), range(dft.shape[0]))
    x -= int(dft.shape[1] / 2)
    y -= int(dft.shape[0] / 2)
    return x, y


def _fast_fourier_transform(image: np.array):
    return np.fft.fftshift(np.fft.fft2(image)) / float(np.size(image))


def plot_fourier_transform(image: np.array, cmap: str = "viridis"):
    """Plots the Fourier transform of a 2-dimensional binary image.

    Args:
        image: Binary image (e.g. as returned by ``discretize_as_image``).
        cmap: Name of the colormap to use.

    Returns:
        Figure of the created plot.
    """
    fig = plt.figure()
    fig.add_subplot(
        1,
        1,
        1,
        title="Fourier transform (absolute value)",
        xlabel="$\\omega_x$",
        ylabel="$\\omega_y$",
    )

    dft = _fast_fourier_transform(image)
    height, width = image.shape
    shift_y, shift_x = height // 2, width // 2
    extent = (
        -shift_x - 0.5,
        width - shift_x - 0.5,
        -shift_y + 0.5,
        height - shift_y + 0.5,
    )

    plt.imshow(
        np.abs(dft),
        cmap=cmap,
        interpolation="nearest",
        vmin=0.0,
        vmax=np.percentile(np.abs(dft), 99),
        extent=extent,
    )
    plt.colorbar()
    return fig


def plot_power_distribution(image: np.array):
    """Plot the distribution of power over radial frequency bands.

    Args:
        image: Binary image (e.g. as returned by ``discretize_as_image``).

    Returns:
        Figure of the created plot.
    """
    dft = _fast_fourier_transform(image)

    fig_radial = plt.figure()
    fig_radial.add_subplot(
        1,
        1,
        1,
        title="Radial power distribution",
        xlabel="Distance from center / pixels",
    )

    radial_frequency = _radial_frequency(*_create_grid(dft))
    radial_power = np.zeros((np.max(radial_frequency) - 1,))
    dft[int(dft.shape[0] / 2), int(dft.shape[1] / 2)] = 0.0
    for i in range(radial_power.shape[0]):
        radial_power[i] = np.sum(
            np.where(radial_frequency == i, np.abs(dft), 0.0)
        ) / np.count_nonzero(radial_frequency == i)
    plt.plot(np.arange(np.max(radial_frequency) - 1) + 0.5, radial_power)
    return fig_radial


def plot_anisotropy(image: np.array):
    """Plot the distribution of power over angular frequency ranges.

    Args:
        image: Binary image (e.g. as returned by ``discretize_as_image``).

    Returns:
        Figure of the created plot.
    """
    dft = _fast_fourier_transform(image)
    x, y = _create_grid(dft)
    radial_frequency = _radial_frequency(x, y)

    fig_aniso = plt.figure()
    fig_aniso.add_subplot(
        1,
        1,
        1,
        title="Anisotropy (angular power distribution)",
        aspect="equal",
        xlabel="Frequency x",
        ylabel="Frequency y",
    )
    circular_mask = np.logical_and(
        0 < radial_frequency,
        radial_frequency < int(min(dft.shape[0], dft.shape[1]) / 2),
    )
    normalized_x = np.asarray(x, dtype=np.float) / np.maximum(
        1.0, np.sqrt(x ** 2 + y ** 2)
    )
    normalized_y = np.asarray(y, dtype=np.float) / np.maximum(
        1.0, np.sqrt(x ** 2 + y ** 2)
    )
    binning_angle = np.linspace(0.0, 2.0 * np.pi, 33)
    angular_power = np.zeros_like(binning_angle)
    for i, Angle in enumerate(binning_angle):
        dot_product = normalized_x * np.cos(Angle) + normalized_y * np.sin(Angle)
        full_mask = np.logical_and(circular_mask, dot_product >= np.cos(np.pi / 32.0))
        angular_power[i] = np.sum(
            np.where(full_mask, np.abs(dft), 0.0)
        ) / np.count_nonzero(full_mask)
    mean_angular_power = np.mean(angular_power[1:])
    dense_angle = np.linspace(0.0, 2.0 * np.pi, 256)
    plt.plot(
        np.cos(dense_angle) * mean_angular_power,
        np.sin(dense_angle) * mean_angular_power,
        color=(0.7, 0.7, 0.7),
    )
    plt.plot(
        np.cos(binning_angle) * angular_power, np.sin(binning_angle) * angular_power
    )
    return fig_aniso


def discretize_as_image(xs: np.array, ys: np.array, resolution: Tuple[int, int]):
    """Discretizes a set of 2-dimensional points to a (binary) image.

    Args:
        xs: x-coordinates of the points
        ys: y-coordinates of the points
        resolution: Image resolution

    Returns:
        An image of size ``resolution`` with a value of 1.0 if it "contains" a point
        and zero otherwise.
    """
    x_extrema = (np.min(xs), np.max(xs))
    y_extrema = (np.min(ys), np.max(ys))

    img = np.zeros(resolution, dtype=np.float32)
    for i in range(len(xs)):
        x = xs[i]
        y = ys[i]

        x_ndc = (x - x_extrema[0]) / (x_extrema[1] - x_extrema[0])
        y_ndc = 1.0 - (y - y_extrema[0]) / (y_extrema[1] - y_extrema[0])

        xi = int(x_ndc * (resolution[0] - 1))
        yi = int(y_ndc * (resolution[1] - 1))

        img[yi, xi] = 1.0

    return img
