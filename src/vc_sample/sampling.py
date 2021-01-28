import numpy as np


def sample(in_samples: np.array, size: int):
    """Returns stratified samples from X.

    Parameters
    ----------
    in_samples : 2D array
    Array of observations (rows) and variables/dimensions/features (columns).

    size : int
    The amount of samples from X that will be returned.

    """
    assert isinstance(in_samples, np.ndarray) and len(in_samples.shape) == 2
    assert 0 <= size <= in_samples.shape[0]

    # TODO: Simple random sampling for now - replace this
    indices = np.arange(0, in_samples.shape[0], dtype=np.uint)
    selected_indices = np.random.choice(indices, size=size)
    return in_samples[selected_indices, :]
