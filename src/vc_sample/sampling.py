import numpy as np


def void_and_cluster(X, size):
    """Performs void-and-cluster sampling to find optimally stratified samples."""
    pass


def sample(X, size):
    """Returns stratified samples from X.

    Parameters
    ----------
    X : 2D array
    Array of observations (rows) and features (columns).

    size : int
    The amount of samples from X that will be returned.

    """
    assert isinstance(X, np.ndarray) and len(X.shape) == 2
    assert size >= 0 and size <= X.shape[0]

    # TODO: Simple random sample - replace this
    inds = np.arange(0, X.shape[0], dtype=np.uint)
    selected_inds = np.random.choice(inds, size=size)
    return X[selected_inds, :]
