import pandas as pd

from vc_sample.sampling import sample


def test_sample():
    iris = pd.read_csv("data/iris.csv")
    X_sepal = iris[["sepal_width", "sepal_length"]].to_numpy()

    X_sampled_all = sample(X_sepal, size=X_sepal.shape[0])
    assert X_sampled_all.shape == X_sepal.shape

    X_sampled_half = sample(X_sepal, size=int(X_sepal.shape[0] * 0.5))
    assert X_sampled_half.shape[0] == (X_sepal.shape[0] * 0.5)
