import numpy as np
import numpy.testing as npt
import matplotlib.pyplot as plt

from src.kernel import IdentityKernel
from src.kernel import GaussianKernel
from src.GP import GaussianProcess

from unittest.mock import patch

np.random.seed(42)


def test_identity_kernel():
    scale = 0.5
    N = 2
    x1 = np.random.rand(N, 1)
    x2 = np.random.rand(N, 1)

    kernel = IdentityKernel(scale=scale)
    npt.assert_allclose(kernel.evaluate(x1, x2), scale * np.eye(N))


def test_gaussian_kernel():
    s = 0.5
    l = 2.0

    kernel = GaussianKernel(s, l)

    x1 = np.array([1.0, 0.0])

    K_desired = np.array([[0.5, 0.5 * np.exp(-0.125)], [0.5 * np.exp(-0.125), 0.5]])
    npt.assert_allclose(kernel.evaluate(x1, x1), K_desired)


@patch("matplotlib.pyplot.show")
def test_gaussian_process_fit(mock_show):
    N = 6
    X_train = np.linspace(0, 1, N)
    x_test = np.array([0.5, 1.1])
    s = 1.0
    l = 1.0
    kernel = GaussianKernel(s, l)

    y_train = np.random.multivariate_normal(
        mean=np.zeros(len(X_train)), cov=kernel.evaluate(X_train, X_train)
    )

    gp = GaussianProcess(kernel)
    gp.fit(X_train, y_train)
    train_error_bars = gp.get_train_error_bars()

    y_predict, predict_error_bars = gp.predict(x_test)

    def plot_fn():
        fig, ax = plt.subplots()
        ax.scatter(X_train, y_train, color="blue", label="train")
        ax.scatter(x_test, y_predict, color="orange", label="predict")
        ax.fill_between(
            X_train, y_train - train_error_bars, y_train + train_error_bars, color="blue", alpha=0.2
        )
        plt.show()
    
    plot_fn()
