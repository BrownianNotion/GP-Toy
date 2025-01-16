import numpy as np
from .kernel import Kernel


class GaussianProcess:
    def __init__(self, kernel: Kernel, mean=None, sigma_noise=1.0, seed=42):
        self.mean = 0.0 if mean is None else mean
        self.kernel = kernel
        self.K_XX = None  # training Gram matrix
        self.K_XX_inv = None
        self.sigma_noise = sigma_noise

        self.X_train = None
        self.y_train = None

        self.seed = seed
        self.random = np.random.default_rng(seed=seed)

    def fit(self, X_train, y_train):
        # TODO: change to cholesky implementation for efficiency
        self.K_XX = self.kernel.evaluate(X_train, X_train)
        self.K_XX_inv = np.linalg.inv(
            self.K_XX + self.sigma_noise**2 * np.eye(len(X_train))
        )

        self.X_train = X_train

        self.y_train = y_train
        if len(self.y_train.shape) == 1:
            self.y_train = np.expand_dims(self.y_train, axis=1)

    def get_train_error_bars(self):
        if self.X_train is None:
            raise ValueError("Please call fit first.")

        return np.sqrt(np.diag(self.K_XX))

    def predict(self, x_test):
        K_xX = self.kernel.evaluate(x_test, self.X_train)
        K_xx = self.kernel.evaluate(x_test, x_test)

        posterior_mean = K_xX @ self.K_XX_inv @ self.y_train
        posterior_covariance = K_xx - K_xX @ self.K_XX_inv @ K_xX.T
        error_bars = np.sqrt(np.diag(posterior_covariance))

        return (
            self.random.multivariate_normal(
                mean=posterior_mean.flatten(), cov=posterior_covariance
            ).flatten(),
            error_bars,
        )
