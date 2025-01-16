import numpy as np
from numpy.typing import ArrayLike


class Kernel:
    def __init__(self, params: dict, kernel_function):
        self.__params = params
        self.__k = kernel_function

    def evaluate(self, x1, x2):
        return self.__k(x1, x2)


class IdentityKernel(Kernel):
    def __init__(self, scale=1.0):
        params = {"scale": scale}
        kernel_function = lambda x1, x2: scale * np.eye(len(x1))
        super().__init__(params, kernel_function)

    def evaluate(self, x1, x2):
        return super().evaluate(x1, x2)
