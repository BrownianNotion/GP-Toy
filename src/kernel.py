import numpy as np
from numpy.typing import ArrayLike


# For now, vectorise and assume 1d input x
class Kernel:
    def __init__(self, params: dict):
        self.params = params
    
    def kernel_function(x1, x2):
        pass

    def evaluate(self, x1, x2):
        return self.kernel_function(x1, x2)


class IdentityKernel(Kernel):
    def __init__(self, scale=1.0):
        params = {"scale": scale}
        super().__init__(params)

    def kernel_function(self, x1, x2):
        return self.params["scale"] * np.eye(len(x1)) 

class GaussianKernel(Kernel):
    def __init__(self, amplitude, length_scale):
        params = {
            "sigma": amplitude,
            "l": length_scale
        }
        super().__init__(params)
    
    def kernel_function(self, x1, x2):
        x1 = np.expand_dims(x1, axis=0)
        x2 = np.expand_dims(x2, axis=1)
        s, l = self.params["sigma"], self.params["l"]
        return s * np.exp(-0.5 * (x1 - x2)**2/ l**2) 

