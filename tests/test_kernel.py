import numpy as np
import numpy.testing as npt

from src.kernel import IdentityKernel
from src.kernel import GaussianKernel

np.random.seed(42)


def test_identity_kernel():
    scale = 0.5
    N = 2
    x1 = np.random.rand(N, 1)
    x2 = np.random.rand(N, 1)

    kernel = IdentityKernel(scale=scale)
    npt.assert_allclose(kernel.evaluate(x1, x2), scale * np.eye(N))

def test_gaussian_kernel():
    s = .5
    l = 2.
    
    kernel = GaussianKernel(s, l)
    
    x1 = np.array([1.0, 0.])
    
    K_desired = np.array([
        [0.5, 0.5 * np.exp(-0.125)],
        [0.5 * np.exp(-0.125), 0.5]
    ])
    npt.assert_allclose(kernel.evaluate(x1, x1), K_desired)