import numpy as np
import numpy.testing as npt

from src.kernel import IdentityKernel

np.random.seed(42)


def test_identity_kernel():
    scale = 0.5
    N = 2
    x1 = np.random.rand(N, 1)
    x2 = np.random.rand(N, 1)

    id_kernel = IdentityKernel(scale=scale)
    npt.assert_allclose(id_kernel.evaluate(x1, x2), scale * np.eye(N))
