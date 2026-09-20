import numpy as np

class Kernel:
    """Base class for dispersal kernels."""
    def __call__(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

class GaussianKernel(Kernel):
    """Standard Gaussian dispersal kernel."""
    def __init__(self, sigma: float = 1.0):
        self.sigma = sigma

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return (1.0 / (np.sqrt(2 * np.pi) * self.sigma)) * np.exp(-0.5 * (x / self.sigma) ** 2)
