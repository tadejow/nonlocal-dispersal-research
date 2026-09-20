import numpy as np
import scipy.special

class Kernel:
    def __call__(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

class GaussianKernel(Kernel):
    def __init__(self, sigma: float = 1.0):
        self.sigma = sigma
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return (1.0 / (np.sqrt(2 * np.pi) * self.sigma)) * np.exp(-0.5 * (x / self.sigma) ** 2)

class LaplaceKernel(Kernel):
    '''Fat Tails (Sub-Gaussian / Laplace) in 1D'''
    def __init__(self, sigma: float = 1.0):
        self.b = sigma / np.sqrt(2.0)
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return (1.0 / (2.0 * self.b)) * np.exp(-np.abs(x) / self.b)

class QuarticKernel(Kernel):
    '''Thin Tails (Super-Gaussian exp(-x^4)) in 1D'''
    def __init__(self, sigma: float = 1.0):
        ratio = scipy.special.gamma(0.75) / scipy.special.gamma(0.25)
        self.B = (ratio / (sigma ** 2)) ** 2
        self.C = (self.B ** 0.25) / (2.0 * scipy.special.gamma(1.25))
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.C * np.exp(-self.B * (x ** 4))
