import numpy as np
from abc import ABC, abstractmethod

class Quadrature(ABC):
    """Base class for generating 1D quadrature nodes and weights on [-L, L]."""
    @abstractmethod
    def get_nodes_and_weights(self, L: float, N: int):
        pass

class TrapezoidalQuadrature(Quadrature):
    def get_nodes_and_weights(self, L: float, N: int):
        nodes = np.linspace(-L, L, N)
        dx = nodes[1] - nodes[0]
        weights = np.full(N, dx)
        weights[0] = dx / 2.0
        weights[-1] = dx / 2.0
        return nodes, weights

class SimpsonQuadrature(Quadrature):
    def get_nodes_and_weights(self, L: float, N: int):
        if N % 2 == 0:
            N += 1  # Simpson's rule requires an odd number of points
        nodes = np.linspace(-L, L, N)
        dx = nodes[1] - nodes[0]
        weights = np.full(N, dx / 3.0)
        weights[1:-1:2] *= 4
        weights[2:-1:2] *= 2
        return nodes, weights

class ClenshawCurtisQuadrature(Quadrature):
    def get_nodes_and_weights(self, L: float, N: int):
        """
        Computes Clenshaw-Curtis nodes and weights on [-L, L].
        Using standard explicit formula for N points.
        """
        if N < 2:
            return np.array([0.0]), np.array([2.0 * L])
            
        n = N - 1
        theta = np.pi * np.arange(N) / n
        x = np.cos(theta) # Chebyshev nodes on [-1, 1]
        
        # Explicit weights formula for Clenshaw-Curtis
        w = np.zeros(N)
        c = 2.0 / n
        for i in range(N):
            s = 0.0
            for j in range(1, n//2 + 1):
                b = 1.0 if j == n//2 else 2.0
                s += (1.0 / (4*j**2 - 1)) * np.cos(2 * j * theta[i]) * b
            w[i] = c * (1.0 - s)
        w[0] /= 2.0
        w[-1] /= 2.0
        
        # Scale to [-L, L]
        nodes = -L * x  # Reverse to match standard -L to L order
        weights = w * L
        
        return nodes[::-1], weights[::-1]
