import numpy as np
import scipy.special
from discretization import Quadrature
from kernel import Kernel
from abc import ABC, abstractmethod

class Basis(ABC):
    @abstractmethod
    def evaluate(self, n: int, x: np.ndarray, L: float) -> np.ndarray:
        pass

class LaplacianEigenfunctions(Basis):
    """Neumann eigenfunctions of the Laplacian on [-L, L]."""
    def evaluate(self, n: int, x: np.ndarray, L: float) -> np.ndarray:
        if n == 0:
            return np.full_like(x, 1.0 / np.sqrt(2 * L))
        else:
            return (1.0 / np.sqrt(L)) * np.cos(n * np.pi * (x + L) / (2 * L))

class LegendrePolynomials(Basis):
    """Legendre polynomials orthogonal on [-L, L]."""
    def evaluate(self, n: int, x: np.ndarray, L: float) -> np.ndarray:
        # Scale x from [-L, L] to [-1, 1]
        x_scaled = x / L
        P_n = scipy.special.legendre(n)
        # Normalization factor for orthonormal basis on [-L, L]
        norm = np.sqrt((2 * n + 1) / (2 * L))
        return norm * P_n(x_scaled)

class CanonicalBasis(Basis):
    """Piecewise constant indicator functions on N uniform subintervals."""
    def evaluate(self, n: int, x: np.ndarray, L: float, N_intervals: int = None) -> np.ndarray:
        if N_intervals is None:
            N_intervals = max(10, n+1) # fallback
        dx = (2 * L) / N_intervals
        edges = np.linspace(-L, L, N_intervals + 1)
        # Interval n is [edges[n], edges[n+1])
        mask = (x >= edges[n]) & (x <= edges[n+1])
        if n == N_intervals - 1:
            mask = (x >= edges[n]) & (x <= edges[n+1]) # include right endpoint
        else:
            mask = (x >= edges[n]) & (x < edges[n+1])
        return np.where(mask, 1.0 / np.sqrt(dx), 0.0)

class GalerkinBuilder:
    def __init__(self, kernel: Kernel, quadrature: Quadrature):
        self.kernel = kernel
        self.quadrature = quadrature

    def build_matrix(self, basis: Basis, N_basis: int, L: float, N_quad: int = 500):
        # 1. Get quadrature nodes and weights
        nodes, weights = self.quadrature.get_nodes_and_weights(L, N_quad)
        actual_N_quad = len(nodes)
        
        # 2. Evaluate basis on nodes
        # Phi shape: (actual_N_quad, N_basis)
        Phi = np.zeros((actual_N_quad, N_basis))
        for n in range(N_basis):
            if isinstance(basis, CanonicalBasis):
                Phi[:, n] = basis.evaluate(n, nodes, L, N_intervals=N_basis)
            else:
                Phi[:, n] = basis.evaluate(n, nodes, L)
                
        # 3. Kernel matrix evaluated at quadrature pairs
        # J_mat shape: (N_quad, N_quad)
        X, Y = np.meshgrid(nodes, nodes, indexing='ij')
        J_mat = self.kernel(X - Y)
        
        # 4. Degree function b(x_k)
        b_vals = J_mat @ weights
        
        # 5. Assemble Galerkin matrix M
        # M_ij = <L phi_j, phi_i> = int int J(x-y)phi_j(y)phi_i(x) dx dy - int b(x)phi_j(x)phi_i(x) dx
        # In matrix form: M = Phi^T W J W Phi - Phi^T W B Phi
        W = np.diag(weights)
        B = np.diag(b_vals)
        
        part1 = Phi.T @ W @ J_mat @ W @ Phi
        part2 = Phi.T @ W @ B @ Phi
        
        M = part1 - part2
        # Symmetrize to remove numerical noise
        M = 0.5 * (M + M.T)
        return M, Phi, nodes
