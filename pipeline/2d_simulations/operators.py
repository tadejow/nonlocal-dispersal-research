import numpy as np
from scipy.sparse import identity, kron

class IntegralOperator2D:
    def __init__(self, x, y, kernel_func, quadrature_type='trapezoidal'):
        self.Nx, self.Ny = len(x), len(y)
        self.N = self.Nx * self.Ny
        hx = x[1] - x[0]

        # Create 2D grid and flatten it to 1D vectors
        xx, yy = np.meshgrid(x, y, indexing='ij')
        x_flat, y_flat = xx.flatten(), yy.flatten()

        # Grid for the integral kernel (all combinations of point pairs)
        grid_x1, grid_x2 = np.meshgrid(x_flat, x_flat, indexing='ij')
        grid_y1, grid_y2 = np.meshgrid(y_flat, y_flat, indexing='ij')

        # Calculate distance matrix and apply the provided kernel function
        r_matrix = np.sqrt((grid_x1 - grid_x2)**2 + (grid_y1 - grid_y2)**2)
        kernel_matrix = kernel_func(r_matrix)

        if quadrature_type == 'trapezoidal':
            wx = self._trapezoidal_weights(x)
            wy = self._trapezoidal_weights(y)
            weights_2d = np.outer(wx, wy).flatten()
        else:
            raise ValueError("Only 'trapezoidal' method is implemented for 2D.")

        # DISCRETE NORMALIZATION: Ensures the integral on our specific sparse grid equals 1.0
        # This prevents the Newton/Gauss-Seidel method from blowing up due to discretization errors.
        grid_inf = np.arange(-40 * hx, 40 * hx + hx / 2, hx)
        X_inf, Y_inf = np.meshgrid(grid_inf, grid_inf)
        r_inf = np.sqrt(X_inf**2 + Y_inf**2)
        discrete_norm = np.sum(kernel_func(r_inf) * hx**2)
        
        kernel_matrix = kernel_matrix / discrete_norm

        # Integral operator matrix (N*N, N*N)
        self.matrix = kernel_matrix * weights_2d[np.newaxis, :]

    def _trapezoidal_weights(self, x_axis):
        N = len(x_axis)
        h = (x_axis[-1] - x_axis[0]) / (N - 1)
        w = np.ones(N) * h
        w[0] *= 0.5
        w[-1] *= 0.5
        return w

class LaplacianOperator2D:
    def __init__(self, x, y, differentation_type="finite-difference"):
        if differentation_type != "finite-difference":
            raise ValueError("Only 'finite-difference' method is implemented for 2D.")

        self.Nx, self.Ny = len(x), len(y)
        self.hx = (x[-1] - x[0]) / (self.Nx - 1)
        self.hy = (y[-1] - y[0]) / (self.Ny - 1)
        
        # Laplacian operator matrix (N*N, N*N)
        self.D2 = self._finite_diff_matrix_2d()

    def _finite_diff_matrix_1d(self, N, h):
        D2 = np.diag(np.ones(N - 1), -1) - 2 * np.eye(N) + np.diag(np.ones(N - 1), 1)
        return D2 / h**2

    def _finite_diff_matrix_2d(self):
        D2x = self._finite_diff_matrix_1d(self.Nx, self.hx)
        D2y = self._finite_diff_matrix_1d(self.Ny, self.hy)
        
        # Kronecker product to create the 2D operator
        Ix = identity(self.Nx)
        Iy = identity(self.Ny)
        
        # Return matrix as a dense NumPy array
        return (kron(D2x, Iy) + kron(Ix, D2y)).toarray()
