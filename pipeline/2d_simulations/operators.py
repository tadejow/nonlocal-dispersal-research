import numpy as np
from scipy.sparse import identity, kron

class IntegralOperator2D:
    def __init__(self, x, y, quadrature_type='trapezoidal'):
        self.Nx, self.Ny = len(x), len(y)
        self.N = self.Nx * self.Ny

        # Tworzenie siatki 2D i jej spłaszczenie do wektorów 1D
        xx, yy = np.meshgrid(x, y, indexing='ij')
        x_flat, y_flat = xx.flatten(), yy.flatten()

        # Siatka dla jądra całkowego (wszystkie kombinacje par punktów)
        grid_x1, grid_x2 = np.meshgrid(x_flat, x_flat, indexing='ij')
        grid_y1, grid_y2 = np.meshgrid(y_flat, y_flat, indexing='ij')

        # Jądro Gaussa 2D
        kernel_func = lambda x1, y1, x2, y2: (1 / np.pi) * np.exp(-((x1-x2)**2 + (y1-y2)**2))
        kernel_matrix = kernel_func(grid_x1, grid_y1, grid_x2, grid_y2)

        if quadrature_type == 'trapezoidal':
            wx = self._trapezoidal_weights(x)
            wy = self._trapezoidal_weights(y)
            weights_2d = np.outer(wx, wy).flatten()
        else:
            raise ValueError("Tylko metoda 'trapezoidal' jest zaimplementowana dla 2D.")

        # Macierz operatora całkowego (N*N, N*N)
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
            raise ValueError("Tylko metoda 'finite-difference' jest zaimplementowana dla 2D.")

        self.Nx, self.Ny = len(x), len(y)
        self.hx = (x[-1] - x[0]) / (self.Nx - 1)
        self.hy = (y[-1] - y[0]) / (self.Ny - 1)
        # Macierz operatora Laplace'a (N*N, N*N)
        self.D2 = self._finite_diff_matrix_2d()

    def _finite_diff_matrix_1d(self, N, h):
        D2 = np.diag(np.ones(N - 1), -1) - 2 * np.eye(N) + np.diag(np.ones(N - 1), 1)
        return D2 / h**2

    def _finite_diff_matrix_2d(self):
        D2x = self._finite_diff_matrix_1d(self.Nx, self.hx)
        D2y = self._finite_diff_matrix_1d(self.Ny, self.hy)
        # Iloczyn Kroneckera do stworzenia operatora 2D
        Ix = identity(self.Nx)
        Iy = identity(self.Ny)
        # Zwracamy macierz jako gęstą tablicę NumPy
        return (kron(D2x, Iy) + kron(Ix, D2y)).toarray()