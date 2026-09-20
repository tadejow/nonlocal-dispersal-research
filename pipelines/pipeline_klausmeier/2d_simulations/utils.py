import numpy as np


def gauss_seidel(A, b, tolerance, max_iterations, x_init):
    """
    Poprawiona wersja metody Gaussa-Seidla, która unika DeprecationWarning.
    """
    x = x_init.copy()
    n = A.shape[0]
    for _ in range(max_iterations):
        x_old = x.copy()
        for i in range(n):
            if np.abs(A[i, i]) > 1e-12:
                sigma1 = np.dot(A[i, :i], x[:i])
                sigma2 = np.dot(A[i, i+1:], x_old[i+1:])
                x[i] = float((b[i] - sigma1 - sigma2) / A[i, i])

        if np.linalg.norm(x - x_old, np.inf) < tolerance:
            break
    return x

def get_boundary_indices_2d(Nx, Ny):
    """Zwraca indeksy brzegowe dla spłaszczonej siatki Nx x Ny."""
    mask = np.zeros((Nx, Ny), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return np.where(mask.flatten())[0]
