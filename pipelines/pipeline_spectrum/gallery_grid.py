import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
from kernel import GaussianKernel, LaplaceKernel, QuarticKernel
from discretization import ClenshawCurtisQuadrature
from galerkin import GalerkinBuilder, LegendrePolynomials

def plot_all_galleries():
    L = 5.0
    N_basis = 150
    N_quad = 1500
    top_k = 6
    
    # Ordered by exponent p: Laplace (p=1), Gaussian (p=2), Quartic (p=4)
    kernels = [
        ('Laplace', LaplaceKernel(1.0)),
        ('Gaussian', GaussianKernel(1.0)),
        ('Quartic', QuarticKernel(1.0))
    ]
    
    quad = ClenshawCurtisQuadrature()
    basis = LegendrePolynomials()
    
    fig, axes = plt.subplots(3, 6, figsize=(24, 12))
    
    for row_idx, (k_name, kernel) in enumerate(kernels):
        builder = GalerkinBuilder(kernel, quad)
        M, Phi, nodes = builder.build_matrix(basis, N_basis, L, N_quad=N_quad, preconditioner='cholesky')
        
        evals, evecs = scipy.linalg.eigh(M)
        efuncs = Phi @ evecs
        
        top_evals = evals[-top_k:]
        # Reverse to have largest eigenvalue on the left
        top_evals = top_evals[::-1] 
        top_efuncs = efuncs[:, -top_k:][:, ::-1]
        
        for col_idx in range(top_k):
            ax = axes[row_idx, col_idx]
            ax.plot(nodes, top_efuncs[:, col_idx], color='tab:blue', linewidth=2)
            if col_idx == 0:
                ax.set_ylabel(f"{k_name} Kernel\nEigenfunction", fontsize=14, fontweight='bold')
            if row_idx == 0:
                ax.set_title(f"Rank {col_idx+1}", fontsize=14)
            
            ax.set_xlabel("x")
            ax.text(0.05, 0.95, f"$\\beta$ = {top_evals[col_idx]:.4f}", 
                    transform=ax.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            ax.grid(True, alpha=0.3)
            
    fig.tight_layout()
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, 'eigenfunctions_grid.png'), dpi=200)
    print("Saved eigenfunctions_grid.png")

if __name__ == '__main__':
    plot_all_galleries()
