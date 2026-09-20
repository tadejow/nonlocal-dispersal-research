import argparse
import numpy as np
import scipy.linalg
import time

from kernel import GaussianKernel
from discretization import TrapezoidalQuadrature, SimpsonQuadrature, ClenshawCurtisQuadrature
from galerkin import GalerkinBuilder, LaplacianEigenfunctions, LegendrePolynomials, CanonicalBasis
from plotting import SpectrumPlotter

def print_header(title):
    print("\n" + "="*50)
    print(f" {title:^48} ")
    print("="*50)

def print_params(**kwargs):
    for k, v in kwargs.items():
        print(f" {k:<15} : {v}")
    print("-"*50)

def get_basis(name):
    bases = {
        'laplacian': LaplacianEigenfunctions(),
        'legendre': LegendrePolynomials(),
        'canonical': CanonicalBasis()
    }
    return bases[name]

def get_quadrature(name):
    quadratures = {
        'trapezoidal': TrapezoidalQuadrature(),
        'simpson': SimpsonQuadrature(),
        'clenshaw-curtis': ClenshawCurtisQuadrature()
    }
    return quadratures[name]

class SpectrumManager:
    def __init__(self, args):
        self.args = args
        self.kernel = GaussianKernel(sigma=args.sigma)
        self.quadrature = get_quadrature(args.quadrature)
        self.basis = get_basis(args.basis)
        self.builder = GalerkinBuilder(self.kernel, self.quadrature)
        self.plotter = SpectrumPlotter(output_dir=args.output)

    def solve(self, L):
        M, Phi, nodes = self.builder.build_matrix(
            self.basis, 
            self.args.N_basis, 
            L, 
            N_quad=self.args.N_quad
        )
        evals, evecs = scipy.linalg.eigh(M)
        efuncs_eval = Phi @ evecs
        return evals, efuncs_eval, nodes

    def run_single(self):
        print_header("SINGLE EXPERIMENT")
        print_params(
            Basis=self.args.basis.capitalize(),
            Quadrature=self.args.quadrature.capitalize(),
            Domain_L=self.args.L,
            N_basis=self.args.N_basis,
            N_quad=self.args.N_quad,
            Kernel_Sigma=self.args.sigma
        )
        
        start_t = time.time()
        evals, efuncs, nodes = self.solve(self.args.L)
        elapsed = time.time() - start_t
        
        filename = f"gallery_L_{self.args.L}.png"
        self.plotter.plot_eigenfunctions_gallery(evals, efuncs, nodes, filename=filename)
        
        print(" [Results]")
        print(f" Top 6 Eigenvalues : {np.round(evals[-6:][::-1], 6)}")
        print(f" Computation Time  : {elapsed:.3f} s")
        print(f" Output Saved To   : {self.args.output}/{filename}\n")

    def run_sweep(self):
        print_header("SWEEP EXPERIMENT (Beta vs L)")
        print_params(
            Basis=self.args.basis.capitalize(),
            Quadrature=self.args.quadrature.capitalize(),
            L_Range=f"[{self.args.L_min}, {self.args.L_max}] (steps: {self.args.num_L})",
            N_basis=self.args.N_basis,
            N_quad=self.args.N_quad,
            Kernel_Sigma=self.args.sigma
        )
        
        L_vals = np.linspace(self.args.L_min, self.args.L_max, self.args.num_L)
        beta_vals = []
        
        start_t = time.time()
        for i, L in enumerate(L_vals):
            print(f" Solving for L = {L:5.2f}  ({i+1}/{self.args.num_L})...", end="\r")
            evals, _, _ = self.solve(L)
            beta_vals.append(evals)
        
        elapsed = time.time() - start_t
        filename = f"beta_vs_L_{self.args.basis}.png"
        self.plotter.plot_beta_vs_L(L_vals, beta_vals, top_k=self.args.top_k, filename=filename)
        
        print(f"\n\n [Results]")
        print(f" Computation Time  : {elapsed:.3f} s")
        print(f" Output Saved To   : {self.args.output}/{filename}\n")


def main():
    parser = argparse.ArgumentParser(
        description="CLI Manager for Non-Local Dispersal Spectrum Numerical Simulations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Mode selection
    parser.add_argument('mode', choices=['single', 'sweep'], help="Experiment mode: 'single' (one domain size) or 'sweep' (range of domain sizes)")
    
    # Main algorithm settings
    algo_group = parser.add_argument_group('Algorithm Settings')
    algo_group.add_argument('-b', '--basis', type=str, choices=['laplacian', 'legendre', 'canonical'], default='laplacian', help="Basis functions for Galerkin projection")
    algo_group.add_argument('-q', '--quadrature', type=str, choices=['trapezoidal', 'simpson', 'clenshaw-curtis'], default='simpson', help="Numerical integration method")
    algo_group.add_argument('-n', '--N_basis', type=int, default=30, help="Number of basis functions")
    algo_group.add_argument('-k', '--N_quad', type=int, default=500, help="Number of quadrature points")
    algo_group.add_argument('-s', '--sigma', type=float, default=1.0, help="Kernel width (sigma)")
    
    # Single mode specific
    single_group = parser.add_argument_group('Single Mode Parameters')
    single_group.add_argument('-L', type=float, default=5.0, help="Domain half-width (used in 'single' mode)")
    
    # Sweep mode specific
    sweep_group = parser.add_argument_group('Sweep Mode Parameters')
    sweep_group.add_argument('--L_min', type=float, default=0.5, help="Minimum domain half-width")
    sweep_group.add_argument('--L_max', type=float, default=10.0, help="Maximum domain half-width")
    sweep_group.add_argument('--num_L', type=int, default=20, help="Number of points in sweep")
    sweep_group.add_argument('--top_k', type=int, default=6, help="Number of principal eigenvalues to plot")
    
    # Output settings
    out_group = parser.add_argument_group('Output Settings')
    out_group.add_argument('-o', '--output', type=str, default='output', help="Directory to save generated plots")

    args = parser.parse_args()
    manager = SpectrumManager(args)
    
    if args.mode == 'single':
        manager.run_single()
    elif args.mode == 'sweep':
        manager.run_sweep()

if __name__ == '__main__':
    main()
