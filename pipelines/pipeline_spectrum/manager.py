import argparse
import numpy as np
import scipy.linalg
import time

from kernel import GaussianKernel, LaplaceKernel, QuarticKernel
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

def get_kernel(name, sigma):
    if name == 'gaussian':
        return GaussianKernel(sigma=sigma)
    elif name == 'laplace':
        return LaplaceKernel(sigma=sigma)
    elif name == 'quartic':
        return QuarticKernel(sigma=sigma)
    else:
        raise ValueError(f"Unknown kernel: {name}")

class SpectrumManager:
    def __init__(self, args):
        self.args = args
        self.kernel = get_kernel(args.kernel, args.sigma)
        self.quadrature = get_quadrature(args.quadrature)
        self.basis = get_basis(args.basis)
        self.builder = GalerkinBuilder(self.kernel, self.quadrature)
        self.plotter = SpectrumPlotter(output_dir=args.output)

    def solve(self, L):
        M, Phi, nodes = self.builder.build_matrix(
            self.basis, 
            self.args.N_basis, 
            L, 
            N_quad=self.args.N_quad,
            preconditioner=self.args.preconditioner
        )
        # Compute min b(x) for essential spectrum bound
        # M = P^T W J W P - P^T W B P. We can just compute b(x) directly.
        J_mat = self.kernel(nodes[:, None] - nodes[None, :])
        _, weights = self.quadrature.get_nodes_and_weights(L, self.args.N_quad)
        b_vals = J_mat @ weights
        ess_bound = -np.min(b_vals)

        evals, evecs = scipy.linalg.eigh(M)
        efuncs_eval = Phi @ evecs
        return evals, efuncs_eval, nodes, ess_bound, M

    def run_single(self):
        print_header("SINGLE EXPERIMENT")
        print_params(
            Basis=self.args.basis.capitalize(),
            Quadrature=self.args.quadrature.capitalize(),
            Kernel=self.args.kernel.capitalize(),
            Preconditioner=self.args.preconditioner,
            Domain_L=self.args.L,
            N_basis=self.args.N_basis,
            N_quad=self.args.N_quad,
            Kernel_Sigma=self.args.sigma
        )
        
        start_t = time.time()
        evals, efuncs, nodes, ess_bound, _ = self.solve(self.args.L)
        elapsed = time.time() - start_t
        
        filename = f"gallery_L_{self.args.L}_{self.args.kernel}.png"
        self.plotter.plot_eigenfunctions_gallery(evals, efuncs, nodes, filename=filename)
        
        print(" [Results]")
        print(f" Ess Bound (max)   : {ess_bound:.6f}")
        print(f" Top 6 Eigenvalues : {np.round(evals[-6:][::-1], 6)}")
        print(f" Computation Time  : {elapsed:.3f} s")
        print(f" Output Saved To   : {self.args.output}/{filename}\n")

    def run_sweep(self):
        print_header("SWEEP EXPERIMENT (Beta vs L)")
        print_params(
            Basis=self.args.basis.capitalize(),
            Quadrature=self.args.quadrature.capitalize(),
            Kernel=self.args.kernel.capitalize(),
            Preconditioner=self.args.preconditioner,
            L_Range=f"[{self.args.L_min}, {self.args.L_max}] (steps: {self.args.num_L})",
            N_basis=self.args.N_basis,
            N_quad=self.args.N_quad,
            Kernel_Sigma=self.args.sigma
        )
        
        L_vals = np.linspace(self.args.L_min, self.args.L_max, self.args.num_L)
        beta_vals = []
        ess_bounds = []
        
        start_t = time.time()
        for i, L in enumerate(L_vals):
            print(f" Solving for L = {L:5.2f}  ({i+1}/{self.args.num_L})...", end="\r")
            evals, _, _, ess_b, _ = self.solve(L)
            beta_vals.append(evals)
            ess_bounds.append(ess_b)
        
        elapsed = time.time() - start_t
        filename = f"beta_vs_L_{self.args.basis}_{self.args.kernel}.png"
        
        self.plotter.plot_beta_vs_L(L_vals, beta_vals, ess_bounds=ess_bounds, top_k=self.args.top_k, filename=filename)
        
        print(f"\n\n [Results]")
        print(f" Computation Time  : {elapsed:.3f} s")
        print(f" Output Saved To   : {self.args.output}/{filename}\n")


def main():
    parser = argparse.ArgumentParser(
        description="CLI Manager for Non-Local Dispersal Spectrum Numerical Simulations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('mode', choices=['single', 'sweep'], help="Experiment mode")
    
    algo_group = parser.add_argument_group('Algorithm Settings')
    algo_group.add_argument('-b', '--basis', type=str, choices=['laplacian', 'legendre', 'canonical'], default='laplacian')
    algo_group.add_argument('-q', '--quadrature', type=str, choices=['trapezoidal', 'simpson', 'clenshaw-curtis'], default='simpson')
    algo_group.add_argument('-K', '--kernel', type=str, choices=['gaussian', 'laplace', 'quartic'], default='gaussian')
    algo_group.add_argument('-p', '--preconditioner', type=str, choices=['none', 'cholesky', 'lowdin'], default='none')
    algo_group.add_argument('-n', '--N_basis', type=int, default=250)
    algo_group.add_argument('-k', '--N_quad', type=int, default=1500)
    algo_group.add_argument('-s', '--sigma', type=float, default=1.0)
    
    single_group = parser.add_argument_group('Single Mode Parameters')
    single_group.add_argument('-L', type=float, default=5.0)
    
    sweep_group = parser.add_argument_group('Sweep Mode Parameters')
    sweep_group.add_argument('--L_min', type=float, default=0.2)
    sweep_group.add_argument('--L_max', type=float, default=10.0)
    sweep_group.add_argument('--num_L', type=int, default=30)
    sweep_group.add_argument('--top_k', type=int, default=100)
    
    out_group = parser.add_argument_group('Output Settings')
    out_group.add_argument('-o', '--output', type=str, default='output')

    args = parser.parse_args()
    manager = SpectrumManager(args)
    
    if args.mode == 'single':
        manager.run_single()
    elif args.mode == 'sweep':
        manager.run_sweep()

if __name__ == '__main__':
    main()
