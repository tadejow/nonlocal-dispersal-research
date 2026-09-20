import os
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
import warnings
from collections import defaultdict

from kernel import GaussianKernel
from discretization import TrapezoidalQuadrature, SimpsonQuadrature, ClenshawCurtisQuadrature
from galerkin import GalerkinBuilder, LaplacianEigenfunctions, LegendrePolynomials, CanonicalBasis

warnings.filterwarnings('ignore')

def get_basis(name):
    return {'laplacian': LaplacianEigenfunctions(), 'legendre': LegendrePolynomials(), 'canonical': CanonicalBasis()}[name]

def get_quadrature(name):
    return {'trapezoidal': TrapezoidalQuadrature(), 'simpson': SimpsonQuadrature(), 'clenshaw-curtis': ClenshawCurtisQuadrature()}[name]

def run_validation():
    L = 5.0
    sigma = 1.0
    kernel = GaussianKernel(sigma)
    
    # Zostawiamy tylko bazę legendre
    bases = ['legendre']
    quads = ['simpson', 'clenshaw-curtis', 'trapezoidal']
    preconds = ['none', 'cholesky', 'lowdin']
    N_vals = [20, 40, 60, 80, 100, 150]
    N_quad_test = 1500
    N_quad_ref = 2500
    
    print("Computing reference solution...")
    ref_quad = get_quadrature('clenshaw-curtis')
    ref_nodes, ref_weights = ref_quad.get_nodes_and_weights(L, N_quad_ref)
    
    ref_builder = GalerkinBuilder(kernel, ref_quad)
    M_ref, Phi_ref, _ = ref_builder.build_matrix(get_basis('legendre'), 300, L, N_quad=N_quad_ref, preconditioner='cholesky')
    ref_evals, ref_evecs = scipy.linalg.eigh(M_ref)
    ref_efuncs = Phi_ref @ ref_evecs
    
    J_mat_ref = kernel(ref_nodes[:, None] - ref_nodes[None, :])
    b_vals_ref = J_mat_ref @ ref_weights
    ess_bound = -np.min(b_vals_ref)
    
    discrete_mask = ref_evals > ess_bound + 1e-4
    ref_discrete_evals = ref_evals[discrete_mask]
    num_discrete = len(ref_discrete_evals)
    top_k = min(num_discrete, 20)
    ref_top_evals = ref_evals[-top_k:]
    
    results = []
    
    for b in bases:
        for q in quads:
            for p in preconds:
                print(f"Testing {b} + {q} + {p}...")
                builder = GalerkinBuilder(kernel, get_quadrature(q))
                for N_basis in N_vals:
                    start_t = time.time()
                    basis_obj = get_basis(b)
                    
                    try:
                        M, Phi_test, test_nodes = builder.build_matrix(basis_obj, N_basis, L, N_quad=N_quad_test, preconditioner=p)
                        evals, evecs = scipy.linalg.eigh(M)
                        cond = np.linalg.cond(M - np.eye(M.shape[0]))
                        
                        test_top_evals = evals[-top_k:]
                        err_evals = np.sqrt(np.mean((test_top_evals - ref_top_evals)**2))
                        
                        efuncs_test = Phi_test @ evecs
                        err_efuncs = 0.0
                        
                        # FIX dla Clenshaw-Curtis: węzły mogą być nieposortowane rosnąco,
                        # co psuje całkowicie interpolację np.interp (wymaga ściśle rosnących xp).
                        sort_idx = np.argsort(test_nodes)
                        test_nodes_sorted = test_nodes[sort_idx]
                        
                        for k in range(1, top_k + 1):
                            ref_f = ref_efuncs[:, -k]
                            
                            # Interpolacja na posortowanych wezłach testowych
                            test_f_sorted = efuncs_test[sort_idx, -k]
                            test_f = np.interp(ref_nodes, test_nodes_sorted, test_f_sorted)
                            
                            ref_norm = np.sqrt(np.sum(ref_weights * ref_f**2))
                            test_norm = np.sqrt(np.sum(ref_weights * test_f**2))
                            
                            if ref_norm > 0: ref_f = ref_f / ref_norm
                            if test_norm > 0: test_f = test_f / test_norm
                            
                            if np.dot(ref_f, test_f) < 0:
                                test_f = -test_f
                                
                            diff = ref_f - test_f
                            err_L2 = np.sqrt(np.sum(ref_weights * diff**2))
                            err_efuncs += err_L2
                            
                        err_efuncs /= top_k
                    except Exception as e:
                        print(f"Failed {b}+{q}+{p} at N={N_basis}")
                        err_evals, err_efuncs, cond = np.nan, np.nan, np.nan
                    
                    elapsed = time.time() - start_t
                    results.append({
                        'basis': b, 'quad': q, 'precond': p, 'N': N_basis,
                        'time': elapsed, 'cond': cond, 'err_evals': err_evals, 'err_efuncs': err_efuncs
                    })
    
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(out_dir, exist_ok=True)
            
    # Convergence Plots: 3x3 Grid
    fig_conv, axes = plt.subplots(3, 3, figsize=(20, 15), sharey='col', sharex=True)
    quad_colors = {'simpson': 'tab:blue', 'clenshaw-curtis': 'tab:orange', 'trapezoidal': 'tab:green'}
    metrics = ['err_evals', 'err_efuncs', 'cond']
    titles = ['Eigenvalue RMSE (Discrete Spectrum)', 'Eigenfunction L2 Error (Discrete Spectrum)', 'Condition Number of Shifted (M - I)']
    
    for r, p in enumerate(preconds):
        for c, metric in enumerate(metrics):
            ax = axes[r, c]
            for q in quads:
                metric_data = []
                for b in bases:
                    row = [res[metric] for res in results if res['basis']==b and res['quad']==q and res['precond']==p]
                    metric_data.append(row)
                metric_data = np.array(metric_data, dtype=float)
                
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    val = metric_data[0]
                
                color = quad_colors[q]
                ax.plot(N_vals, val, color=color, label=f"{q}", marker='o', linewidth=2)
                
            ax.set_yscale('log')
            ax.tick_params(labelbottom=True, labelleft=True)
            if r == 2: ax.set_xlabel('N')
            ax.grid(True, alpha=0.3)
            
            if r == 0: ax.set_title(titles[c], fontsize=14)
            if c == 0: ax.set_ylabel(f"Precond: {p.upper()}\n\n{titles[c].split(' (')[0]}", fontsize=12)
            if r == 0 and c == 2: ax.legend(loc='upper right')
            
    fig_conv.tight_layout()
    fig_conv.savefig(os.path.join(out_dir, 'validation_convergence.png'), dpi=200)
    plt.close(fig_conv)

    # Boxplots (Optional cleanup for single basis)
    fig_box, axes_box = plt.subplots(3, 3, figsize=(24, 18), sharey='col', sharex=True)
    for r, p in enumerate(preconds):
        for c, metric in enumerate(metrics):
            ax = axes_box[r, c]
            labels = []
            data = []
            for b in bases:
                for q in quads:
                    labels.append(f"{q[:4]}")
                    d = [res[metric] for res in results if res['basis']==b and res['quad']==q and res['precond']==p]
                    d = [v for v in d if not np.isnan(v)]
                    data.append(d if d else [np.nan])
                    
            ax.boxplot(data, tick_labels=labels)
            ax.set_yscale('log')
            ax.tick_params(labelbottom=True, labelleft=True)
            ax.grid(True, alpha=0.3)
            
            if r == 0: ax.set_title(titles[c], fontsize=14)
            if c == 0: ax.set_ylabel(f"Precond: {p.upper()}\n\n{titles[c].split(' (')[0]}", fontsize=12)
            
    fig_box.tight_layout()
    fig_box.savefig(os.path.join(out_dir, 'validation_boxplots.png'), dpi=200)
    plt.close(fig_box)

if __name__ == '__main__':
    run_validation()
