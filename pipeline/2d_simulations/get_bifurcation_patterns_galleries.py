import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.sparse import identity, kron

# ==============================================================================
# IMPORTS FROM YOUR LOCAL MODULES
# ==============================================================================
from utils import gauss_seidel, get_boundary_indices_2d

# ==============================================================================
# 1. DEFINITION OF CALIBRATED KERNELS (2D) - Variance = 1
# ==============================================================================
C_SUB_2D = 3.0 / np.pi
B_SUB_2D = np.sqrt(6.0)

C_SG_2D = 2.0 / (np.pi**2)
B_SG_2D = 1.0 / np.pi

def kernel_sub_gaussian_2D(r_matrix):
    """Fat Tails (Sub-Gaussian) in 2D"""
    return C_SUB_2D * np.exp(-B_SUB_2D * r_matrix)

def kernel_super_gaussian_2D(r_matrix):
    """Thin Tails (Super-Gaussian) in 2D"""
    return C_SG_2D * np.exp(-B_SG_2D * r_matrix**4)

# ==============================================================================
# 2. DEFINITION OF YOUR 2D OPERATORS (Adapted for the experiment)
# ==============================================================================

class IntegralOperator2D:
    def __init__(self, x, y, kernel_func, quadrature_type='trapezoidal'):
        self.Nx, self.Ny = len(x), len(y)
        self.N = self.Nx * self.Ny
        hx = x[1] - x[0]

        xx, yy = np.meshgrid(x, y, indexing='ij')
        x_flat, y_flat = xx.flatten(), yy.flatten()

        grid_x1, grid_x2 = np.meshgrid(x_flat, x_flat, indexing='ij')
        grid_y1, grid_y2 = np.meshgrid(y_flat, y_flat, indexing='ij')

        r_matrix = np.sqrt((grid_x1 - grid_x2)**2 + (grid_y1 - grid_y2)**2)
        kernel_matrix = kernel_func(r_matrix)

        if quadrature_type == 'trapezoidal':
            wx = self._trapezoidal_weights(x)
            wy = self._trapezoidal_weights(y)
            weights_2d = np.outer(wx, wy).flatten()
        else:
            raise ValueError("Only 'trapezoidal' method is implemented for 2D.")

        grid_inf = np.arange(-40*hx, 40*hx + hx/2, hx)
        X_inf, Y_inf = np.meshgrid(grid_inf, grid_inf)
        r_inf = np.sqrt(X_inf**2 + Y_inf**2)
        discrete_norm = np.sum(kernel_func(r_inf) * hx**2)
        
        kernel_matrix = kernel_matrix / discrete_norm
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
        self.D2 = self._finite_diff_matrix_2d()

    def _finite_diff_matrix_1d(self, N, h):
        D2 = np.diag(np.ones(N - 1), -1) - 2 * np.eye(N) + np.diag(np.ones(N - 1), 1)
        return D2 / h**2

    def _finite_diff_matrix_2d(self):
        D2x = self._finite_diff_matrix_1d(self.Nx, self.hx)
        D2y = self._finite_diff_matrix_1d(self.Ny, self.hy)
        Ix = identity(self.Nx)
        Iy = identity(self.Ny)
        return (kron(D2x, Iy) + kron(Ix, D2y)).toarray()

# ==============================================================================
# 3. PARAMETERS AND INITIALIZATION
# ==============================================================================
B = 0.45
A_max, A_min = 1.8, 0.1
d_u, d_v = 1.0, 50.0   
ht = 0.01
tol = 1e-2
max_iter_steady_state = 100000

L = 40.0
K_steps = 100   
N = 100         

total_size = N * N
x_domain = np.linspace(-L, L, N)
y_domain = np.linspace(-L, L, N)
boundary_indices = get_boundary_indices_2d(N, N)

A_targets =[1.7, 1.4, 1.1, 0.8, 0.5, 0.2]

print("Building Laplacian operator...")
laplacian_operator = LaplacianOperator2D(x_domain, y_domain)
D2_dense = laplacian_operator.D2

models_to_run =[
    {"name": "Local", "type": "local", "kernel": None, "file_suffix": "local"},
    {"name": "Non-local (Thin Tails)", "type": "integral", "kernel": kernel_super_gaussian_2D, "file_suffix": "thin_tails"},
    {"name": "Non-local (Fat Tails)", "type": "integral", "kernel": kernel_sub_gaussian_2D, "file_suffix": "fat_tails"}
]

A_values = np.linspace(A_max, A_min, K_steps)
saved_branches = {m["name"]:[] for m in models_to_run}

# ==============================================================================
# 4. MAIN COMPUTATION LOOP
# ==============================================================================

for model in models_to_run:
    print(f"\n{'='*60}\nStarting computations for model: {model['name']}\n{'='*60}")
    
    diff_matrix = np.eye(total_size) - ht * (d_v * D2_dense - np.eye(total_size))
    
    if model["type"] == "local":
        lhs_u_matrix = np.eye(total_size) - ht * ((d_u / 2.0) * D2_dense - B * np.eye(total_size))
    else:
        print("  -> Building integral operator matrix using your class...")
        integral_op = IntegralOperator2D(x_domain, y_domain, kernel_func=model["kernel"])
        lhs_u_matrix = np.eye(total_size) - ht * (d_u * integral_op.matrix - d_u * np.eye(total_size) - B * np.eye(total_size))

    for i in boundary_indices:
        lhs_u_matrix[i, :] = 0; lhs_u_matrix[i, i] = 1
        diff_matrix[i, :] = 0; diff_matrix[i, i] = 1

    u_old, v_old = None, None

    for idx, A in enumerate(tqdm(A_values, desc="Tracking branch over A")):
        if idx == 0:
            v_init_val = (A - np.sqrt(abs(A**2 - 4*B**2))) / 2
            u_init_val = (2 * B) / v_init_val
            
            u_old = np.full(total_size, u_init_val)
            v_old = np.full(total_size, v_init_val)
            
            u_old[boundary_indices] = 0.0
            v_old[boundary_indices] = 0.0

        for it in range(max_iter_steady_state):
            non_linear_term = (u_old**2) * v_old

            rhs_u = u_old + ht * non_linear_term
            rhs_v = v_old + ht * (A - non_linear_term)

            rhs_u[boundary_indices] = 0    
            rhs_v[boundary_indices] = 0    

            u_new = gauss_seidel(lhs_u_matrix, rhs_u, 1e-5, 10, u_old)
            v_new = gauss_seidel(diff_matrix, rhs_v, 1e-5, 10, v_old)

            if np.isnan(u_new).any() or np.isnan(v_new).any():
                tqdm.write(f"\nBLOWUP at A = {A:.3f}")
                break

            total_error = np.linalg.norm(u_new - u_old) + np.linalg.norm(v_new - v_old)
            
            if idx == 0:
                tqdm.write(f"A = {A:.3f} | Iteration {it}: Total Error = {total_error:.6e}")

            if total_error < tol:
                break
                
            u_old, v_old = u_new.copy(), v_new.copy()
            
        saved_branches[model["name"]].append((A, u_new.reshape((N, N)), v_new.reshape((N, N))))

# ==============================================================================
# 5. GENERATING AND SAVING 2x3 GALLERY PLOTS
# ==============================================================================
print("\nGenerating vegetation and water galleries...")

# DYNAMICZNA ŚCIEŻKA (Relatywna względem pliku skryptu)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Założenie: skrypt leży w repo/pipeline/2d_simulations/
project_root = os.path.abspath(os.path.join(current_dir, "..", "..")) 
output_dir = os.path.join(project_root, "data", "2d_simulations")
os.makedirs(output_dir, exist_ok=True)

extent =[-L, L, -L, L]
# Stały element nazwy pliku przechowujący wszystkie parametry
params_str = f"B_{B}_L_{L}_N_{N}_du_{d_u}_dv_{d_v}"

for model in models_to_run:
    model_name = model["name"]
    file_suffix = model["file_suffix"]
    branch_data = saved_branches[model_name]
    
    if not branch_data:
        print(f"No data to plot for model {model_name}")
        continue
        
    global_vmax_V = max([np.max(data[1]) for data in branch_data])
    if global_vmax_V <= 0: global_vmax_V = 0.1 
    
    global_vmax_W = max([np.max(data[2]) for data in branch_data])
    if global_vmax_W <= 0: global_vmax_W = 0.1 

    # --- PLOTTING VEGETATION (V) ---
    fig_V, axes_V = plt.subplots(2, 3, figsize=(15, 10))
    fig_V.suptitle(f"Vegetation density ($V$) profiles - {model_name}", fontsize=18)
    axes_V_flat = axes_V.flatten()
    
    for idx, target_A in enumerate(A_targets):
        ax = axes_V_flat[idx]
        closest_data = min(branch_data, key=lambda item: abs(item[0] - target_A))
        actual_A, V_matrix, _ = closest_data
            
        im_V = ax.imshow(V_matrix, extent=extent, cmap='summer_r', origin='lower', vmin=0, vmax=global_vmax_V)
        ax.set_title(f"A $\\approx$ {actual_A:.2f}")
        fig_V.colorbar(im_V, ax=ax, fraction=0.046, pad=0.04, label="Vegetation $V$")
        
        if idx >= 3: ax.set_xlabel("x")
        if idx % 3 == 0: ax.set_ylabel("y")
            
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Zapis do poprawnego folderu z parametrami w nazwie
    filename_V = f"gallery_V_{file_suffix}_{params_str}.png"
    file_path_V = os.path.join(output_dir, filename_V)
    plt.savefig(file_path_V, dpi=150)
    print(f"Successfully saved: {file_path_V}")
    plt.close(fig_V)

    # --- PLOTTING WATER (W) ---
    fig_W, axes_W = plt.subplots(2, 3, figsize=(15, 10))
    fig_W.suptitle(f"Water density ($W$) profiles - {model_name}", fontsize=18)
    axes_W_flat = axes_W.flatten()
    
    for idx, target_A in enumerate(A_targets):
        ax = axes_W_flat[idx]
        closest_data = min(branch_data, key=lambda item: abs(item[0] - target_A))
        actual_A, _, W_matrix = closest_data
            
        im_W = ax.imshow(W_matrix, extent=extent, cmap='Blues', origin='lower', vmin=0, vmax=global_vmax_W)
        ax.set_title(f"A $\\approx$ {actual_A:.2f}")
        fig_W.colorbar(im_W, ax=ax, fraction=0.046, pad=0.04, label="Water $W$")
        
        if idx >= 3: ax.set_xlabel("x")
        if idx % 3 == 0: ax.set_ylabel("y")
            
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    filename_W = f"gallery_W_{file_suffix}_{params_str}.png"
    file_path_W = os.path.join(output_dir, filename_W)
    plt.savefig(file_path_W, dpi=150)
    print(f"Successfully saved: {file_path_W}")
    plt.close(fig_W)

print("\nDone. All computations and plots finished.")
