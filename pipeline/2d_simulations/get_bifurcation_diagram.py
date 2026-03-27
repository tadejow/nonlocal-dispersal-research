import os
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# IMPORTS FROM LOCAL MODULES
from utils import gauss_seidel, get_boundary_indices_2d
from operators import IntegralOperator2D, LaplacianOperator2D

# ==============================================================================
# 1. KERNEL DEFINITIONS
# ==============================================================================
C_SUB_2D = 3.0 / np.pi
B_SUB_2D = np.sqrt(6.0)
C_SG_2D = 2.0 / (np.pi**2)
B_SG_2D = 1.0 / np.pi

def kernel_sub_gaussian_2D(r_matrix):
    return C_SUB_2D * np.exp(-B_SUB_2D * r_matrix)

def kernel_super_gaussian_2D(r_matrix):
    return C_SG_2D * np.exp(-B_SG_2D * r_matrix**4)

# ==============================================================================
# 2. DYNAMIC PATH SETUP & FILE SELECTION
# ==============================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", "..")) 
configs_dir = os.path.join(project_root, "configs")
output_dir = os.path.join(project_root, "data", "2d_simulations")

os.makedirs(configs_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# List available configs
json_files = sorted([f for f in os.listdir(configs_dir) if f.endswith('.json')])

if not json_files:
    print(f"No .json config files found in {configs_dir}.")
    print("Please run 'generate_configs.py' first.")
    exit(1)

print("\nAvailable configuration files:")
print("-" * 50)
for idx, file_name in enumerate(json_files):
    print(f"[{idx + 1}] {file_name}")
print("-" * 50)

try:
    selection = int(input("\nSelect the config file number to run: ")) - 1
    if selection < 0 or selection >= len(json_files):
        raise ValueError
except ValueError:
    print("Invalid selection. Exiting.")
    exit(1)

selected_file = json_files[selection]
config_path = os.path.join(configs_dir, selected_file)

# ==============================================================================
# 3. LOAD PARAMETERS FROM JSON
# ==============================================================================
print(f"\nLoading parameters from {selected_file}...")
with open(config_path, 'r') as f:
    params = json.load(f)

# Extract parameters
A_max = params["A_max"]
A_min = params["A_min"]
B = params["B"]
d_u = params["d_u"]
d_v = params["d_v"]
L = params["L"]
N = params["N"]
K_steps = params["K_steps"]
ht = params["ht"]
tol = params["tol"]
max_iter = params["max_iter"]

params_str = f"B_{B}_L_{L}_N_{N}_du_{d_u}_dv_{d_v}"

print(f"Parameters loaded: N={N}, L={L}, d_u={d_u}, d_v={d_v}, B={B}")

# ==============================================================================
# 4. INITIALIZATION FOR COMPUTATIONS
# ==============================================================================
total_size = N * N
x_domain = np.linspace(-L, L, N)
y_domain = np.linspace(-L, L, N)
boundary_indices = get_boundary_indices_2d(N, N)

print("\nBuilding Laplacian operator...")
laplacian_operator = LaplacianOperator2D(x_domain, y_domain)
D2_dense = laplacian_operator.D2

models_to_run =[
    {"name": "Local", "type": "local", "kernel": None},
    {"name": "Non-local (Thin Tails)", "type": "integral", "kernel": kernel_super_gaussian_2D},
    {"name": "Non-local (Fat Tails)", "type": "integral", "kernel": kernel_sub_gaussian_2D}
]

A_values = np.linspace(A_max, A_min, K_steps)
# branch_results[model_name] = lists of (A, avg_u, max_u)
branch_results = {m["name"]:[] for m in models_to_run}

# ==============================================================================
# 5. MAIN COMPUTATION LOOP
# ==============================================================================
for model in models_to_run:
    print(f"\n{'='*60}\nStarting computations for model: {model['name']}\n{'='*60}")
    
    diff_matrix = np.eye(total_size) - ht * (d_v * D2_dense - np.eye(total_size))
    
    if model["type"] == "local":
        lhs_u_matrix = np.eye(total_size) - ht * ((d_u / 2.0) * D2_dense - B * np.eye(total_size))
    else:
        print("  -> Building integral operator matrix...")
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

        for it in range(max_iter):
            non_linear_term = (u_old**2) * v_old

            rhs_u = u_old + ht * non_linear_term
            rhs_v = v_old + ht * (A - non_linear_term)

            rhs_u[boundary_indices] = 0    
            rhs_v[boundary_indices] = 0    

            u_new = gauss_seidel(lhs_u_matrix, rhs_u, 1e-5, 10, u_old)
            v_new = gauss_seidel(diff_matrix, rhs_v, 1e-5, 10, v_old)

            if np.isnan(u_new).any() or np.isnan(v_new).any():
                tqdm.write(f"BLOWUP at A = {A:.3f}")
                break

            total_error = np.linalg.norm(u_new - u_old) + np.linalg.norm(v_new - v_old)
            if total_error < tol:
                avg_u = u_new.mean()
                max_u = np.max(u_new)
                branch_results[model["name"]].append((A, avg_u, max_u))
                break
                
            u_old, v_old = u_new.copy(), v_new.copy()

# ==============================================================================
# 6. PLOTTING THE COMBINED BIFURCATION DIAGRAM
# ==============================================================================
print("\nGenerating combined bifurcation diagram...")

plt.figure(figsize=(10, 7))

# Plot settings for each model
styles = {
    "Local": {"color": "blue", "ls": "--", "marker": "o"},
    "Non-local (Thin Tails)": {"color": "cyan", "ls": "-", "marker": "s"},
    "Non-local (Fat Tails)": {"color": "orange", "ls": "-", "marker": "^"}
}

for model_name, data in branch_results.items():
    if not data:
        continue
    
    A_arr = [item[0] for item in data]
    avg_u_arr = [item[1] for item in data]
    max_u_arr = [item[2] for item in data]
    c = styles[model_name]["color"]
    ls = styles[model_name]["ls"]
    m = styles[model_name]["marker"]
    
    plt.plot(A_arr, avg_u_arr, color=c, linestyle=ls, marker=m, markersize=4, label=f'Avg V - {model_name}')
    plt.plot(A_arr, max_u_arr, color=c, linestyle=ls, marker=m, markersize=4, alpha=0.5, label=f'Max V - {model_name}')

# Theoretical critical level ~ B/A
critical_level_values = [B / A for A in A_values]
plt.plot(A_values, critical_level_values, color='red', linestyle=':', linewidth=2, label='Critical Level (~B/A)')
plt.axvline(x=2*B, color='purple', linestyle='--', linewidth=2, label='A = 2B')

plt.xlabel('Rainfall (A)', fontsize=14)
plt.ylabel('Biomass Density (V)', fontsize=14)
plt.title(f'Desertification Bifurcation Diagram (2D)\nParameters: B={B}, L={L}, N={N}, dv={d_u}, dw={d_v}', fontsize=12)
plt.gca().invert_xaxis()
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=10, loc='best')

plt.tight_layout()

# Save the diagram
output_png_path = os.path.join(output_dir, f"bifurcation_diagram_2D_{params_str}.png")
print(f"Saving diagram to: {output_png_path}")
plt.savefig(output_png_path, dpi=150)
plt.show()

print("\nDone.")
