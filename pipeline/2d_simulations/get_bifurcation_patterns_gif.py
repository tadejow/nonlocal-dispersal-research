import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm

from utils import gauss_seidel, get_boundary_indices_2d
from operators import IntegralOperator2D, LaplacianOperator2D

# ==============================================================================
# PARAMETERS
# ==============================================================================
# Hardcoded to allow execution on a remote server without human interaction
B = 0.45
A_max, A_min = 1.8, 0.1
d_u, d_v = 1.0, 10.0
ht = 0.01
tol = 1e-2
max_iter_steady_state = 100000

L = 25.0
K = 100  # grid size for rainfall bifurcation steps
N = 100  # grid size for the spatial domain (N x N)

total_size = N * N
x_domain = np.linspace(-L, L, N)
y_domain = np.linspace(-L, L, N)

# ==============================================================================
# OUTPUT PATH SETUP
# ==============================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", "..")) 
output_dir = os.path.join(project_root, "data", "2d_simulations")
os.makedirs(output_dir, exist_ok=True)

params_str = f"B_{B}_L_{L}_N_{N}_du_{d_u}_dv_{d_v}"
output_gif_path = os.path.join(output_dir, f"bifurcation_animation_{params_str}.gif")

# ==============================================================================
# COMPUTATION
# ==============================================================================
print("Initialization of the operators 2D...")
integral_operator = IntegralOperator2D(x_domain, y_domain)
laplacian_operator = LaplacianOperator2D(x_domain, y_domain)

print("Construction of the discretized matrices...")
integral_matrix = np.eye(total_size) - ht * (d_u * integral_operator.matrix - d_u * np.eye(total_size) - B * np.eye(total_size))
diff_matrix = np.eye(total_size) - ht * (d_v * laplacian_operator.D2 - np.eye(total_size))

boundary_indices = get_boundary_indices_2d(N, N)

for i in boundary_indices:
    integral_matrix[i, :] = 0; integral_matrix[i, i] = 1
    diff_matrix[i, :] = 0; diff_matrix[i, i] = 1

A_values = np.linspace(A_max, A_min, K)
branch_results = {A:[] for A in A_values}

u_init, v_init = None, None

for idx, A in enumerate(tqdm(A_values, desc="Bifurcation continuation")):
    if idx == 0:
        v_init_val = (A - np.sqrt(abs(A**2 - 4*B**2))) / 2
        u_init_val = (2 * B) / v_init_val
        v_init = np.full(N**2, v_init_val)
        u_init = np.full(N**2, u_init_val)
    else:
        prev_A = A_values[idx - 1]
        if branch_results[prev_A]:
            u_init = branch_results[prev_A][0][1].copy()
            v_init = branch_results[prev_A][0][2].copy()
        else:
            continue

    u_old = u_init.copy()
    v_old = v_init.copy()

    for it in range(max_iter_steady_state):
        non_linear_term = (u_old**2) * v_old

        rhs_u = u_old + ht * non_linear_term
        rhs_u[boundary_indices] = 0    

        rhs_v = v_old + ht * (A - non_linear_term)
        rhs_v[boundary_indices] = 0    

        u_new = gauss_seidel(integral_matrix, rhs_u, 1e-5, 10, u_old)
        v_new = gauss_seidel(diff_matrix, rhs_v, 1e-5, 10, v_old)

        if np.isnan(u_new).any() or np.isnan(v_new).any():
            tqdm.write(f"A = {A:.3f} | iter = {it} - BLOWUP")
            break
            
        total_error = np.linalg.norm(u_new - u_old) + np.linalg.norm(v_new - v_old)
        
        if total_error < tol:
            biomass = u_new.mean()
            branch_results[A].append((biomass, u_new.copy(), v_new.copy()))
            tqdm.write(f"A = {A:.3f} | biomass = {biomass:.4f} | iter = {it} | Converged!")
            break

        u_old, v_old = u_new.copy(), v_new.copy()

        if A == A_values[0] and it == 0:
             tqdm.write(f"Initial Error: {total_error}")

        if it == max_iter_steady_state - 1:
            tqdm.write(f"A = {A:.3f} | iter = {it} - MAX ITERATIONS REACHED")
            biomass = u_new.mean()
            branch_results[A].append((biomass, u_new.copy(), v_new.copy()))

# ==============================================================================
# ANIMATION
# ==============================================================================
animation_data =[]
for A in A_values:
    if branch_results[A]:
        biomass, u, v = branch_results[A][0]
        if not np.isnan(biomass):
            animation_data.append({'A': A, 'biomass': biomass, 'u': u.reshape((N, N)), 'v': v.reshape((N, N))})

if not animation_data:
    print("No numerical data was generated in this experiment.")
else:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    first_frame = animation_data[0]

    max_u = max(np.max(frame['u']) for frame in animation_data)
    max_v = max(np.max(frame['v']) for frame in animation_data)

    im1 = axes[0].imshow(first_frame['u'], extent=[-L, L, -L, L], cmap='summer_r', origin='lower', vmin=0, vmax=max_u)
    axes[0].set_title('Vegetation density (V)')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    cb1 = fig.colorbar(im1, ax=axes[0])

    im2 = axes[1].imshow(first_frame['v'], extent=[-L, L, -L, L], cmap='Blues', origin='lower', vmin=0, vmax=max_v)
    axes[1].set_title('Water density (W)')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    cb2 = fig.colorbar(im2, ax=axes[1])

    fig_title = fig.suptitle('')
    plt.tight_layout()

    def update(frame_index):
        data = animation_data[frame_index]
        im1.set_data(data['u'])
        im2.set_data(data['v'])
        title_text = f'Water supply: A = {data["A"]:.3f} | Average biomass: {data["biomass"]:.4f}'
        fig_title.set_text(title_text)
        return im1, im2, fig_title

    ani = animation.FuncAnimation(fig, update, frames=len(animation_data), interval=100, blit=True)
    plt.close(fig) 

    print(f"Saving animation to file: {output_gif_path} (this may take a while)...")
    ani.save(output_gif_path, writer='imagemagick', fps=10)
    print("Animation saved successfully.")
