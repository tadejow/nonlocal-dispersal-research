import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm
from utils import gauss_seidel, get_boundary_indices_2d
from operators import IntegralOperator2D, LaplacianOperator2D


# Parameters
B = float(input("Provide mortality rate B > 0: "))
d_u, d_v = 1.0, 10.0              # diffusion rates
ht = 0.01                         # time step
tol = 1e-2                        # convergence tolerance
max_iter_steady_state = 100000    # convergence max iter

A_min = float(input("Provide minimal rainfall rate A < 2B: "))
A_max = float(input("Provide maximal rainfall rate A > 2B: "))

L = 20.                           # size of the investigated spatial domain
K = 100                          # grid size for rainfall bifurcation
N = 50                            # grid size for the spatial domain (N x N)

# 2D domain
x_domain = np.linspace(-L, L, N)
y_domain = np.linspace(-L, L, N)
total_size = N * N

# Differential / Integral operators
print("Initialization of the operators 2D...")
integral_operator = IntegralOperator2D(x_domain, y_domain)
laplacian_operator = LaplacianOperator2D(x_domain, y_domain)

print("Construction of the discretized matrices...")
# Matrix for 'u'
integral_matrix = np.eye(total_size) - ht * (d_u * integral_operator.matrix - d_u * np.eye(total_size) - B * np.eye(total_size))
# Matrix for 'v'
diff_matrix = np.eye(total_size) - ht * (d_v * laplacian_operator.D2 - np.eye(total_size))

boundary_indices = get_boundary_indices_2d(N, N)

for i in boundary_indices:
    integral_matrix[i, :] = 0
    integral_matrix[i, i] = 1
    diff_matrix[i, :] = 0
    diff_matrix[i, i] = 1

# Bifurcation analysis
A_values = np.linspace(A_min, A_max, K)[::-1]
branch_results = {A: [] for A in A_values}

u_init, v_init = None, None

for idx, A in enumerate(tqdm(A_values, desc="Analiza bifurkacji 2D")):
    if idx == 0:
        # Begin analysis with constant steady states (of kinetic system)
        v_init_val = (A - np.sqrt(abs(A**2 - 4*B**2))) / 2
        u_init_val = (2 * B) / v_init_val
        v_init = np.full(N**2, v_init_val)
        u_init = np.full(N**2, u_init_val)
    else:
        # Otherwise use previous iteration as initial data
        prev_A = A_values[idx - 1]
        if branch_results[prev_A]:
            u_init = branch_results[prev_A][0][1].copy()
            v_init = branch_results[prev_A][0][2].copy()
        else:
            continue

    u_old = u_init.copy()
    v_old = v_init.copy()

    # Solution of the discrete system with implicit Euler
    for it in range(max_iter_steady_state):
        non_linear_term = (u_old**2) * v_old

        rhs_u = u_old + ht * non_linear_term
        rhs_u[boundary_indices] = 0    # Dirichlet boundary condition

        rhs_v = v_old + ht * (A - non_linear_term)
        rhs_v[boundary_indices] = 0    # Dirichlet boundary condition

        u_new = gauss_seidel(integral_matrix, rhs_u, 1e-5, 10, u_old)
        v_new = gauss_seidel(diff_matrix, rhs_v, 1e-5, 10, v_old)

        # Stopping criteria
        if np.isnan(u_new).any() or np.isnan(v_new).any():
            print(f"A = {A:.3f} | iter = {it} - BLOWUP")
            break
        total_error = np.linalg.norm(u_new - u_old) + np.linalg.norm(v_new - v_old)
        if total_error < tol:
            biomass = u_new.mean()
            branch_results[A].append((biomass, u_new.copy(), v_new.copy()))
            print(f"A = {A:.3f} | biomasa = {biomass:.4f} |"
                  f" iter = {it} |"
                  f" error = {total_error} |"
                  f" Converged!")
            break

        u_old, v_old = u_new.copy(), v_new.copy()

        if A == A_values[0]:
            print("Iteration: ", it, "Error: ", total_error)

        if it == max_iter_steady_state - 1:
            print(f"A = {A:.3f} | iter = {it} - MAX ITERATIONS REACHED")
            biomass = u_new.mean()
            branch_results[A].append((biomass, u_new.copy(), v_new.copy()))

animation_data = []
for A in A_values:
    if branch_results[A]:
        biomass, u, v = branch_results[A][0]
        if not np.isnan(biomass):
            animation_data.append({'A': A,
                                   'biomass': biomass,
                                   'u': u.reshape((N, N)),
                                   'v': v.reshape((N, N))})

if not animation_data:
    print("No numerical data was generated in this experiment")
else:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    first_frame = animation_data[0]

    max_u = max(np.max(frame['u']) for frame in animation_data)
    max_v = max(np.max(frame['v']) for frame in animation_data)

    im1 = axes[0].imshow(first_frame['u'], extent=[-L, L, -L, L],
                        cmap='summer_r', origin='lower', vmin=0, vmax=max_u)
    axes[0].set_title('Biomass density')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    cb1 = fig.colorbar(im1, ax=axes[0])

    im2 = axes[1].imshow(first_frame['v'], extent=[-L, L, -L, L],
                        cmap='Blues', origin='lower', vmin=0, vmax=max_v)
    axes[1].set_title('Water density')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    cb2 = fig.colorbar(im2, ax=axes[1])

    # Tytuł całej figury, który też będzie aktualizowany
    fig_title = fig.suptitle('')

    plt.tight_layout()

    # 3. Definicja funkcji aktualizującej (co ma się dziać w każdej klatce)
    def update(frame_index):
        # Pobieramy dane dla bieżącej klatki
        data = animation_data[frame_index]
        u, v = data['u'], data['v']

        im1.set_data(u)
        im2.set_data(v)

        title_text = (f'Water supply: A = {data["A"]:.3f} | '
                      f'Average biomass: {data["biomass"]:.4f}')
        fig_title.set_text(title_text)
        return im1, im2, fig_title

    ani = animation.FuncAnimation(fig, update, frames=len(animation_data),
                                  interval=100, blit=True)

    plt.close(fig) # Zamykamy statyczny wykres, żeby nie wyświetlił się podwójnie
    plt.show()

    print("Zapisywanie animacji do pliku... (może to potrwać)")
    ani.save("bifurcation_animation.gif", writer='imagemagick', fps=10)
    print("Animacja zapisana.")

