# ==============================================================================
# IMPORTS & PACKAGES
# ==============================================================================
using LinearAlgebra
using Plots
using ProgressMeter
using LaTeXStrings
using SpecialFunctions # Required for the Gamma function (gamma)
using DelimitedFiles   # Required for CSV export

# ==============================================================================
# 1. MATHEMATICAL KERNEL CALIBRATION
# ==============================================================================

"""
Calculates the constants C_p and lambda_p for the generalized Gaussian kernel:
    J(x) = C_p * exp(-lambda_p * |x|^p)
These constants guarantee that the kernel integrates to 1.0 over R 
and has a variance (second moment) of exactly 1.0.
"""
function get_kernel_constants(p::Float64)
    # Analytically derived constraints using properties of the Gamma function
    lambda_p = (gamma(3.0 / p) / gamma(1.0 / p))^(p / 2.0)
    C_p = (p * lambda_p^(1.0 / p)) / (2.0 * gamma(1.0 / p))
    return C_p, lambda_p
end

# ==============================================================================
# 2. OPERATOR DEFINITIONS
# ==============================================================================

struct IntegralOperator
    kernel::Matrix{Float64}
    weights::Vector{Float64}
end

function IntegralOperator(x::AbstractVector, p::Float64)
    dx = x[2] - x[1]
    C_p, lambda_p = get_kernel_constants(p)
    
    # Create pairwise distance matrices
    grid_x = [i for i in x, j in x]
    grid_y =[j for i in x, j in x]
    
    # Evaluate the generalized exponential kernel
    kernel_mat = C_p .* exp.(-lambda_p .* abs.(grid_x .- grid_y).^p)
    
    # Apply standard trapezoidal integration weights
    weights = ones(length(x))
    weights[begin] *= 0.5
    weights[end] *= 0.5
    weights .*= dx
    
    return IntegralOperator(kernel_mat, weights)
end

"Applies the non-local integral operator to a state vector u."
function compute(op::IntegralOperator, u::AbstractVector)
    # The normalization coefficient C_p is already baked into the kernel_mat
    return op.kernel * (u .* op.weights)
end

struct LaplacianFD
    D2::Matrix{Float64}
end

function LaplacianFD(x::AbstractVector)
    N = length(x)
    dx = x[2] - x[1]
    D2 = diagm(0 => -2 * ones(N), 1 => ones(N - 1), -1 => ones(N - 1)) / dx^2
    return LaplacianFD(D2)
end

"Applies the discrete Laplacian operator to a state vector u."
function compute(op::LaplacianFD, u::AbstractVector)
    return op.D2 * u
end

# ==============================================================================
# 3. SIMULATION LOGIC
# ==============================================================================

function run_simulation(L::Float64, N::Int, u_init::Vector{Float64}, v_init::Vector{Float64}, 
                        p::Float64, A::Float64, B::Float64, d_u::Float64, d_v::Float64, ht::Float64, tol::Float64)
    
    domain_omega = range(-L, L, length=N)
    integral_operator = IntegralOperator(domain_omega, p)
    laplacian_fd = LaplacianFD(domain_omega)

    u_old = copy(u_init)
    v_old = copy(v_init)

    # Strictly enforce Dirichlet boundary conditions (hostile surroundings)
    u_old[begin] = u_old[end] = 0.0
    v_old[begin] = v_old[end] = 0.0
    
    # Implicit/Explicit stepping
    for j in 1:200000 
        u_spatial_term = d_u * compute(integral_operator, u_old) - d_u * u_old
        v_spatial_term = d_v * compute(laplacian_fd, v_old)

        u_new = u_old .+ ht .* (u_spatial_term .+ u_old.^2 .* v_old .- B .* u_old)
        v_new = v_old .+ ht .* (v_spatial_term .- u_old.^2 .* v_old .- v_old .+ A)

        u_new[begin] = u_new[end] = 0.0
        v_new[begin] = v_new[end] = 0.0

        # Check for steady-state convergence
        if norm(u_new - u_old) < tol
            biomass = sum(u_new) / length(u_new)
            return biomass
        end

        # Check for numerical instability (blow-up)
        if any(isnan, u_new) || any(isnan, v_new)
            return NaN
        end

        u_old, v_old = copy(u_new), copy(v_new)
    end
    
    return NaN # Max iterations reached without satisfying tolerance
end

# ==============================================================================
# 4. MAIN EXPERIMENT LOOP
# ==============================================================================

println("Starting Critical Patch Size vs Kernel Exponent (p) experiment...")

# --- Global Ecosystem Parameters ---
A, B = 1.8, 0.45
d_u, d_v = 2.0, 0.1
ht = 0.0001
tol = 1e-10
zero_threshold = 0.1

# --- Domain Scanning Range ---
K_L = 80
L_min, L_max = 1.0, 80.0
L_vals = exp.(range(log(L_min), log(L_max), length=K_L))

# --- Exponent (p) Scanning Range ---
# Spanning from highly leptokurtic (fat tails) to highly platykurtic (thin tails)
p_vals = collect(range(0.5, 4.0, length=40))
L_crits = zeros(length(p_vals))

# Calculate the analytical positive uniform steady states
uniform_u = (A + sqrt(A^2 - 4*B^2)) / (2 * B)
uniform_v = (2 * B^2) / (A + sqrt(A^2 - 4*B^2))

# --- Execution ---
@showprogress "Scanning 'p' exponents: " for (i, p) in enumerate(p_vals)
    L_crit = NaN
    
    # Scan L from smallest to largest to find the precise emergence threshold
    for L in L_vals
        
        # DYNAMIC GRID SIZING:
        # 1. Base of 30 prevents dx from becoming dangerously small at L=1, avoiding CFL blowups.
        # 2. Linear growth (+ 3.0*L) keeps N moderate (~270) at large L, speeding up matrix ops.
        N = Int(round(20 + 2*L))
        
        u_init = ones(N) .* uniform_u
        v_init = ones(N) .* uniform_v
        
        biomass = run_simulation(L, N, u_init, v_init, p, A, B, d_u, d_v, ht, tol)
        
        # Check if vegetation survived the hostile boundaries
        if !isnan(biomass) && biomass > zero_threshold
            L_crit = L
            break # Optimization: Halt the L-scan upon finding the minimum viable patch
        end
    end
    
    L_crits[i] = L_crit
end

# ==============================================================================
# 5. DATA EXPORT (DYNAMIC PATH RESOLUTION)
# ==============================================================================

# DYNAMIC PATH: @__DIR__ points to the directory where this script is located.
# Assuming the script is in `pipeline/1d_simulations/`, we go up two levels (`..`, `..`)
# to reach the project root, then down into `data/1d_simulations/critical_patch_size/`.
current_dir = @__DIR__
project_root = normpath(joinpath(current_dir, "..", ".."))
output_dir = joinpath(project_root, "data", "1d_simulations", "critical_patch_size")

mkpath(output_dir) # Creates directory safely if it doesn't exist

output_file = joinpath(output_dir, "Lcrit_vs_p.csv")
open(output_file, "w") do io
    write(io, "p,L_crit\n")
    writedlm(io, hcat(p_vals, L_crits), ',')
end
println("\nRaw data successfully saved to: $output_file")

# ==============================================================================
# 6. VISUALIZATION (Edward Tufte's Minimalist Principles)
# ==============================================================================
println("Generating minimalist Tufte-style plot...")

# Configure Plots.jl to maximize the data-to-ink ratio
default(
    fontfamily="sans-serif",
    grid=false,                 # Remove distracting heavy grids
    framestyle=:axes,           # Eliminate top and right borders (Tufte's L-axes)
    tickdirection=:out,         # Ticks point outward, keeping data area clean
    linewidth=2.5,
    markerstrokewidth=0,
    guidefontsize=12,
    tickfontsize=10,
    legend=false
)

# Initialize the main figure
p_plot = plot(
    p_vals, L_crits,
    color=:black,
    xlabel=L"Kernel Shape Exponent $p$",
    ylabel=L"Critical Patch Size $L_{crit}$",
    title="Impact of Dispersal Tail Heaviness on Vegetation Persistence",
    titlefontsize=14,
    size=(700, 500),
    margin=5Plots.mm
)

# Add scattered data points to visualize the discrete simulation steps
scatter!(p_plot, p_vals, L_crits, color=:black, markersize=4)

# Locate specific canonical kernels for annotation
idx_laplace = argmin(abs.(p_vals .- 1.0))
idx_gauss   = argmin(abs.(p_vals .- 2.0))

# Highlight the Sub-Gaussian (Laplace) scenario
scatter!(p_plot, [p_vals[idx_laplace]], [L_crits[idx_laplace]], color=:orange, markersize=8)
annotate!(p_plot, p_vals[idx_laplace] + 0.1, L_crits[idx_laplace], 
          text("Fat Tails\n(Sub-Gaussian, p=1)", :orange, :left, 11))

# Highlight the Standard Gaussian scenario
scatter!(p_plot, [p_vals[idx_gauss]],[L_crits[idx_gauss]], color=:blue, markersize=8)
annotate!(p_plot, p_vals[idx_gauss] + 0.1, L_crits[idx_gauss] - 0.2, 
          text("Normal Tails\n(Gaussian, p=2)", :blue, :left, 11))

# Save the final high-resolution figure using the dynamic path
plot_file = joinpath(output_dir, "Lcrit_vs_p.png")
savefig(p_plot, plot_file)
println("Plot successfully saved to: $plot_file")

display(p_plot)
println("Done.")