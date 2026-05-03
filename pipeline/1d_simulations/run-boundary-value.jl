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
    
    grid_x = [i for i in x, j in x]
    grid_y = [j for i in x, j in x]
    
    kernel_mat = C_p .* exp.(-lambda_p .* abs.(grid_x .- grid_y).^p)
    
    weights = ones(length(x))
    weights[begin] *= 0.5
    weights[end] *= 0.5
    weights .*= dx
    
    return IntegralOperator(kernel_mat, weights)
end

function compute(op::IntegralOperator, u::AbstractVector)
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

function compute(op::LaplacianFD, u::AbstractVector)
    return op.D2 * u
end

# ==============================================================================
# 3. SIMULATION LOGIC
# ==============================================================================

function run_simulation(L::Float64, N::Int, u_init::Vector{Float64}, v_init::Vector{Float64}, 
                        p::Float64, A::Float64, B::Float64, d_u::Float64, d_v::Float64, ht::Float64, tol::Float64, is_local::Bool=false)
    
    domain_omega = range(-L, L, length=N)
    laplacian_fd = LaplacianFD(domain_omega)
    
    local integral_operator
    if !is_local
        integral_operator = IntegralOperator(domain_omega, p)
    end

    u_old = copy(u_init)
    v_old = copy(v_init)

    # Strictly enforce Dirichlet boundary conditions
    u_old[begin] = u_old[end] = 0.0
    v_old[begin] = v_old[end] = 0.0
    
    for j in 1:300000 
        
        if is_local
            u_spatial_term = (d_u / 2.0) * compute(laplacian_fd, u_old)
        else
            u_spatial_term = d_u * compute(integral_operator, u_old) - d_u * u_old
        end
        
        v_spatial_term = d_v * compute(laplacian_fd, v_old)

        u_new = u_old .+ ht .* (u_spatial_term .+ u_old.^2 .* v_old .- B .* u_old)
        v_new = v_old .+ ht .* (v_spatial_term .- u_old.^2 .* v_old .- v_old .+ A)

        u_new[begin] = u_new[end] = 0.0
        v_new[begin] = v_new[end] = 0.0

        if norm(u_new - u_old) < tol
            return u_new # Return the full profile upon convergence
        end

        if any(isnan, u_new) || any(isnan, v_new)
            return nothing
        end

        u_old, v_old = copy(u_new), copy(v_new)
    end
    
    return nothing # Max iterations reached
end

# ==============================================================================
# 4. MAIN EXPERIMENTS & FILE CHECKING
# ==============================================================================

# --- Change to false if you're not sure ---
const OVERWRITE_DATA = true 

# --- Global Ecosystem Parameters ---
const A, B = 1.8, 0.45
const d_u, d_v = 0.1, 0.1
const ht = 0.0001
const tol = 1e-5
const zero_threshold = 0.1

uniform_u = (A + sqrt(A^2 - 4*B^2)) / (2 * B)
uniform_v = (2 * B^2) / (A + sqrt(A^2 - 4*B^2))

# Setup directories and filenames based on parameters
current_dir = @__DIR__
project_root = normpath(joinpath(current_dir, "..", ".."))
output_dir = joinpath(project_root, "data", "1d_simulations", "boundary_gap")
mkpath(output_dir)

file_prefix = "boundary_gap_A$(A)_B$(B)_du$(d_u)_dv$(d_v)"
csv_L       = joinpath(output_dir, "$(file_prefix)_vs_L.csv")
csv_heatmap = joinpath(output_dir, "$(file_prefix)_heatmap.csv")

# Check if data already exists
if (isfile(csv_L) || isfile(csv_heatmap))
    if !OVERWRITE_DATA
        error("\n[INFO]: Files already exist and OVERWRITE_DATA is set to false. Stopping execution to prevent overwrite.")
    else
        print("\n[INFO]: Files already exist and OVERWRITE_DATA is set to true. Continuing execution and overwriting.")
    end
end

# ------------------------------------------------------------------------------
# EXPERIMENT 1: Boundary Gap vs. Domain Size (L) for various p
# ------------------------------------------------------------------------------
println("\nStarting Experiment 1: Boundary Gap vs Domain Size (L)...")

# We test L on a logarithmic scale from 1 to 100
L_vals = 10 .^ range(0.0, 2.0, length=40)

# We want lines for p = 1, 2, 3, 4
p_tests = [1.0, 2.0, 3.0, 4.0]
gap_vs_L_all = zeros(length(L_vals), length(p_tests))

@showprogress "Scanning 'L' domains: " for (i, L) in enumerate(L_vals)
    N_exp1 = Int(round(30 + 3 * L))
    u_init = ones(N_exp1) .* uniform_u
    v_init = ones(N_exp1) .* uniform_v
    
    for (idx_p, p) in enumerate(p_tests)
        u_res = run_simulation(L, N_exp1, u_init, v_init, p, A, B, d_u, d_v, ht, tol, false)
        if u_res !== nothing && (sum(u_res) / N_exp1) > zero_threshold
            gap_vs_L_all[i, idx_p] = (u_res[2] + u_res[end-1]) / 2.0
        else
            gap_vs_L_all[i, idx_p] = NaN
        end
    end
end

# ------------------------------------------------------------------------------
# EXPERIMENT 2: Boundary Gap Heatmap (p vs L)
# ------------------------------------------------------------------------------
println("\nStarting Experiment 2: Boundary Gap Heatmap (p vs L)...")

p_vals_hm = collect(range(1.0, 4.0, length=40))
L_vals_hm = 10 .^ range(0.0, 2.0, length=40)
gap_heatmap = zeros(length(p_vals_hm), length(L_vals_hm))

total_iters = length(p_vals_hm) * length(L_vals_hm)
prog_hm = Progress(total_iters, desc="Computing Heatmap: ")

for (j, L) in enumerate(L_vals_hm)
    N_hm = Int(round(30 + 3 * L)) 
    u_init = ones(N_hm) .* uniform_u
    v_init = ones(N_hm) .* uniform_v
    
    for (i, p) in enumerate(p_vals_hm)
        u_hm = run_simulation(L, N_hm, u_init, v_init, p, A, B, d_u, d_v, ht, tol, false)
        
        if u_hm !== nothing
            gap_heatmap[i, j] = (u_hm[2] + u_hm[end-1]) / 2.0
        else
            gap_heatmap[i, j] = NaN
        end
        next!(prog_hm)
    end
end

# ==============================================================================
# 5. DATA EXPORT
# ==============================================================================

# Save Data for Exp 1
open(csv_L, "w") do io
    write(io, "L,p1,p2,p3,p4\n")
    writedlm(io, hcat(L_vals, gap_vs_L_all), ',')
end

# Save Data for Exp 2 (Heatmap format: p, L, gap)
open(csv_heatmap, "w") do io
    write(io, "p,L,boundary_gap\n")
    for j in 1:length(L_vals_hm)
        for i in 1:length(p_vals_hm)
            write(io, "$(p_vals_hm[i]),$(L_vals_hm[j]),$(gap_heatmap[i,j])\n")
        end
    end
end

println("\nNumerical data successfully saved to: $output_dir")

# ==============================================================================
# 6. TUFTE-STYLE PLOTTING
# ==============================================================================
println("Generating minimalist Tufte-style plots...")

# Define unified font sizes
global_font_size = 12

default(
    fontfamily="sans-serif",
    grid=true, gridalpha=0.3, gridstyle=:dash,
    framestyle=:axes,
    tickdirection=:out,
    linewidth=2.5,
    markerstrokewidth=0,
    guidefontsize=global_font_size,
    tickfontsize=global_font_size,
    legendfontsize=global_font_size,
    titlefontsize=global_font_size + 2
)

# --- Plot 1: Gap vs L ---
# Line styles for p=1, 2, 3, 4
styles = [:solid, :dash, :dot, :dashdot]

plot_L = plot(
    xscale=:log10,
    xticks=[1, 10, 100],
    xlabel=L"Patch Half-Width $L$",
    ylabel=L"Boundary Drop-off Value $V(-L^+)$",
    title="Boundary Gap Size vs Habitat Size",
    legend=:bottomright
)

for (idx_p, p) in enumerate(p_tests)
    plot!(plot_L, L_vals, gap_vs_L_all[:, idx_p], 
          color=:black, 
          ls=styles[idx_p], 
          label="p = $(Int(p))")
    scatter!(plot_L, L_vals, gap_vs_L_all[:, idx_p], color=:black, markersize=3, label=false)
end

# --- Plot 2: Heatmap ---
# Custom color gradient: white to red
custom_grad = cgrad([:white, :red])

plot_hm = heatmap(
    L_vals_hm, p_vals_hm, gap_heatmap,
    xscale=:log10,
    xticks=[1, 10, 100],
    xlabel=L"Patch Half-Width $L$",
    ylabel=L"Kernel Shape Exponent $p$",
    title="Boundary Gap Size Heatmap",
    color=custom_grad, 
    colorbar_title="Boundary Drop-off Value $V(-L^+)$",
    right_margin=5Plots.mm
)

# --- Combine into a 1x2 subplot ---
final_plot = plot(
    plot_L, plot_hm, 
    layout = (1, 2),
    size = (1200, 500),
    bottom_margin = 8Plots.mm,
    left_margin = 8Plots.mm
)

file_combined_png = joinpath(output_dir, "$(file_prefix)_plot_combined.png")
savefig(final_plot, file_combined_png)

println("Combined plot successfully saved to: $file_combined_png")
display(final_plot)