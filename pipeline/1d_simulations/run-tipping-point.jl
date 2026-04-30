# ==============================================================================
# IMPORTS & PACKAGES
# ==============================================================================
using BifurcationKit
using LinearAlgebra
using Plots
using SparseArrays
using Setfield
using Statistics
using ProgressMeter
using LaTeXStrings
using SpecialFunctions # Required for the Gamma function (gamma)
using DelimitedFiles   # Required for CSV export

const BK = BifurcationKit

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
# 2. OPERATOR AND SYSTEM DEFINITIONS
# ==============================================================================

function create_integral_matrix(x::AbstractVector, p::Float64)
    N = length(x)
    dx = x[2] - x[1]
    C_p, lambda_p = get_kernel_constants(p)
    
    grid_x = [i for i in x, j in x]
    grid_y = [j for i in x, j in x]
    
    kernel_mat = C_p .* exp.(-lambda_p .* abs.(grid_x .- grid_y).^p)
    
    # Trapezoidal rule weights
    weights = ones(N)
    weights[begin] *= 0.5
    weights[end] *= 0.5
    weights .*= dx
    
    return sparse(kernel_mat * diagm(weights))
end

function Laplacian1D_Dirichlet(N, L)
    hx = 2 * L / (N - 1)
    D2 = spdiagm(0 => -2 * ones(N), 1 => ones(N - 1), -1 => ones(N - 1)) / hx^2
    return D2
end

function F_system!(f, x, params)
    N = params.N
    u, v = (@view x[1:N]), (@view x[N+1:2N])
    fu, fv = (@view f[1:N]), (@view f[N+1:2N])
    
    nl_term = u .* u .* v
    mul!(fu, params.Op_u, u); fu .= params.d_u .* fu .- params.B .* u .+ nl_term
    mul!(fv, params.D2, v);   fv .= params.d_v .* fv .- v .+ params.A .- nl_term
    
    # Enforce Dirichlet BCs
    fu[1] = fu[end] = 0.0
    fv[1] = fv[end] = 0.0
    return f
end

function J_system(x, params)
    N = params.N
    u, v = (@view x[1:N]), (@view x[N+1:2N])
    
    J_uu = params.d_u .* params.Op_u - params.B .* I + spdiagm(0 => 2 .* u .* v)
    J_uv = spdiagm(0 => u .* u)
    J_vu = spdiagm(0 => -2 .* u .* v)
    J_vv = params.d_v .* params.D2 - I - spdiagm(0 => u .* u)
    
    J = [J_uu J_uv; J_vu J_vv]
    
    # Boundary conditions on Jacobian
    J[1, :] .= 0; J[1, 1] = 1.0
    J[N, :] .= 0; J[N, N] = 1.0
    J[N+1, :] .= 0; J[N+1, N+1] = 1.0
    J[2*N, :] .= 0; J[2*N, 2*N] = 1.0
    return J
end

# ==============================================================================
# 3. MAIN EXPERIMENT LOOP
# ==============================================================================

println("Starting Tipping Point vs Kernel Exponent (p) experiment using BifurcationKit...")

# --- Global Ecosystem Parameters ---
B = 0.45; d_u = 2.0; d_v = 0.1
A_min, A_max = 0.05, 1.8
L = 25.0
N = Int(round(6 * L)) # Good resolution for 1D
domain = LinRange(-L, L, N)
D2_laplace = Laplacian1D_Dirichlet(N, L)

zero_threshold = 0.1

# --- Scanning Ranges ---
p_vals = collect(range(0.5, 4.0, length=35))
tipping_A = zeros(length(p_vals))

opt_newton = NewtonPar(tol = 1e-10, max_iterations = 1000, verbose = false)
opts_br = ContinuationPar(p_min = A_min, p_max = A_max, ds = -0.01, dsmax = 0.05,
    nev = 10, detect_bifurcation = 3, n_inversion = 10, max_steps=2000, newton_options = opt_newton)

record_sol(x, p; kwargs...) = (u_avg=mean(view(x, 1:N)),)

lens = @optic _.A

# --- Execution ---
@showprogress "Tracking bifurcations for 'p': " for (i, p) in enumerate(p_vals)
    
    # 1. Build specific non-local operator for current 'p'
    K_integral = create_integral_matrix(domain, p)
    Op_nonlocal = K_integral - I
    
    params = (N=N, L=L, B=B, d_u=d_u, d_v=d_v, A=A_max, Op_u=Op_nonlocal, D2=D2_laplace)
    
    # 2. Initial guess (Analytical uniform state with spatial cosine perturbation)
    u_hom = (A_max + sqrt(A_max^2 - 4*B^2)) / (2*B)
    v_hom = B / u_hom
    cos_profile = cos.((pi/2) .* domain ./ L)
    u0 = u_hom .* cos_profile .+ 0.01 .* (rand(N) .- 0.5) .* cos_profile
    v0 = v_hom .* cos_profile
    x0_guess = vcat(u0, v0)
    
    # 3. Continuation tracking using PALC
    prob = BifurcationProblem(F_system!, x0_guess, params, lens; J=J_system, record_from_solution=record_sol)
    br = continuation(prob, PALC(), opts_br; verbosity = 0)
    
    # 4. Find the tipping point 
    # PALC correctly traces around the fold (turning point). 
    # We just need the minimum A reached while biomass is still above the desert threshold.
    valid_A_indices = findall(u -> u > zero_threshold, br.u_avg)
    
    if isempty(valid_A_indices)
        tipping_A[i] = NaN
    else
        valid_A_vals = br.param[valid_A_indices]
        tipping_A[i] = minimum(valid_A_vals)
    end
end

# ==============================================================================
# 4. DATA EXPORT
# ==============================================================================

current_dir = @__DIR__
project_root = normpath(joinpath(current_dir, "..", ".."))
output_dir = joinpath(project_root, "data", "1d_simulations", "tipping_points")

mkpath(output_dir) 

output_file = joinpath(output_dir, "tipping_A_vs_p_BifurcationKit.csv")
open(output_file, "w") do io
    write(io, "p,Tipping_A\n")
    writedlm(io, hcat(p_vals, tipping_A), ',')
end
println("\nRaw data successfully saved to: $output_file")

# ==============================================================================
# 5. TUFTE-STYLE PLOTTING
# ==============================================================================
println("Generating minimalist Tufte-style plot...")

default(
    fontfamily="sans-serif",
    grid=false,                 
    framestyle=:axes,           
    tickdirection=:out,         
    linewidth=2.5,
    markerstrokewidth=0,
    guidefontsize=12,
    tickfontsize=10,
    legend=false
)

p_plot = plot(
    p_vals, tipping_A,
    color=:black,
    xlabel=L"Kernel Shape Exponent $p$",
    ylabel=L"Tipping Point (Rainfall $A_{crit}$)",
    title="Vegetation Tipping Point vs Dispersal Tail Heaviness (BifurcationKit)",
    titlefontsize=14,
    size=(700, 500),
    margin=5Plots.mm
)

scatter!(p_plot, p_vals, tipping_A, color=:black, markersize=4)

# Reference line: The theoretical tipping point for the purely kinetic system (A = 2B)
hline!(p_plot, [2*B], color=:red, linestyle=:dash, lw=2)
annotate!(p_plot, 3.5, 2*B + 0.03, text(L"Kinetic threshold $A=2B$", :red, :center, 10))

# Highlight key ecological thresholds (Laplace vs Gaussian)
idx_laplace = argmin(abs.(p_vals .- 1.0))
idx_gauss   = argmin(abs.(p_vals .- 2.0))

scatter!(p_plot,[p_vals[idx_laplace]],[tipping_A[idx_laplace]], color=:orange, markersize=8)
annotate!(p_plot, p_vals[idx_laplace] + 0.1, tipping_A[idx_laplace] + 0.05, 
          text("Fat Tails\n(p=1)", :orange, :left, 10))

scatter!(p_plot, [p_vals[idx_gauss]], [tipping_A[idx_gauss]], color=:cyan, markersize=8)
annotate!(p_plot, p_vals[idx_gauss] + 0.1, tipping_A[idx_gauss] - 0.05, 
          text("Normal Tails\n(p=2)", :cyan, :left, 10))

plot_file = joinpath(output_dir, "Tipping_A_vs_p_BifurcationKit.png")
savefig(p_plot, plot_file)
println("Plot successfully saved to: $plot_file")

display(p_plot)