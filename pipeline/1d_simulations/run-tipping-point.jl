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
# 0. FILE OVERWRITE PROMPT FUNCTION
# ==============================================================================

function prompt_overwrite()
    println("\n[WARNING]: Data files for these parameters already exist in the destination directory.")
    print("Do you want to terminate the program? [y/n]: ")
    
    # Safe, blocking read
    while true
        res = readline()
        res_clean = lowercase(strip(res))
        
        if res_clean == "y" || res_clean == "yes" || res_clean == "t" || res_clean == "tak"
            println("\nTerminated by user.")
            return true
        elseif res_clean == "n" || res_clean == "no" || res_clean == "nie"
            println("\nContinuing and overwriting files...")
            return false
        else
            print("Invalid input. Please enter 'y' or 'n': ")
        end
    end
end

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
# 3. BIFURCATION LOGIC WRAPPER
# ==============================================================================

"""
Runs the PALC continuation for a specific set of parameters and returns the Tipping Point (A_crit).
Returns NaN if the continuation fails or no valid point is found.
"""
function get_tipping_point(p_val, du_val, dv_val, N, L, A_max, A_min, B, opts_br, zero_threshold)
    domain = LinRange(-L, L, N)
    D2_laplace = Laplacian1D_Dirichlet(N, L)
    
    K_integral = create_integral_matrix(domain, p_val)
    Op_nonlocal = K_integral - I
    
    params = (N=N, L=L, B=B, d_u=du_val, d_v=dv_val, A=A_max, Op_u=Op_nonlocal, D2=D2_laplace)
    
    # Initial guess (Analytical uniform state with spatial cosine perturbation)
    u_hom = (A_max + sqrt(A_max^2 - 4*B^2)) / (2*B)
    v_hom = B / u_hom
    cos_profile = cos.((pi/2) .* domain ./ L)
    u0 = u_hom .* cos_profile .+ 0.01 .* (rand(N) .- 0.5) .* cos_profile
    v0 = v_hom .* cos_profile
    x0_guess = vcat(u0, v0)
    
    lens = @optic _.A
    record_sol(x, p; kwargs...) = (u_avg=mean(view(x, 1:N)),)
    
    prob = BifurcationProblem(F_system!, x0_guess, params, lens; J=J_system, record_from_solution=record_sol)
    
    try
        br = continuation(prob, PALC(), opts_br; verbosity = 0)
        valid_A_indices = findall(u -> u > zero_threshold, br.u_avg)
        
        if isempty(valid_A_indices)
            return NaN
        else
            return minimum(br.param[valid_A_indices])
        end
    catch
        return NaN # Fallback in case Newton solver fails completely
    end
end

# ==============================================================================
# 4. MAIN EXPERIMENTS SETUP
# ==============================================================================

const OVERWRITE_DATA = true 

# Global parameters
const B = 0.45
const A_min, A_max = 0.05, 1.8
const L = 25.0
const N = Int(round(5 * L)) 
const zero_threshold = 0.1

# BifurcationKit options
opt_newton = NewtonPar(tol = 1e-10, max_iterations = 1000, verbose = false)
opts_br = ContinuationPar(p_min = A_min, p_max = A_max, ds = -0.01, dsmax = 0.05,
    nev = 10, detect_bifurcation = 3, n_inversion = 10, max_steps=2000, newton_options = opt_newton)

# Output paths
current_dir = @__DIR__
project_root = normpath(joinpath(current_dir, "..", ".."))
output_dir = joinpath(project_root, "data", "1d_simulations", "tipping_points")
mkpath(output_dir) 

csv_line = joinpath(output_dir, "tipping_line.csv")
csv_hm1  = joinpath(output_dir, "tipping_heatmap_water_diff.csv")
csv_hm2  = joinpath(output_dir, "tipping_heatmap_plant_disp.csv")

if isfile(csv_line) || isfile(csv_hm1) || isfile(csv_hm2)
    if !OVERWRITE_DATA
        error("\n[INFO]: Files already exist and OVERWRITE_DATA is set to false. Stopping.")
    else
        should_exit = prompt_overwrite()
        if should_exit
            exit(0)
        end
    end
end

# ==============================================================================
# 5. RUNNING THE EXPERIMENTS
# ==============================================================================

# --- EXP 1: Line plot (p vs Tipping Point) ---
# Fixed parameters: Plant dispersal (d_u) = 2.0, Water diffusion (d_v) = 0.1
println("\n[Exp 1/3] Computing Tipping Point vs p (Line Plot)...")
p_vals_line = collect(range(0.5, 4.0, length=25))
tipping_line = zeros(length(p_vals_line))

@showprogress for (i, p) in enumerate(p_vals_line)
    tipping_line[i] = get_tipping_point(p, 2.0, 0.1, N, L, A_max, A_min, B, opts_br, zero_threshold)
end

# --- EXP 2: Heatmap 1 (p vs Water Diffusion d_w) ---
# Fixed parameter: Plant dispersal (d_u) = 2.0
println("\n[Exp 2/3] Computing Heatmap: p vs Water Diffusion d_w (code: d_v)...")
p_vals_hm = collect(range(0.5, 4.0, length=15))
# Using a logarithmic-like spacing for water diffusion since it spans 0.1 to 80.0
dv_vals_hm = 10 .^ range(log10(0.1), log10(80.0), length=15)
tipping_hm1 = zeros(length(p_vals_hm), length(dv_vals_hm))

@showprogress for (j, dv) in enumerate(dv_vals_hm)
    for (i, p) in enumerate(p_vals_hm)
        tipping_hm1[i, j] = get_tipping_point(p, 2.0, dv, N, L, A_max, A_min, B, opts_br, zero_threshold)
    end
end

# --- EXP 3: Heatmap 2 (p vs Plant Dispersal d_v) ---
# Fixed parameter: Water diffusion (d_v in code, d_w in math) = 80.0 (Fast diffusion regime)
# Note: You can change fixed_water_diff to 0.1 if you prefer to observe the slow regime.
const fixed_water_diff = 80.0 
println("\n[Exp 3/3] Computing Heatmap: p vs Plant Dispersal d_v (code: d_u)...")
du_vals_hm = collect(range(0.1, 2.0, length=15))
tipping_hm2 = zeros(length(p_vals_hm), length(du_vals_hm))

@showprogress for (j, du) in enumerate(du_vals_hm)
    for (i, p) in enumerate(p_vals_hm)
        tipping_hm2[i, j] = get_tipping_point(p, du, fixed_water_diff, N, L, A_max, A_min, B, opts_br, zero_threshold)
    end
end

# ==============================================================================
# 6. DATA EXPORT
# ==============================================================================

open(csv_line, "w") do io
    write(io, "p,Tipping_A\n")
    writedlm(io, hcat(p_vals_line, tipping_line), ',')
end

open(csv_hm1, "w") do io
    write(io, "p,d_w_water_diff,Tipping_A\n")
    for j in 1:length(dv_vals_hm), i in 1:length(p_vals_hm)
        write(io, "$(p_vals_hm[i]),$(dv_vals_hm[j]),$(tipping_hm1[i,j])\n")
    end
end

open(csv_hm2, "w") do io
    write(io, "p,d_v_plant_disp,Tipping_A\n")
    for j in 1:length(du_vals_hm), i in 1:length(p_vals_hm)
        write(io, "$(p_vals_hm[i]),$(du_vals_hm[j]),$(tipping_hm2[i,j])\n")
    end
end
println("\nAll data successfully saved to: $output_dir")

# ==============================================================================
# 7. TUFTE-STYLE PLOTTING
# ==============================================================================
println("Generating combined Tufte-style layout plot...")

global_font_size = 12

default(
    fontfamily="sans-serif",
    grid=false,                 
    framestyle=:axes,           
    tickdirection=:out,         
    linewidth=2.5,
    markerstrokewidth=0,
    guidefontsize=global_font_size,
    tickfontsize=global_font_size,
    legendfontsize=global_font_size
)

# Custom colormap for heatmaps (viridis or turbo look good for tipping points)
cmap = cgrad(:turbo)

# --- Subplot A: Line Plot ---
# Note: Using math notation d_v (plants) and d_w (water) in titles/labels
p_line = plot(
    p_vals_line, tipping_line,
    color=:black,
    xlabel=L"Kernel Shape Exponent $p$",
    ylabel=L"Tipping Point $A_{crit}$",
    title="Tipping Point vs Tail Heaviness ($d_v=2.0$, $d_w=0.1$)",
    titlefontsize=13,
    legend=false,
    margin=5Plots.mm
)
scatter!(p_line, p_vals_line, tipping_line, color=:black, markersize=4)
hline!(p_line, [2*B], color=:red, linestyle=:dash, lw=2)
annotate!(p_line, 3.5, 2*B + 0.03, text(L"Kinetic threshold $A=2B$", :red, :center, 10))

# --- Subplot B: Heatmap 1 (p vs Water Diffusion d_w) ---
p_hm1 = heatmap(
    p_vals_hm, dv_vals_hm, tipping_hm1',
    yscale=:log10,
    xlabel=L"Kernel Shape Exponent $p$",
    ylabel=L"Water Diffusion $d_w$",
    title="Effect of Water Diffusion ($d_v=2.0$)",
    color=cmap,
    colorbar_title=L"$A_{crit}$",
    titlefontsize=13,
    margin=5Plots.mm
)

# --- Subplot C: Heatmap 2 (p vs Plant Dispersal d_v) ---
p_hm2 = heatmap(
    p_vals_hm, du_vals_hm, tipping_hm2',
    xlabel=L"Kernel Shape Exponent $p$",
    ylabel=L"Plant Dispersal $d_v$",
    title="Effect of Plant Dispersal ($d_w=80.0$)",
    color=cmap,
    colorbar_title=L"$A_{crit}$",
    titlefontsize=13,
    margin=5Plots.mm
)

# --- Combine Layout ---
# layout: 1 row with 1 plot spanning the top, 1 row with 2 plots spanning the bottom
l = @layout [a; b c]

final_plot = plot(
    p_line, p_hm1, p_hm2,
    layout=l,
    size=(1000, 850),
    left_margin=8Plots.mm,
    bottom_margin=8Plots.mm
)

plot_file = joinpath(output_dir, "Tipping_Points_Combined_Analysis.png")
savefig(final_plot, plot_file)
println("Plot successfully saved to: $plot_file")

display(final_plot)