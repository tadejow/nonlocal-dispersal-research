#%%
# ==============================================================================
# CELL 1: Import packages and define operators and the model
# ==============================================================================
using BifurcationKit, LinearAlgebra, Plots, SparseArrays, Setfield, Statistics
const BK = BifurcationKit

function Laplacian1D_Dirichlet(N, L)
    hx = 2 * L / (N - 1)
    D2 = spdiagm(0 => -2 * ones(N), 1 => ones(N - 1), -1 => ones(N - 1)) / hx^2
    return D2
end

function F_system!(f, x, p)
    N = p.N
    u, v = (@view x[1:N]), (@view x[N+1:2N])
    fu, fv = (@view f[1:N]), (@view f[N+1:2N])
    nl_term = u .* u .* v
    mul!(fu, p.Op_u, u); fu .= p.d_u .* fu .- p.B .* u .+ nl_term
    mul!(fv, p.D2, v);   fv .= p.d_v .* fv .- v .+ p.A .- nl_term
    fu[1] = fu[end] = 0.0
    fv[1] = fv[end] = 0.0
    return f
end

function J_system(x, p)
    N = p.N
    u, v = (@view x[1:N]), (@view x[N+1:2N])
    J_uu = p.d_u .* p.Op_u - p.B .* I + spdiagm(0 => 2 .* u .* v)
    J_uv = spdiagm(0 => u .* u)
    J_vu = spdiagm(0 => -2 .* u .* v)
    J_vv = p.d_v .* p.D2 - I - spdiagm(0 => u .* u)
    J = [J_uu J_uv; J_vu J_vv]
    J[1, :] .= 0; J[1, 1] = 1.0
    J[N, :] .= 0; J[N, N] = 1.0
    J[N+1, :] .= 0; J[N+1, N+1] = 1.0
    J[2*N, :] .= 0; J[2*N, 2*N] = 1.0
    return J
end

println("Cell 1 executed: Packages, operators, and model defined.")


#%%
# ==============================================================================
# CELL 2: Definition of calibrated dispersal kernels
# ==============================================================================

# Numerically calculated constants to ensure each kernel has a variance of 1.
const C_SG = 0.32088
const B_SG = 0.11424

"""
Kernel 1: Sub-Gaussian (Fat Tails, higher chance of long-range jumps)
Variance = 1
"""
function kernel_sub_gaussian(x)
    return (1 / sqrt(2)) * exp(-sqrt(2) * abs(x))
end

"""
Kernel 2: Super-Gaussian (Thin Tails, short-range dispersal)
Variance = 1
"""
function kernel_super_gaussian(x)
    return C_SG * exp(-B_SG * x^4)
end

# We rename the function to avoid a name collision with a struct, which caused the MethodError.
function create_integral_matrix(x, kernel_func::Function)
    N = length(x)
    hx = x[2] - x[1]
    weights = ones(N); weights[1] = 0.5; weights[end] = 0.5; weights .*= hx
    # We use the provided function to build the kernel matrix.
    K_matrix = [kernel_func(xi - xj) for xi in x, xj in x]
    return sparse(K_matrix * diagm(weights))
end

println("Cell 2 executed: New kernels defined and matrix constructor function renamed.")


#%%
# ==============================================================================
# CELL 3: Main computation loop for the 2 non-local kernels
# ==============================================================================

# Simulation parameters
B = 0.45; d_u, d_v = 2.0, 80.0
A_min, A_max = 0.05, 3.0
L = 25
N = Int(6 * L)
domain = LinRange(-L, L, N)

# Dictionary to store the kernels to be tested
kernels_to_test = Dict(
    "Sub-Gaussian (Fat Tails)" => kernel_sub_gaussian,
    "Super-Gaussian (Thin Tails)" => kernel_super_gaussian
)

# Container for all results
all_results = Dict()

# Main loop that iterates over each kernel
for (kernel_name, kernel_func) in kernels_to_test
    println("\n" * "="^60)
    println("--- Starting computations for kernel: $kernel_name ---")
    
    # We create the non-local operator using the renamed function to avoid the error
    K_integral = create_integral_matrix(domain, kernel_func)
    Op_nonlocal = K_integral - I
    D2_laplace = Laplacian1D_Dirichlet(N, L)
    
    lens = @optic _.A
    params_nonlocal = (N=N, L=L, B=B, d_u=d_u, d_v=d_v, A=A_max, Op_u=Op_nonlocal, D2=D2_laplace)
    # For the local model, d_u is halved to correspond to the kernel variance of 1
    params_local = (N=N, L=L, B=B, d_u=d_u / 2, d_v=d_v, A=A_max, Op_u=D2_laplace, D2=D2_laplace)
    
    record_sol(x, p; kwargs...) = (u_max=maximum(view(x, 1:N)), u_avg=mean(view(x, 1:N)))
    opt_newton = NewtonPar(tol = 1e-10, max_iterations = 2000, verbose = false)
    opts_br = ContinuationPar(p_min = A_min, p_max = A_max, ds = -0.01, dsmax = 0.05,
        nev = 10, detect_bifurcation = 3, n_inversion = 10, max_steps=2000, newton_options = opt_newton)
    
    A_start = A_max
    u_hom = (A_start + sqrt(A_start^2 - 4*B^2)) / (2*B)
    v_hom = B / u_hom
    cos_profile = cos.((π/2) .* domain ./ L)
    u0 = u_hom .* cos_profile .+ 0.01 * (rand(N) .- 0.5) .* cos_profile
    v0 = v_hom .* cos_profile
    x0_guess = vcat(u0, v0)
    
    println("Computing bifurcation diagram for the non-local model...")
    prob_nl = BifurcationProblem(F_system!, x0_guess, params_nonlocal, lens; J=J_system, record_from_solution=record_sol)
    br_nl = @time continuation(prob_nl, PALC(), opts_br; verbosity = 0)
    
    println("Computing bifurcation diagram for the local model...")
    prob_l = BifurcationProblem(F_system!, x0_guess, params_local, lens; J=J_system, record_from_solution=record_sol)
    br_l = @time continuation(prob_l, PALC(), opts_br; verbosity = 0)
    
    # Save the results to the dictionary
    all_results[kernel_name] = Dict("nonlocal" => br_nl, "local" => br_l)
end
println("\n" * "="^60)
println("--- All computations finished. Ready for plotting. ---")


#%%
# ==============================================================================
# CELL 4: Plotting BIFURCATION DIAGRAMS (1x2 layout)
# ==============================================================================

# Plotting parameters
plot_size_diagrams = (1500, 800)
f_title = 15
f_guide = 15
f_tick = 15
f_legend = 15
line_width = 2.0
main_title_size = 18

# Array to store the individual plots
plot_array_diagrams = []

# We define the order in which to display the plots for consistency
kernel_order = ["Super-Gaussian (Thin Tails)", "Sub-Gaussian (Fat Tails)"]

# Loop for creating and saving the plots
for (idx, kernel_name) in enumerate(kernel_order)
    println("\n--- Preparing bifurcation diagram for kernel: $kernel_name ---")
    
    results = all_results[kernel_name]
    br_nl = results["nonlocal"]
    br_l = results["local"]

    p = plot(
        title = kernel_name,
        titlefontsize = f_title,
        tickfontsize = f_tick,
        guidefontsize = f_guide,
        legendfontsize = f_legend,
        legend = (idx == 1) ? :top : false,
        gridalpha = 0.3, 
        gridstyle = :dash
    )

    # Data from the non-local model
    plot!(p, br_nl.param, br_nl.u_max, label="Max (Non-local)", color=:orange, lw=line_width)
    plot!(p, br_nl.param, br_nl.u_avg, label="Avg (Non-local)", color=:green, lw=line_width)
    
    # Data from the local model
    plot!(p, br_l.param, br_l.u_max, label="Max (Local)", color=:cyan, ls=:dash, lw=line_width)
    plot!(p, br_l.param, br_l.u_avg, label="Avg (Local)", color=:blue, ls=:dash, lw=line_width)

    # Reference lines
    A_curve = LinRange(A_min, A_max, 200)
    plot!(p, A_curve, B ./ A_curve, color=:red, lw=line_width, label="B/A")
    vline!(p, [2 * B], label="A = 2B", color=:purple, ls=:dash, lw=2)

    # Axis labels
    xlabel!(p, "Rainfall A")
    if idx == 1; ylabel!(p, "(Avg / Max) biomass density"); else; plot!(p, yformatter=_->""); end

    push!(plot_array_diagrams, p)
end

# Assemble all plots into a single figure
final_diagrams_plot = plot(plot_array_diagrams..., 
    layout = (1, 2),
    size = plot_size_diagrams,
    link = :all,
    plot_titlefontsize = main_title_size,
    left_margin = 12Plots.mm, 
    bottom_margin = 10Plots.mm
)
    
display(final_diagrams_plot)


#%%
# ==============================================================================
# CELL 5: Plotting a 2x3 Profile Gallery (Always Upper Branch)
# ==============================================================================

println("\n--- Creating a unified profile gallery (always selecting the upper branch)... ---")

# --- 1. Helper Function for Intelligent Solution Finding ---

"""
Finds the solution profile in a branch `br` that is closest to a `target_A`.
Optionally, it can be guided by a `biomass_target` (e.g., :high or :low)
to resolve ambiguities on multi-valued branches.
"""
function find_closest_solution(br, target_A, N; biomass_target=:high)
    # Find all points on the branch close to the target A
    indices = findall(p -> abs(p - target_A) < 0.05, br.param)
    if isempty(indices)
        # Fallback if no points are very close: find the single closest point
        _, idx = findmin(abs.(br.param .- target_A))
        return br.sol[idx].x[1:N]
    end

    # If we have multiple candidates, use the recorded average biomass to choose.
    candidate_avg_biomass = br.u_avg[indices]
    
    if biomass_target == :high
        # Find the candidate with the highest average biomass
        _, max_idx_in_candidates = findmax(candidate_avg_biomass)
        correct_index = indices[max_idx_in_candidates]
    elseif biomass_target == :low
        # Find the candidate with the lowest average biomass
        _, min_idx_in_candidates = findmin(candidate_avg_biomass)
        correct_index = indices[min_idx_in_candidates]
    else # Default to just the closest A if target is not specified
        _, closest_idx_in_candidates = findmin(abs.(br.param[indices] .- target_A))
        correct_index = indices[closest_idx_in_candidates]
    end
    
    return br.sol[correct_index].x[1:N]
end


# --- 2. Define Plot Styles and Parameters ---
plot_size = (1500, 800)
num_cols = 3
f_title = 15
f_guide = 15
f_tick = 15
f_legend = 15
main_title_size = 18
line_width = 2.5

# Define the specific rainfall levels (A).
# MODIFIED: The biomass target is now always :high.
A_targets = [
    (A=1.5,   biomass=:high),
    (A=1.2,   biomass=:high),
    (A=1.1,   biomass=:high),
    (A=1.025, biomass=:high),
    (A=0.774, biomass=:high),
    (A=0.651, biomass=:high),
]

# Define a consistent plotting style
model_styles = Dict(
    "Local" => (label="Local", color=:blue, ls=:dash),
    "Super-Gaussian (Thin Tails)" => (label="Non-local (Thin Tails)", color=:cyan, ls=:solid),
    "Sub-Gaussian (Fat Tails)" => (label="Non-local (Fat Tails)", color=:orange, ls=:solid)
)

gallery_plots = []

# --- 3. Main Loop: Iterate over each target configuration ---
for (idx, target) in enumerate(A_targets)
    target_A = round(target.A, digits=2)
    biomass_regime = target.biomass
    
    println("--- Generating subplot for A ≈ $target_A (biomass: $biomass_regime) ---")
    
    p = plot(
        title = "Rainfall A ≈ $target_A",
        titlefontsize = f_title,
        tickfontsize = f_tick,
        guidefontsize = f_guide,
        legendfontsize = f_legend,
        legend = (idx == 1) ? :bottom : false
    )
    
    # --- 4. Inner Part: Plot each model profile using the helper function ---
    
    # First, the LOCAL model
    br_l = all_results["Super-Gaussian (Thin Tails)"]["local"]
    u_l_profile = find_closest_solution(br_l, target_A, N; biomass_target=biomass_regime)
    style = model_styles["Local"]
    plot!(p, domain, u_l_profile, 
        label=style.label, color=style.color, ls=style.ls, lw=line_width)

    # Next, the NON-LOCAL models
    kernel_order = ["Super-Gaussian (Thin Tails)", "Sub-Gaussian (Fat Tails)"]
    for kernel_name in kernel_order
        br_nl = all_results[kernel_name]["nonlocal"]
        
        u_nl_profile = find_closest_solution(br_nl, target_A, N; biomass_target=biomass_regime)
        style = model_styles[kernel_name]
        
        plot!(p, domain, u_nl_profile, 
            label=style.label, color=style.color, ls=style.ls, lw=line_width)
    end
    
    if idx % num_cols == 1; ylabel!(p, "Biomass density"); end
    if idx > num_cols; xlabel!(p, "x"); end
    
    push!(gallery_plots, p)
end

# --- 5. Assemble all subplots into a single final figure ---
final_gallery = plot(gallery_plots...,
    layout = (2, num_cols),
    size = plot_size,
    plot_titlefontsize = main_title_size,
    left_margin = 12Plots.mm,
    bottom_margin = 10Plots.mm,
    link = :all
)

display(final_gallery)

println("\n--- End of script. ---")