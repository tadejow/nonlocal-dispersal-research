using LinearAlgebra
using Plots
using ProgressMeter
using LaTeXStrings
using SpecialFunctions

# =========================================================================
# 1. DISCRETIZATION STRUCTS & OPERATORS
# =========================================================================

struct IntegralOperator
    kernel::Matrix{Float64}
    weights::Vector{Float64}
end

function IntegralOperator(x::AbstractVector, kernel_func::Function)
    dx = x[2] - x[1]
    grid_x = [i for i in x, j in x]
    grid_y = [j for i in x, j in x]
    kernel = kernel_func.(grid_x, grid_y)
    weights = ones(length(x))
    weights[begin] *= 0.5
    weights[end] *= 0.5
    weights .*= dx
    return IntegralOperator(kernel, weights)
end

function compute(op::IntegralOperator, u::AbstractVector)
    return (op.kernel * (u .* op.weights))
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

# =========================================================================
# 2. KERNEL DEFINITIONS
# =========================================================================

function kernel_fat(x, y)
    dist = abs(x - y)
    return (1.0 / sqrt(2.0)) * exp(-sqrt(2.0) * dist)
end

const g34 = gamma(0.75)
const g14 = gamma(0.25)
const J2_C1 = 2.0 * (sqrt(g34) / (g14^1.5))
const J2_C2 = (g34 / g14)^2

function kernel_thin(x, y)
    dist = abs(x - y)
    return J2_C1 * exp(-J2_C2 * dist^4)
end

# =========================================================================
# 3. SIMULATION LOGIC (Updated for dw sweep)
# =========================================================================

function run_simulation(L, N, u_init, v_init, op_type, operator, d_w_val)
    # Fixed parameters
    A, B, d_u = 1.8, 0.45, 2.0
    # d_v (water diffusion) is passed as d_w_val
    
    ht = 0.0001
    tol = 1e-6 
    max_iter = 500000

    u_old = copy(u_init)
    v_old = copy(v_init)
    u_old[begin] = u_old[end] = 0
    v_old[begin] = v_old[end] = 0
    
    dx = 2L / (N-1)

    for j in 1:max_iter
        if op_type == :integral
            u_spatial = d_u * compute(operator, u_old) - d_u * u_old
        else
            u_spatial = (d_u / 2) * compute(operator, u_old)
        end

        # Water diffusion uses the passed d_w_val
        w_spatial = zeros(N)
        w_spatial[2:end-1] = (v_old[3:end] - 2*v_old[2:end-1] + v_old[1:end-2]) / dx^2
        w_spatial .*= d_w_val

        u_reac = u_old.^2 .* v_old .- B .* u_old
        v_reac = .- u_old.^2 .* v_old .- v_old .+ A

        u_new = u_old .+ ht .* (u_spatial .+ u_reac)
        v_new = v_old .+ ht .* (w_spatial .+ v_reac)

        u_new[begin] = u_new[end] = 0
        v_new[begin] = v_new[end] = 0

        if norm(u_new - u_old) < tol
            # RETURN VALUE: The value at the node just inside the boundary
            # index `end-1` corresponds to x ≈ L - dx
            val_at_boundary = u_new[end-1]
            return val_at_boundary
        end

        if any(isnan, u_new) || any(isnan, v_new)
            return NaN
        end

        u_old, v_old = u_new, v_new
    end
    return NaN
end

# =========================================================================
# 4. EXECUTION
# =========================================================================

println("Starting Experiment 5.3 (Effect of Water Diffusion)...")

const K = 30 
const L_fixed = 25.0
const dw_min, dw_max = 0.01, 100.0

const A_p, B_p = 1.8, 0.45
const v_star = (A_p + sqrt(A_p^2 - 4*B_p^2)) / (2 * B_p)
const w_star = (2 * B_p^2) / (A_p + sqrt(A_p^2 - 4*B_p^2))

dw_vals = exp.(range(log(dw_min), log(dw_max), length=K))

# Store only the boundary value results
boundary_vals = Dict(
    :fat => zeros(K),
    :thin => zeros(K),
    :local => zeros(K)
)

@showprogress "Computing..." for (i, dw) in enumerate(dw_vals)
    # Use higher N to capture the boundary layer accurately
    N = 100 
    x_domain = range(-L_fixed, L_fixed, length=N)
    u_init = ones(N) .* v_star
    v_init = ones(N) .* w_star
    
    # 1. Fat
    op_fat = IntegralOperator(x_domain, kernel_fat)
    boundary_vals[:fat][i] = run_simulation(L_fixed, N, u_init, v_init, :integral, op_fat, dw)

    # 2. Thin
    op_thin = IntegralOperator(x_domain, kernel_thin)
    boundary_vals[:thin][i] = run_simulation(L_fixed, N, u_init, v_init, :integral, op_thin, dw)

    # 3. Local
    op_lap = LaplacianFD(x_domain)
    boundary_vals[:local][i] = run_simulation(L_fixed, N, u_init, v_init, :local, op_lap, dw)
end

# =========================================================================
# 5. PLOTTING
# =========================================================================

f_size = 12

# Create Single Plot
p = plot(
    title="Effect of Water Diffusion on Boundary Biomass",
    xaxis=("Water diffusion rate \$d_w\$", :log),
    yaxis="Biomass at boundary node (\$x \\approx L\$)",
    legend=:topright, 
    gridalpha=0.4,
    titlefontsize=f_size+1, 
    tickfontsize=f_size, 
    guidefontsize=f_size, 
    legendfontsize=10,
    size=(800, 600),
    margin=5Plots.mm
)

# Plot Lines with Scatters (and black outlines)
# 1. Local (Blue)
plot!(p, dw_vals, boundary_vals[:local], label="Local", 
      color=:navy, lw=2,
      marker=:circle, markersize=4, 
      markerstrokecolor=:black, markerstrokewidth=0.5)

# 2. Thin (Cyan)
plot!(p, dw_vals, boundary_vals[:thin], label="Non-local (Thin Tails)", 
      color=:cyan, lw=2,
      marker=:circle, markersize=4, 
      markerstrokecolor=:black, markerstrokewidth=0.5)

# 3. Fat (Orange)
plot!(p, dw_vals, boundary_vals[:fat], label="Non-local (Fat Tails)", 
      color=:orange, lw=2,
      marker=:circle, markersize=4, 
      markerstrokecolor=:black, markerstrokewidth=0.5)

display(p)