using LinearAlgebra
using Plots
using ProgressMeter
using LaTeXStrings
using SpecialFunctions

# =========================================================================
# 1. STRUCTS & OPERATORS
# =========================================================================

struct IntegralOperator
    kernel::Matrix{Float64}
    weights::Vector{Float64}
end

function IntegralOperator(x::AbstractVector, kernel_matrix::Matrix{Float64})
    dx = x[2] - x[1]
    weights = ones(length(x))
    weights[begin] *= 0.5
    weights[end] *= 0.5
    weights .*= dx
    return IntegralOperator(kernel_matrix, weights)
end

function compute(op::IntegralOperator, u::AbstractVector)
    return (op.kernel * (u .* op.weights))
end

# =========================================================================
# 2. GENERALIZED KERNEL GENERATOR (Strict Normalization)
# =========================================================================

"""
Generates a Generalized Normal Kernel J(x) = C * exp(-|lambda * x|^beta).
Constraints applied:
1. Integral(J) = 1
2. Variance(J) = 1
Returns: (kernel_function, M4_value)
"""
function get_gnd_kernel_data(beta)
    # --- 1. Fix Variance to 1 ---
    # The variance of GND is: sigma^2 = Gamma(3/b) / (lambda^2 * Gamma(1/b))
    # We set sigma^2 = 1 and solve for lambda:
    lam = sqrt(gamma(3/beta) / gamma(1/beta))
    
    # --- 2. Fix Integral to 1 (Normalization) ---
    # The raw integral of exp(-|lam*x|^beta) is: 2 * Gamma(1/b) / (beta * lambda)
    # The Normalization Constant C is the inverse of this:
    norm_C = (beta * lam) / (2 * gamma(1/beta))
    
    # --- 3. Calculate Resulting 4th Moment (Kurtosis) ---
    # M4 = Gamma(5/b) / (lambda^4 * Gamma(1/b))
    m4_val = gamma(5/beta) / (lam^4 * gamma(1/beta))
    
    # Define the function closure
    k_func = (x, y) -> begin
        dist = abs(x - y)
        return norm_C * exp(-abs(lam * dist)^beta)
    end
    
    return k_func, m4_val
end

# =========================================================================
# 3. SIMULATION LOGIC (Bisection Search Support)
# =========================================================================

function check_survival(L, kernel_func)
    # Parameters
    A, B, d_u, d_v = 1.8, 0.45, 2.0, 0.1
    ht = 0.0001 
    tol = 1e-5
    max_iter = 100000

    # Grid (Ensure enough points for fat tails/small L)
    N = max(60, floor(Int, 20 * L))
    x_domain = range(-L, L, length=N)
    
    # Build Operator
    grid_x = [i for i in x_domain, j in x_domain]
    grid_y = [j for i in x_domain, j in x_domain]
    k_mat = kernel_func.(grid_x, grid_y)
    op_int = IntegralOperator(x_domain, k_mat)
    
    # Laplacian for water
    dx = x_domain[2] - x_domain[1]
    
    # Initial Condition
    v_star = (A + sqrt(A^2 - 4*B^2)) / (2 * B)
    w_star = (2 * B^2) / (A + sqrt(A^2 - 4*B^2))
    
    u = ones(N) .* v_star
    v = ones(N) .* w_star
    
    # BCs
    u[begin]=u[end]=0; v[begin]=v[end]=0;

    for j in 1:max_iter
        u_spatial = d_u * compute(op_int, u) - d_u * u
        
        w_spatial = zeros(N)
        w_spatial[2:end-1] = (v[3:end] - 2*v[2:end-1] + v[1:end-2]) / dx^2
        w_spatial .*= d_v
        
        u_reac = u.^2 .* v .- B .* u
        v_reac = .- u.^2 .* v .- v .+ A
        
        u_new = u .+ ht .* (u_spatial .+ u_reac)
        v_new = v .+ ht .* (w_spatial .+ v_reac)
        
        u_new[begin]=u_new[end]=0
        v_new[begin]=v_new[end]=0
        
        # Fast exit: Extinction
        mean_biomass = sum(u_new)/N
        if mean_biomass < 0.05
            return false 
        end
        
        # Fast exit: Convergence to survival
        if norm(u_new - u) < tol && mean_biomass > 0.1
            return true 
        end
        
        u, v = u_new, v_new
    end
    
    return false
end

# =========================================================================
# 4. EXPERIMENT EXECUTION
# =========================================================================

println("Starting Experiment 5.4...")

# Range of shapes: Beta < 2 (Fat), Beta = 2 (Gaussian), Beta > 2 (Thin)
betas = range(1, 4, 100)

m4_results = Float64[]
lcrit_results = Float64[]

# Search range for L_crit
L_search_min = 0.5
L_search_max = 8.0
bisection_tol = 0.02

@showprogress "Scanning Betas..." for beta in betas
    k_func, m4 = get_gnd_kernel_data(beta)
    push!(m4_results, m4)
    
    # Bisection Search for L_crit
    low = L_search_min
    high = L_search_max
    L_crit_est = NaN
    
    for _ in 1:12 # ~12 iterations for precision ~0.002
        mid = (low + high) / 2
        survives = check_survival(mid, k_func)
        
        if survives
            high = mid # Try smaller L
            L_crit_est = mid
        else
            low = mid # Need larger L
        end
    end
    
    push!(lcrit_results, L_crit_est)
end

# =========================================================================
# 5. PLOTTING
# =========================================================================

f_size = 12

p = plot(
    m4_results, lcrit_results,
    title = "Impact of Kernel Kurtosis on Resilience",
    xaxis = ("4th Moment \$M_4\$ (Kurtosis)", :log),
    yaxis = ("Critical Patch Half-width \$L_{crit}\$"),
    label = "Non-local Model",
    color = :purple,
    lw = 2,
    marker = :circle,
    markersize = 5,
    markerstrokecolor = :black,
    markerstrokewidth = 0.5,
    gridalpha = 0.4,
    titlefontsize = f_size,
    tickfontsize = f_size,
    guidefontsize = f_size,
    legendfontsize = 10,
    size = (700, 500),
    margin = 5Plots.mm
)

# Reference: Gaussian Kurtosis = 3.0
vline!(p, [3.0], linestyle=:dash, color=:grey, label="Gaussian (\$M_4=3\$)")

display(p)