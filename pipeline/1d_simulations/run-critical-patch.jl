# ==============================================================================
# IMPORTS & PACKAGES
# ==============================================================================
using LinearAlgebra
using Plots
using ProgressMeter
using LaTeXStrings
using SpecialFunctions # Required for gamma and loggamma
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

"""
Calculates the analytical fourth moment of the calibrated kernel:
    M_4(p) = int z^4 J_p(z) dz = [ Gamma(5/p) * Gamma(1/p) ] / [ Gamma(3/p)^2 ]
Calculated in log-space to prevent overflow for small p (fat tails).
"""
function get_fourth_moment(p::Float64)
    log_m4 = loggamma(5.0 / p) + loggamma(1.0 / p) - 2.0 * loggamma(3.0 / p)
    return exp(log_m4)
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
                        p::Float64, A::Float64, B::Float64, d_u::Float64, d_v::Float64, ht::Float64, tol::Float64)
    
    domain_omega = range(-L, L, length=N)
    integral_operator = IntegralOperator(domain_omega, p)
    laplacian_fd = LaplacianFD(domain_omega)

    u_old = copy(u_init)
    v_old = copy(v_init)

    # Strictly enforce Dirichlet boundary conditions
    u_old[begin] = u_old[end] = 0.0
    v_old[begin] = v_old[end] = 0.0
    
    for j in 1:300000 
        u_spatial_term = d_u * compute(integral_operator, u_old) - d_u * u_old
        v_spatial_term = d_v * compute(laplacian_fd, v_old)

        u_new = u_old .+ ht .* (u_spatial_term .+ u_old.^2 .* v_old .- B .* u_old)
        v_new = v_old .+ ht .* (v_spatial_term .- u_old.^2 .* v_old .- v_old .+ A)

        u_new[begin] = u_new[end] = 0.0
        v_new[begin] = v_new[end] = 0.0

        if norm(u_new - u_old) < tol
            return u_new # Return the full profile upon convergence
        end

        if any(isnan, u_new) || any(isnan, v_new) || any(isinf, u_new) || any(isinf, v_new)
            return nothing
        end

        u_old, v_old = copy(u_new), copy(v_new)
    end
    
    return nothing # Max iterations reached
end

# ==============================================================================
# 4. MAIN EXPERIMENTS & PLOTTING (Wrapped to avoid global scope issues)
# ==============================================================================

let
    println("\nStarting Critical Patch Size vs Kernel Exponent and Fourth Moment...")

    # --- Global Ecosystem Parameters ---
    A, B = 1.8, 0.45
    d_u, d_v = 2.0, 0.1
    ht = 0.0001
    tol = 1e-6
    zero_threshold = 0.1

    uniform_u = (A + sqrt(A^2 - 4*B^2)) / (2 * B)
    uniform_v = (2 * B^2) / (A + sqrt(A^2 - 4*B^2))

    # --- Scanning Ranges ---
    p_vals = 2 .^ range(log2(0.25), log2(4.0), length=40)
    L_vals = 10 .^ range(0.0, 2.0, length=200)

    # Arrays to store successful data points
    valid_p = Float64[]
    valid_m4 = Float64[]
    valid_Lcrit = Float64[]

    @showprogress "Scanning 'p' exponents: " for p in p_vals
        L_crit = NaN
        
        for L in L_vals
            N = Int(round(20 + 2*L))
            u_init = ones(N) .* uniform_u
            v_init = ones(N) .* uniform_v
            
            u_final = run_simulation(L, N, u_init, v_init, p, A, B, d_u, d_v, ht, tol)
            
            # Check if the solution is valid and biomass survived
            if u_final !== nothing
                biomass = sum(u_final) / N
                if biomass > zero_threshold
                    L_crit = L
                    break # We found the critical patch size, stop L scan
                end
            end
        end
        
        # Store only if a valid L_crit was found
        if !isnan(L_crit)
            push!(valid_p, p)
            push!(valid_m4, get_fourth_moment(p))
            push!(valid_Lcrit, L_crit)
        end
    end

    # ==============================================================================
    # 5. DATA EXPORT
    # ==============================================================================

    current_dir = @__DIR__
    project_root = normpath(joinpath(current_dir, "..", ".."))
    output_dir = joinpath(project_root, "data", "1d_simulations", "critical_patch_size")

    mkpath(output_dir)

    output_file_csv = joinpath(output_dir, "Lcrit_M4_vs_p.csv")
    open(output_file_csv, "w") do io
        write(io, "p,M4,L_crit\n")
        writedlm(io, hcat(valid_p, valid_m4, valid_Lcrit), ',')
    end
    println("\nData successfully saved to: $output_file_csv")

# ==============================================================================
    # 6. VISUALIZATION (Two side-by-side plots)
    # ==============================================================================
    println("Generating minimalist Tufte-style plots...")

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
        titlefontsize=global_font_size,
        legend=false
    )

    # --- Plot 1: L_crit vs p ---
    # Zmieniono skalę Y na liniową i usunięto hardkodowane ticki
    plot_Lcrit = plot(
        valid_p, valid_Lcrit,
        color=:black,
        xscale=:log2,
        xticks=([0.5, 1.0, 2.0, 4.0], ["0.5", "1.0", "2.0", "4.0"]),
        xlabel=L"Kernel Exponent $p$",
        ylabel=L"Critical Patch Size $L_{crit}$",
        title="Impact of Dispersal Tail on Persistence"
    )
    scatter!(plot_Lcrit, valid_p, valid_Lcrit, color=:black, markersize=5)

    # --- Plot 2: M4 vs p ---
    # Usunięto skalę logarytmiczną na osi Y i zmieniono kolor na czarny
    plot_M4 = plot(
        valid_p, valid_m4,
        color=:black,
        xscale=:log2,
        xticks=([0.5, 1.0, 2.0, 4.0], ["0.5", "1.0", "2.0", "4.0"]),
        xlabel=L"Kernel Exponent $p$",
        ylabel=L"Fourth Moment $\int_\mathbb{R} J_p(z)z^4 dz$",
        title="Dispersal Tail Heaviness"
    )
    scatter!(plot_M4, valid_p, valid_m4, color=:black, markersize=5)

    # Combine plots side-by-side
    final_plot = plot(
        plot_Lcrit, plot_M4, 
        layout = (1, 2),
        size = (1000, 450),
        bottom_margin = 5Plots.mm,
        left_margin = 5Plots.mm
    )

    plot_file = joinpath(output_dir, "Lcrit_and_M4_vs_p.png")
    savefig(final_plot, plot_file)
    println("Combined plot successfully saved to: $plot_file")

    display(final_plot)
    println("Done.")
end