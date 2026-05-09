# ==============================================================================
# IMPORTS & PARALLEL SETUP
# ==============================================================================
using Distributed

# Liczba procesów roboczych - dostosuj do swojego procesora
const NUM_WORKERS = 6
if nprocs() < NUM_WORKERS + 1
    addprocs(NUM_WORKERS)
end

using Plots
using ProgressMeter
using LaTeXStrings
using SharedArrays
using Random

# ==============================================================================
# 1. MATHEMATICAL KERNEL CALIBRATION & OPERATORS (Broadcasted)
# ==============================================================================
@everywhere begin
    using LinearAlgebra
    using SpecialFunctions

    function get_kernel_constants(p::Float64)
        log_lambda_p = (p / 2.0) * (loggamma(3.0 / p) - loggamma(1.0 / p))
        lambda_p = exp(log_lambda_p)
        log_C_p = log(p) + (1.0 / p) * log_lambda_p - log(2.0) - loggamma(1.0 / p)
        C_p = exp(log_C_p)
        return C_p, lambda_p
    end

    function NeumannNonlocalOperator(x::AbstractVector, p::Float64)
        N = length(x)
        dx = x[2] - x[1]
        C_p, lambda_p = get_kernel_constants(p)
        
        weights = ones(N) .* dx
        weights[begin] *= 0.5
        weights[end] *= 0.5
        
        L_mat = zeros(Float64, N, N)
        
        for i in 1:N
            row_sum = 0.0
            for j in 1:N
                if i != j
                    dist = abs(x[i] - x[j])
                    val = C_p * exp(-lambda_p * dist^p) * weights[j]
                    L_mat[i, j] = val
                    row_sum += val
                end
            end
            L_mat[i, i] = -row_sum # Enforce exact mass conservation
        end
        
        return L_mat, weights
    end

    # ==============================================================================
    # 2. REACTION KINETICS (Allee Effect)
    # ==============================================================================
    # f(v) = r * v * (1 - v/K) * (v/A - 1), with K = 1.0
    reaction_f(v, r, A) = r * v * (1.0 - v) * (v/A - 1.0)

    # ==============================================================================
    # 3. CORE SIMULATION LOGIC
    # ==============================================================================
    function simulate_until_steady(dv, v0, L_N, W, r, A_allee, ht, max_time, tol, idx_peak, idx_valley; return_timeseries=false)
        N = length(v0)
        v_curr = copy(v0)
        
        # IMEX Matrix: (I - ht * d_v * L_N) * v_new = RHS
        I_mat = Matrix{Float64}(I, N, N)
        M_imex = I_mat .- (ht * dv) .* L_N
        
        # [OPTYMALIZACJA] Faktoryzacja LU wykonywana raz, przyspiesza pętlę >10x
        M_fact = lu(M_imex) 
        
        num_steps = Int(max_time / ht)
        
        time_pts = Float64[0.0]
        mean_v = Float64[A_allee]
        peak_v = Float64[v0[idx_peak]]
        valley_v = Float64[v0[idx_valley]]
        
        for step in 1:num_steps
            # Explicit Reaction
            rhs = v_curr .+ ht .* reaction_f.(v_curr, r, A_allee)
            
            # Implicit Dispersal (fast back-substitution via LU)
            v_next = M_fact \ rhs
            
            diff = maximum(abs.(v_next .- v_curr))
            v_curr .= v_next
            
            if return_timeseries && step % 10 == 0
                push!(time_pts, step * ht)
                push!(mean_v, sum(v_curr .* W) / sum(W))
                push!(peak_v, v_curr[idx_peak])
                push!(valley_v, v_curr[idx_valley])
            end
            
            if diff < tol
                break
            end
        end
        
        final_mean = sum(v_curr .* W) / sum(W)
        final_max = maximum(v_curr)
        final_min = minimum(v_curr)
        
        if return_timeseries
            return time_pts, mean_v, peak_v, valley_v
        else
            return final_max, final_min, final_mean
        end
    end
end

# ==============================================================================
# 4. MAIN EXPERIMENT SCRIPT
# ==============================================================================

function run_experiment()
    # --- Parameters ---
    r = 1.0
    A_allee = 0.6
    K_cap = 1.0
    p = 2.0
    L = 5.0
    N = 300
    ht = 5e-3
    tol = 1e-6
    max_time = 150.0
    
    x = collect(range(-L, L, length=N))
    L_N, W = NeumannNonlocalOperator(x, p)
    
    # --------------------------------------------------------------------------
    # INITIAL CONDITION CONSTRUCTION (Mean exactly A_allee)
    # --------------------------------------------------------------------------
    Random.seed!(42) # For reproducible noise
    
    # Base perturbation: 3 distinct patches (peaks and valleys)
    base_pert = 0.15 .* cos.(3.0 .* pi .* x ./ L)
    
    # Add some random noise
    noise = 0.02 .* (rand(N) .- 0.5)
    
    # Combine and rigorously enforce the exact mean
    pert = base_pert .+ noise
    pert_mean = sum(pert .* W) / sum(W)
    pert_corrected = pert .- pert_mean # Now strictly 0 mean
    
    v0 = A_allee .+ pert_corrected
    
    # Find indices for specific Peak and Valley to track
    idx_peak = argmax(v0)
    idx_valley = argmin(v0)
    
    println("\n--- Experiment Setup ---")
    println("Initial Mean: ", sum(v0 .* W) / sum(W), " (Target: $A_allee)")
    println("Tracking Peak at x = $(round(x[idx_peak], digits=2)), Initial v = $(round(v0[idx_peak], digits=3))")
    println("Tracking Valley at x = $(round(x[idx_valley], digits=2)), Initial v = $(round(v0[idx_valley], digits=3))")

    # --------------------------------------------------------------------------
    # EXPERIMENT 1: STEADY STATE SWEEP vs d_v (ASYNC)
    # --------------------------------------------------------------------------
    println("\n>>> PHASE 1: Running Parameter Sweep (Async)...")
    dv_range = 10 .^ range(log10(0.01), log10(15.0), length=40)
    
    final_maxs = SharedVector{Float64}(length(dv_range))
    final_mins = SharedVector{Float64}(length(dv_range))
    final_means = SharedVector{Float64}(length(dv_range))
    
    @showprogress 1 "Sweeping dv: " @distributed for i in 1:length(dv_range)
        dv = dv_range[i]
        f_max, f_min, f_mean = simulate_until_steady(dv, v0, L_N, W, r, A_allee, ht, max_time, tol, idx_peak, idx_valley, return_timeseries=false)
        final_maxs[i] = f_max
        final_mins[i] = f_min
        final_means[i] = f_mean
    end
    
    # --------------------------------------------------------------------------
    # EXPERIMENT 2: TRANSIENT TIME EVOLUTIONS (ASYNC ON WORKERS)
    # --------------------------------------------------------------------------
    println("\n>>> PHASE 2: Running Transient Evolutions (Async)...")
    dv_slow = 0.1
    dv_fast = 6.0
    
    # Send tasks to the worker pool
    future_slow = @spawnat :any simulate_until_steady(dv_slow, v0, L_N, W, r, A_allee, ht, max_time, tol, idx_peak, idx_valley, return_timeseries=true)
    future_fast = @spawnat :any simulate_until_steady(dv_fast, v0, L_N, W, r, A_allee, ht, max_time, tol, idx_peak, idx_valley, return_timeseries=true)
    
    # Fetch results once finished
    t_slow, mean_slow, peak_slow, val_slow = fetch(future_slow)
    t_fast, mean_fast, peak_fast, val_fast = fetch(future_fast)
    
    # ==============================================================================
    # 4. TUFTE-STYLE PLOTTING
    # ==============================================================================
    println("\n>>> PHASE 3: Generating Tufte-style plots...")
    
    global_font_size = 11
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
        titlefontsize=global_font_size
    )
    
    # --- PANEL 1: Parameter Sweep ---
    p_sweep = plot(
        xscale=:log10,
        xlabel=L"Dispersal Rate $d_v$",
        ylabel=L"Stationary Biomass $v(x,\infty)$",
        title="Steady States vs Dispersal",
        legend=:bottomleft,
        xticks=([0.01, 0.1, 1.0, 10.0], ["0.01", "0.1", "1.0", "10.0"])
    )
    
    hline!(p_sweep, [K_cap], color=:gray, ls=:dash, lw=1.5, label="Capacity K=1.0")
    hline!(p_sweep, [A_allee], color=:red, ls=:dash, lw=1.5, label="Allee A=0.6")
    hline!(p_sweep, [0.0], color=:black, ls=:solid, lw=1.0, label=false)
    
    plot!(p_sweep, dv_range, final_maxs, label=L"Spatial Max $\max(v)$", color=:dodgerblue, ls=:solid)
    plot!(p_sweep, dv_range, final_mins, label=L"Spatial Min $\min(v)$", color=:darkorange, ls=:solid)
    plot!(p_sweep, dv_range, final_means, label=L"Global Mean $v_\Omega$", color=:black, ls=:dot, lw=3.0)
    ylims!(p_sweep, -0.05, 1.1)

    # --- PANEL 2: Transient (Slow Dispersal) ---
    p_slow = plot(
        xlabel=L"Time $t$",
        ylabel=L"Transient Biomass $v(x,t)$",
        title=L"Fragmentation ($d_v = 0.1$)",
        legend=:right
    )
    
    hline!(p_slow, [K_cap], color=:gray, ls=:dash, lw=1.5, label=false)
    hline!(p_slow, [A_allee], color=:red, ls=:dash, lw=1.5, label=false)
    hline!(p_slow, [0.0], color=:black, ls=:solid, lw=1.0, label=false)
    
    plot!(p_slow, t_slow, peak_slow, label="Peak (Started > A)", color=:dodgerblue, ls=:solid)
    plot!(p_slow, t_slow, val_slow, label="Valley (Started < A)", color=:darkorange, ls=:solid)
    plot!(p_slow, t_slow, mean_slow, label="Global Mean", color=:black, ls=:dot, lw=3.0)
    
    ylims!(p_slow, -0.05, 1.1)
    xlims!(p_slow, 0.0, maximum(t_slow))

    # --- PANEL 3: Transient (Fast Dispersal) ---
    p_fast = plot(
        xlabel=L"Time $t$",
        ylabel=L"Transient Biomass $v(x,t)$",
        title=L"Homogenization ($d_v = 6.0$)",
        legend=false
    )
    
    hline!(p_fast, [K_cap], color=:gray, ls=:dash, lw=1.5, label=false)
    hline!(p_fast, [A_allee], color=:red, ls=:dash, lw=1.5, label=false)
    hline!(p_fast, [0.0], color=:black, ls=:solid, lw=1.0, label=false)
    
    plot!(p_fast, t_fast, peak_fast, label="Peak", color=:dodgerblue, ls=:solid)
    plot!(p_fast, t_fast, val_fast, label="Valley", color=:darkorange, ls=:solid)
    plot!(p_fast, t_fast, mean_fast, label="Global Mean", color=:black, ls=:dot, lw=3.0)
    
    # Zoom in slightly on time to see the rapid collapse
    ylims!(p_fast, -0.05, 1.1)
    xlims!(p_fast, 0.0, 10.0)

    # Combine Layout
    final_plot = plot(
        p_sweep, p_slow, p_fast, 
        layout = (1, 3),
        size = (1500, 450),
        bottom_margin = 8Plots.mm,
        left_margin = 8Plots.mm
    )
    
    # Setup directories and save
    current_dir = @__DIR__
    project_root = normpath(joinpath(current_dir, "..", ".."))
    output_dir = joinpath(project_root, "data", "1d_simulations", "convergence")
    mkpath(output_dir)
    
    plot_path = joinpath(output_dir, "convergence_mean_mass_dynamics.png")
    savefig(final_plot, plot_path)
    println("Plot successfully saved to: $plot_path")
    display(final_plot)
end

run_experiment()