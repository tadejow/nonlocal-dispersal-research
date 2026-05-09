# ==============================================================================
# SCRIPT 1: PHASE TRANSITION (1D)
# ==============================================================================
using Distributed
const NUM_WORKERS = 6
if nprocs() < NUM_WORKERS + 1
    addprocs(NUM_WORKERS)
end

using Plots
using ProgressMeter
using LaTeXStrings
using SharedArrays
using Random

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
            L_mat[i, i] = -row_sum 
        end
        return L_mat, weights
    end

    reaction_f(v, r, A) = r * v * (1.0 - v) * (v/A - 1.0)
    reaction_df(v, r, A) = r * ( -3.0/A * v^2 + 2.0*(1.0/A + 1.0)*v - 1.0 )

    function get_max_df(r, A)
        v_ext = (1.0 + A) / 3.0
        vals = [abs(reaction_df(0.0, r, A)), abs(reaction_df(1.0, r, A)), abs(reaction_df(v_ext, r, A))]
        return maximum(vals)
    end

    function simulate_steady_state(dv, v0, L_N, W, r, A, ht, max_time, tol)
        N = length(v0)
        v_curr = copy(v0)
        M_imex = Matrix{Float64}(I, N, N) .- (ht * dv) .* L_N
        M_fact = lu(M_imex) 
        
        num_steps = Int(max_time / ht)
        for step in 1:num_steps
            rhs = v_curr .+ ht .* reaction_f.(v_curr, r, A)
            v_next = M_fact \ rhs
            if maximum(abs.(v_next .- v_curr)) < tol
                v_curr .= v_next
                break
            end
            v_curr .= v_next
        end
        return maximum(v_curr), minimum(v_curr), sum(v_curr .* W)/sum(W)
    end
end

function run_phase_transition()
    r = 1.0; A = 0.6; p = 2.0; N = 200; ht = 0.01; max_time = 150.0; tol = 1e-6
    L_fixed = 5.0
    
    # 1. 1D Sweep for Line Plot
    dv_range = 10 .^ range(log10(0.01), log10(15.0), length=50)
    x_fixed = collect(range(-L_fixed, L_fixed, length=N))
    L_N_fixed, W_fixed = NeumannNonlocalOperator(x_fixed, p)
    
    Random.seed!(42)
    base_pert = 0.15 .* cos.(3.0 .* pi .* x_fixed ./ L_fixed) .+ 0.02 .* (rand(N) .- 0.5)
    pert_corrected = base_pert .- (sum(base_pert .* W_fixed) / sum(W_fixed))
    v0_fixed = A .+ pert_corrected

    final_maxs = SharedVector{Float64}(length(dv_range))
    final_mins = SharedVector{Float64}(length(dv_range))
    final_means = SharedVector{Float64}(length(dv_range))
    
    println("Phase 1: 1D Sweep for Line Plot...")
    @showprogress 1 "dv sweep: " @distributed for i in 1:length(dv_range)
        f_max, f_min, f_mean = simulate_steady_state(dv_range[i], v0_fixed, L_N_fixed, W_fixed, r, A, ht, max_time, tol)
        final_maxs[i] = f_max
        final_mins[i] = f_min
        final_means[i] = f_mean
    end
    
    # 2. 2D Sweep for Heatmap
    L_range = collect(range(1.0, 10.0, length=30))
    dv_hm_range = 10 .^ range(log10(0.01), log10(15.0), length=40)
    hm_means = SharedMatrix{Float64}(length(dv_hm_range), length(L_range))
    
    # Theoretical curve
    theoretical_dv = zeros(length(L_range))
    max_df = get_max_df(r, A)
    
    println("Phase 2: 2D Sweep for Heatmap...")
    @showprogress 1 "Heatmap: " for j in 1:length(L_range)
        curr_L = L_range[j]
        x_curr = collect(range(-curr_L, curr_L, length=N))
        L_N_curr, W_curr = NeumannNonlocalOperator(x_curr, p)
        
        beta_1 = sort(eigvals(L_N_curr), rev=true)[2]
        theoretical_dv[j] = -max_df / beta_1
        
        base_pert_curr = 0.15 .* cos.(3.0 .* pi .* x_curr ./ curr_L) .+ 0.02 .* (rand(N) .- 0.5)
        pert_corr_curr = base_pert_curr .- (sum(base_pert_curr .* W_curr) / sum(W_curr))
        v0_curr = A .+ pert_corr_curr
        
        @sync @distributed for i in 1:length(dv_hm_range)
            _, _, f_mean = simulate_steady_state(dv_hm_range[i], v0_curr, L_N_curr, W_curr, r, A, ht, max_time, tol)
            hm_means[i, j] = f_mean
        end
    end
    hm_means_local = Matrix(hm_means)

    # 3. Plotting
    global_font = 12
    default(fontfamily="sans-serif", grid=true, gridalpha=0.3, gridstyle=:dash, framestyle=:axes, tickdirection=:out, linewidth=2.5, guidefontsize=global_font, tickfontsize=global_font, legendfontsize=global_font, titlefontsize=global_font)

    p_line = plot(xscale=:log10, xlabel=L"Dispersal Rate $d_v$", ylabel=L"Stationary Biomass $v(x,\infty)$", title="Steady States ($L = 5.0$)", legend=:left)
    hline!(p_line, [1.0], color=:gray, ls=:dash, lw=1.5, label="Capacity K=1.0")
    hline!(p_line, [A], color=:red, ls=:dash, lw=1.5, label="Allee A=0.6")
    plot!(p_line, dv_range, final_maxs, label=L"Spatial Max $\max(v)$", color=:dodgerblue, ls=:solid)
    plot!(p_line, dv_range, final_mins, label=L"Spatial Min $\min(v)$", color=:darkorange, ls=:solid)
    plot!(p_line, dv_range, final_means, label=L"Global Mean $v_\Omega$", color=:black, ls=:dot, lw=3.0)
    ylims!(p_line, -0.05, 1.1)

    cmap = cgrad([:white, :forestgreen])
    p_hm = heatmap(L_range, dv_hm_range, hm_means_local, yscale=:log10, xlabel=L"Domain Half-Width $L$", ylabel=L"Dispersal Rate $d_v$", title="Mean Biomass Heatmap", color=cmap, colorbar_title=L"$v_\Omega(\infty)$")
    plot!(p_hm, L_range, theoretical_dv, color=:white, ls=:dash, lw=3.0, label="Theoretical Boundary")
    annotate!(p_hm, 3.0, 10.0, text("Homogenization\n(Extinction)", :white, :center, 10))
    annotate!(p_hm, 8.0, 0.05, text("Fragmentation\n(Survival)", :black, :center, 10))
    ylims!(p_hm, 0.01, 15.0)

    final_plot = plot(p_line, p_hm, layout=(1,2), size=(1200, 500), bottom_margin=8Plots.mm, left_margin=8Plots.mm)
    
    out_dir = joinpath(@__DIR__, "..", "..", "data", "1d_simulations", "convergence")
    mkpath(out_dir)
    fname = joinpath(out_dir, "phase_transition_r$(r)_A$(A)_p$(p)_N$(N)_combined.png")
    savefig(final_plot, fname)
    println("Saved to: $fname")
    display(final_plot)
end

run_phase_transition()