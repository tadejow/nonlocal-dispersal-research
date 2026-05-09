# ==============================================================================
# SCRIPT 3: FRAGMENTATION (2D) - ASYNC SPAWNED
# ==============================================================================
using Distributed

const NUM_WORKERS = 2
if nprocs() < NUM_WORKERS + 1
    addprocs(NUM_WORKERS)
end

using Plots
using LaTeXStrings

@everywhere begin
    using LinearAlgebra
    using SpecialFunctions
    using Random

    function get_kernel_constants_2d(p::Float64)
        log_lambda_p = (p / 2.0) * (loggamma(4.0 / p) - loggamma(2.0 / p))
        lambda_p = exp(log_lambda_p)
        log_C_p = log(p) + (2.0 / p) * log_lambda_p - log(2.0*pi) - loggamma(2.0 / p)
        C_p = exp(log_C_p)
        return C_p, lambda_p
    end

    function NeumannOperator2D(x::AbstractVector, p::Float64)
        N = length(x)
        dx = x[2] - x[1]
        C_p, lambda_p = get_kernel_constants_2d(p)
        
        W_1d = ones(N) .* dx
        W_1d[begin] *= 0.5; W_1d[end] *= 0.5
        
        N2 = N^2
        L_mat = zeros(Float64, N2, N2)
        W_2d = zeros(Float64, N2)
        
        for i in 1:N, j in 1:N
            row = (i-1)*N + j
            W_2d[row] = W_1d[i] * W_1d[j]
        end
        
        for i1 in 1:N, j1 in 1:N
            row = (i1-1)*N + j1
            row_sum = 0.0
            for i2 in 1:N, j2 in 1:N
                col = (i2-1)*N + j2
                if row != col
                    dist = sqrt((x[i1] - x[i2])^2 + (x[j1] - x[j2])^2)
                    val = C_p * exp(-lambda_p * dist^p) * W_2d[col]
                    L_mat[row, col] = val
                    row_sum += val
                end
            end
            L_mat[row, row] = -row_sum
        end
        return L_mat, W_2d
    end

    reaction_f(v, r, A) = r * v * (1.0 - v) * (v/A - 1.0)
end

function run_fragmentation_2d_async()
    r = 1.0; A = 0.6; p = 2.0; L = 5.0; N = 50; ht = 0.05; max_time = 50.0; dv = 0.1
    
    println(">>> Spawning 2D simulation to a worker (may take a moment to factorize LU)...")
    
    future_sim = @spawnat :any begin
        x = collect(range(-L, L, length=N))
        L_N, W = NeumannOperator2D(x, p)
        
        Random.seed!(42)
        base_pert = 0.15 .* [cos(2.0*pi*xi/L)*cos(2.0*pi*yj/L) for xi in x, yj in x]
        noise = 0.02 .* (rand(N, N) .- 0.5)
        
        v0_mat = base_pert .+ noise
        v0 = vec(v0_mat)
        v0 .= A .+ (v0 .- (sum(v0 .* W) / sum(W))) # Force exact mean
        
        idx_peak = argmax(v0)
        idx_valley = argmin(v0)
        
        M_fact = lu(Matrix{Float64}(I, N^2, N^2) .- (ht * dv) .* L_N)
        v_curr = copy(v0)
        
        t_pts = Float64[0.0]; mean_v = Float64[A]
        peak_v = Float64[v0[idx_peak]]; val_v = Float64[v0[idx_valley]]
        
        for step in 1:Int(max_time/ht)
            v_next = M_fact \ (v_curr .+ ht .* reaction_f.(v_curr, r, A))
            v_curr .= v_next
            if step % 10 == 0
                push!(t_pts, step * ht)
                push!(mean_v, sum(v_curr .* W) / sum(W))
                push!(peak_v, v_curr[idx_peak])
                push!(val_v, v_curr[idx_valley])
            end
        end
        
        return x, v0, v_curr, t_pts, peak_v, val_v, mean_v
    end
    
    println(">>> Waiting for 2D results...")
    x, v0, v_curr, t_pts, peak_v, val_v, mean_v = fetch(future_sim)
    
    v_final_mat = reshape(v_curr, N, N)
    v0_mat_plot = reshape(v0, N, N)
    
    println(">>> Generating plots...")
    global_font = 12
    default(fontfamily="sans-serif", grid=false, framestyle=:box, tickdirection=:out, guidefontsize=global_font, tickfontsize=global_font, legendfontsize=global_font, titlefontsize=global_font)
    
    p_trans = plot(xlabel=L"Time $t$", ylabel=L"Biomass $v$", title=L"Fragmentation ($d_v=0.1$)", legend=:right, grid=true, gridstyle=:dash, linewidth=2.5)
    hline!(p_trans, [1.0], color=:gray, ls=:dash, lw=1.5, label=false)
    hline!(p_trans, [A], color=:red, ls=:dash, lw=1.5, label=false)
    plot!(p_trans, t_pts, peak_v, label="2D Peak", color=:dodgerblue)
    plot!(p_trans, t_pts, val_v, label="2D Valley", color=:darkorange)
    plot!(p_trans, t_pts, mean_v, label="Global Mean", color=:black, ls=:dot, lw=3.0)
    ylims!(p_trans, -0.05, 1.1)

    cmap = cgrad([:white, :forestgreen])
    p_init = heatmap(x, x, v0_mat_plot', aspect_ratio=1, title=L"Initial State $v(x,y,0)$", color=cmap, clims=(0, 1.0))
    p_final = heatmap(x, x, v_final_mat', aspect_ratio=1, title=L"Final Pattern $v(x,y,\\infty)$", color=cmap, clims=(0, 1.0))

    final_plot = plot(p_trans, p_init, p_final, layout=(1,3), size=(1500, 450), bottom_margin=8Plots.mm)
    
    out_dir = joinpath(@__DIR__, "..", "..", "data", "1d_simulations", "convergence")
    mkpath(out_dir)
    fname = joinpath(out_dir, "fragmentation_2d_r$(r)_A$(A)_L$(L)_dv$(dv)_combined.png")
    savefig(final_plot, fname)
    display(final_plot)
end

run_fragmentation_2d_async()