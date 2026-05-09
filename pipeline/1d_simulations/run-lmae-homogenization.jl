# ==============================================================================
# SCRIPT 2: HOMOGENIZATION (1D) - ASYNC SPAWNED
# ==============================================================================
using Distributed

const NUM_WORKERS = 6
if nprocs() < NUM_WORKERS + 1
    addprocs(NUM_WORKERS)
end

using Plots
using LaTeXStrings

@everywhere begin
    using LinearAlgebra
    using SpecialFunctions
    using Random

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
        return maximum([abs(reaction_df(0.0, r, A)), abs(reaction_df(1.0, r, A)), abs(reaction_df(v_ext, r, A))])
    end
end

function run_homogenization_async()
    r = 1.0; A = 0.6; p = 2.0; L = 5.0; N = 300; ht = 0.005; max_time = 10.0; dv = 6.0
    
    println(">>> Spawning simulation task to a worker...")
    
    # Spawn the heavy simulation logic to any available worker
    future_sim = @spawnat :any begin
        x = collect(range(-L, L, length=N))
        L_N, W = NeumannNonlocalOperator(x, p)
        
        beta_1 = sort(eigvals(L_N), rev=true)[2]
        max_df = get_max_df(r, A)
        sigma = dv * beta_1 + max_df
        
        Random.seed!(42)
        base_pert = 0.15 .* cos.(3.0 .* pi .* x ./ L) .+ 0.02 .* (rand(N) .- 0.5)
        v0 = A .+ (base_pert .- (sum(base_pert .* W) / sum(W)))
        
        idx_peak = argmax(v0)
        idx_valley = argmin(v0)
        
        v_curr = copy(v0)
        M_fact = lu(Matrix{Float64}(I, N, N) .- (ht * dv) .* L_N)
        
        t_pts = Float64[0.0]; mean_v = Float64[A]
        peak_v = Float64[v0[idx_peak]]; val_v = Float64[v0[idx_valley]]
        diff_norms = Float64[sqrt(sum(((v0 .- A).^2) .* W))]
        
        for step in 1:Int(max_time/ht)
            v_next = M_fact \ (v_curr .+ ht .* reaction_f.(v_curr, r, A))
            v_curr .= v_next
            if step % 5 == 0
                t = step * ht
                m = sum(v_curr .* W) / sum(W)
                push!(t_pts, t)
                push!(mean_v, m)
                push!(peak_v, v_curr[idx_peak])
                push!(val_v, v_curr[idx_valley])
                push!(diff_norms, sqrt(sum(((v_curr .- m).^2) .* W)))
            end
        end
        
        theoretical = diff_norms[1] .* exp.(sigma .* t_pts)
        return t_pts, peak_v, val_v, mean_v, diff_norms, theoretical, sigma
    end
    
    println(">>> Waiting for results...")
    t_pts, peak_v, val_v, mean_v, diff_norms, theoretical, sigma = fetch(future_sim)
    
    println("Simulation finished. Sigma = $sigma (Expected < 0)")
    println(">>> Generating plots...")
    
    global_font = 12
    default(fontfamily="sans-serif", grid=true, gridalpha=0.3, gridstyle=:dash, framestyle=:axes, tickdirection=:out, linewidth=2.5, guidefontsize=global_font, tickfontsize=global_font, legendfontsize=global_font, titlefontsize=global_font)
    
    p_trans = plot(xlabel=L"Time $t$", ylabel=L"Biomass $v(x,t)$", title="Homogenization ($d_v=6.0$)", legend=:right)
    hline!(p_trans, [1.0], color=:gray, ls=:dash, lw=1.5, label=false)
    hline!(p_trans, [A], color=:red, ls=:dash, lw=1.5, label=false)
    plot!(p_trans, t_pts, peak_v, label="Peak (Started > A)", color=:dodgerblue)
    plot!(p_trans, t_pts, val_v, label="Valley (Started < A)", color=:darkorange)
    plot!(p_trans, t_pts, mean_v, label="Global Mean", color=:black, ls=:dot, lw=3.0)
    ylims!(p_trans, -0.05, 1.1)

    p_decay = plot(xlabel=L"Time $t$", ylabel=L"Norm $\|v(\cdot,t) - v_\Omega\|_2$", title="Convergence to Mean", yscale=:log10, legend=:bottomleft)
    plot!(p_decay, t_pts, theoretical, label="Theoretical Bound", color=:black, ls=:dash)
    plot!(p_decay, t_pts, diff_norms, label="Numerical Difference", color=:dodgerblue)
    ylims!(p_decay, 1e-10, 1e0)

    final_plot = plot(p_trans, p_decay, layout=(1,2), size=(1200, 500), bottom_margin=8Plots.mm, left_margin=8Plots.mm)
    
    out_dir = joinpath(@__DIR__, "..", "..", "data", "1d_simulations", "convergence")
    mkpath(out_dir)
    fname = joinpath(out_dir, "homogenization_1d_r$(r)_A$(A)_L$(L)_dv$(dv)_combined.png")
    savefig(final_plot, fname)
    display(final_plot)
end

run_homogenization_async()