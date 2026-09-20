using Plots
using LinearAlgebra
using FastGaussQuadrature

# Define kernels
J_gauss(z, sigma=1.0) = (1 / (sigma * sqrt(pi))) * exp(-(z / sigma)^2)
J_laplace(z, b=1.0) = (1 / (2 * b)) * exp(-abs(z) / b)

# Define test functions
v_smooth(x) = cos(pi * x / 10.0)
v_rough(x) = abs(100.0 - x^2)^1.5 * (abs(x) <= 10.0 ? 1.0 : 0.0)

function build_quadrature_matrices(N, L, rule="trap")
    h = 2L / N
    x = collect(range(-L, L, length=N+1))
    
    W = zeros(N+1)
    if rule == "trap"
        W .= h
        W[1] = h/2; W[end] = h/2
    elseif rule == "simp"
        @assert iseven(N) "N must be even for Simpson's rule"
        W .= h/3
        W[1] = h/3; W[end] = h/3
        W[2:2:end-1] .= 4h/3
        W[3:2:end-2] .= 2h/3
    end
    
    T_gauss = [J_gauss(x[i] - x[j]) for i in 1:N+1, j in 1:N+1]
    T_laplace = [J_laplace(x[i] - x[j]) for i in 1:N+1, j in 1:N+1]
    
    K_gauss = T_gauss * Diagonal(W)
    K_laplace = T_laplace * Diagonal(W)
    
    return x, K_gauss, K_laplace
end

function build_spectral_matrices(N, L)
    nodes, weights = gausslegendre(N+1)
    x = L .* nodes # Map to [-L, L]
    w = L .* weights
    
    T_gauss =[J_gauss(x[i] - x[j]) for i in 1:N+1, j in 1:N+1]
    T_laplace = [J_laplace(x[i] - x[j]) for i in 1:N+1, j in 1:N+1]
    
    # In spectral collocation with Gauss-Legendre, the quadrature weights act as the integration weights
    K_gauss = T_gauss * Diagonal(w)
    K_laplace = T_laplace * Diagonal(w)
    
    return x, K_gauss, K_laplace
end

# Exact evaluation (using a very dense trapezoidal rule as ground truth)
x_exact, K_g_exact, K_l_exact = build_quadrature_matrices(5000, 10.0, "trap")
v_s_exact = v_smooth.(x_exact)
v_r_exact = v_rough.(x_exact)

Kv_g_s_exact = K_g_exact * v_s_exact
Kv_g_r_exact = K_g_exact * v_r_exact
Kv_l_s_exact = K_l_exact * v_s_exact
Kv_l_r_exact = K_l_exact * v_r_exact

# Experiment Loop
Ns = 10:20:500
err_trap_g_s, err_simp_g_s, err_spec_g_s = Float64[], Float64[], Float64[]
err_trap_l_s, err_simp_l_s, err_spec_l_s = Float64[], Float64[], Float64[]
err_spec_g_r = Float64[]

for N in Ns
    # Quad
    x_q, Kg_q, Kl_q = build_quadrature_matrices(N, 10.0, "trap")
    _, Kg_s, Kl_s = build_quadrature_matrices(N, 10.0, "simp")
    v_q_s = v_smooth.(x_q)
    
    # Spectral
    x_sp, Kg_sp, Kl_sp = build_spectral_matrices(N, 10.0)
    v_sp_s = v_smooth.(x_sp)
    v_sp_r = v_rough.(x_sp)
    
    # Interpolate exact to node points to compute error
    exact_g_s_q = [Kv_g_s_exact[argmin(abs.(x_exact .- xi))] for xi in x_q]
    exact_l_s_q = [Kv_l_s_exact[argmin(abs.(x_exact .- xi))] for xi in x_q]
    
    exact_g_s_sp = [Kv_g_s_exact[argmin(abs.(x_exact .- xi))] for xi in x_sp]
    exact_l_s_sp =[Kv_l_s_exact[argmin(abs.(x_exact .- xi))] for xi in x_sp]
    exact_g_r_sp = [Kv_g_r_exact[argmin(abs.(x_exact .- xi))] for xi in x_sp]
    
    push!(err_trap_g_s, norm(Kg_q * v_q_s - exact_g_s_q, Inf))
    push!(err_simp_g_s, norm(Kg_s * v_q_s - exact_g_s_q, Inf))
    push!(err_spec_g_s, norm(Kg_sp * v_sp_s - exact_g_s_sp, Inf))
    
    push!(err_trap_l_s, norm(Kl_q * v_q_s - exact_l_s_q, Inf))
    push!(err_simp_l_s, norm(Kl_s * v_q_s - exact_l_s_q, Inf))
    push!(err_spec_l_s, norm(Kl_sp * v_sp_s - exact_l_s_sp, Inf))
    
    push!(err_spec_g_r, norm(Kg_sp * v_sp_r - exact_g_r_sp, Inf))
end

p1 = plot(Ns, err_trap_g_s, yaxis=:log, label="Trap (Smooth)", lw=2)
plot!(p1, Ns, err_simp_g_s, label="Simp (Smooth)", lw=2)
plot!(p1, Ns, err_spec_g_s, label="Spec (Smooth)", lw=2)
plot!(p1, Ns, err_spec_g_r, label="Spec (Rough v)", linestyle=:dash, lw=2)
title!(p1, "Gaussian Kernel")
xlabel!(p1, "N nodes")
ylabel!(p1, "Max Error")

p2 = plot(Ns, err_trap_l_s, yaxis=:log, label="Trap (Smooth)", lw=2)
plot!(p2, Ns, err_simp_l_s, label="Simp (Smooth)", lw=2)
plot!(p2, Ns, err_spec_l_s, label="Spec (Smooth)", lw=2)
title!(p2, "Laplace Kernel")
xlabel!(p2, "N nodes")

plot(p1, p2, layout=(1,2), size=(900, 400), margin=5Plots.mm)
savefig("convergence_operator_K.png")
