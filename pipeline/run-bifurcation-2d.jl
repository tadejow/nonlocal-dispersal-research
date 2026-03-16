#%%
# ==============================================================================
# CELL 1: Import packages and define operators and the model (2D Version)
# ==============================================================================
using BifurcationKit, LinearAlgebra, Plots, SparseArrays, Setfield, Statistics, ProgressMeter
const BK = BifurcationKit

function Laplacian2D_Dirichlet(N, L)
    hx = 2 * L / (N - 1)
    # 1D Laplacian
    D2_1D = spdiagm(0 => -2 * ones(N), 1 => ones(N - 1), -1 => ones(N - 1)) / hx^2
    I_1D = sparse(I, N, N)
    
    # 2D Laplacian using Kronecker sum
    D2_2D = kron(I_1D, D2_1D) + kron(D2_1D, I_1D)
    return D2_2D
end

function F_system!(f, x, p)
    Nxy = p.Nxy
    u, v = (@view x[1:Nxy]), (@view x[Nxy+1:2Nxy])
    fu, fv = (@view f[1:Nxy]), (@view f[Nxy+1:2Nxy])
    
    nl_term = u .* u .* v
    mul!(fu, p.Op_u, u); fu .= p.d_u .* fu .- p.B .* u .+ nl_term
    mul!(fv, p.D2, v);   fv .= p.d_v .* fv .- v .+ p.A .- nl_term
    
    # Apply Dirichlet BCs (u = v = 0 on boundaries) in 2D
    fu[p.bnd_indices] .= 0.0
    fv[p.bnd_indices] .= 0.0
    return f
end

function J_system(x, p)
    Nxy = p.Nxy
    u, v = (@view x[1:Nxy]), (@view x[Nxy+1:2Nxy])
    
    J_uu = p.d_u .* p.Op_u - p.B .* I + spdiagm(0 => 2 .* u .* v)
    J_uv = spdiagm(0 => u .* u)
    J_vu = spdiagm(0 => -2 .* u .* v)
    J_vv = p.d_v .* p.D2 - I - spdiagm(0 => u .* u)
    
    # Apply Dirichlet BCs efficiently using a precomputed masking matrix
    M = spdiagm(0 => p.interior_mask)
    J_uu = M * J_uu; J_uv = M * J_uv
    J_vu = M * J_vu; J_vv = M * J_vv
    
    Bnd_diag = spdiagm(0 => 1.0 .- p.interior_mask)
    J_uu += Bnd_diag
    J_vv += Bnd_diag
    
    J = [J_uu J_uv; J_vu J_vv]
    return J
end

println("Cell 1 executed: Packages, 2D operators, and model defined.")

#%%
# ==============================================================================
# CELL 2: Definition of 2D calibrated dispersal kernels
# ==============================================================================

# Numerically/Analytically calculated constants to ensure each 2D kernel 
# has a variance of 1.0 over R^2.
const C_SUB_2D = 3.0 / pi
const B_SUB_2D = sqrt(6.0)

const C_SG_2D = 2.0 / (pi^2)
const B_SG_2D = 1.0 / pi

"""
Kernel 1: Sub-Gaussian in 2D (Fat Tails)
Variance = 1
"""
function kernel_sub_gaussian_2D(r)
    return C_SUB_2D * exp(-B_SUB_2D * r)
end

"""
Kernel 2: Super-Gaussian in 2D (Thin Tails)
Variance = 1
"""
function kernel_super_gaussian_2D(r)
    return C_SG_2D * exp(-B_SG_2D * r^4)
end

function create_integral_matrix_2D(domain, kernel_func::Function)
    N = length(domain)
    hx = domain[2] - domain[1]
    
    # 2D trapezoidal weights (tensor product of 1D weights)
    w1D = ones(N); w1D[1] = 0.5; w1D[end] = 0.5; w1D .*= hx
    W_2D = kron(w1D, w1D)
    
    # Create array of 2D coordinates
    pts = [[x, y] for x in domain, y in domain]
    pts = vec(pts)
    
    # Dense matrix based on Euclidean distance
    K_matrix =[kernel_func(norm(p1 - p2)) for p1 in pts, p2 in pts]
    return sparse(K_matrix * spdiagm(0 => W_2D))
end

println("Cell 2 executed: New 2D kernels defined.")

#%%
# ==============================================================================
# CELL 3: Main computation loop for the 2 non-local kernels (2D)
# ==============================================================================

# Simulation parameters
B = 0.45; d_u, d_v = 0.1, 100.0
A_min, A_max = 0.05, 1.8
L = 30

# W 2D grid ma rozmiar N^2. N=41 daje 1681 punktów (wystarczające do wzorów).
N = 25
Nxy = N^2
domain = LinRange(-L, L, N)

# Precompute boundary indices and mask for the 2D grid
bnd_indices = Int[]
for i in 1:N, j in 1:N
    if i == 1 || i == N || j == 1 || j == N
        push!(bnd_indices, (j-1)*N + i)
    end
end
interior_mask = ones(Nxy)
interior_mask[bnd_indices] .= 0.0

kernels_to_test = Dict(
    "Sub-Gaussian (Fat Tails)" => kernel_sub_gaussian_2D,
    "Super-Gaussian (Thin Tails)" => kernel_super_gaussian_2D
)

all_results = Dict()

for (kernel_name, kernel_func) in kernels_to_test
    println("\n" * "="^60)
    println("--- Starting computations for kernel: $kernel_name ---")
    
    K_integral = create_integral_matrix_2D(domain, kernel_func)
    Op_nonlocal = K_integral - I
    D2_laplace = Laplacian2D_Dirichlet(N, L)
    
    lens = @optic _.A
    params_nonlocal = (N=N, Nxy=Nxy, L=L, B=B, d_u=d_u, d_v=d_v, A=A_max, 
                       Op_u=Op_nonlocal, D2=D2_laplace, 
                       bnd_indices=bnd_indices, interior_mask=interior_mask)
                       
    params_local = (N=N, Nxy=Nxy, L=L, B=B, d_u=d_u / 2, d_v=d_v, A=A_max, 
                    Op_u=D2_laplace, D2=D2_laplace, 
                    bnd_indices=bnd_indices, interior_mask=interior_mask)
    
    opt_newton = NewtonPar(tol = 1e-6, max_iterations = 10000, verbose = false)
    opts_br = ContinuationPar(p_min = A_min, p_max = A_max, ds = -0.01, dsmax = 0.05,
        nev = 10, detect_bifurcation = 3, n_inversion = 10, max_steps=2000, newton_options = opt_newton)
    
    A_start = A_max
    u_hom = (A_start + sqrt(A_start^2 - 4*B^2)) / (2*B)
    v_hom = B / u_hom
    
    cos_profile =[cos((π/2) * x / L) * cos((π/2) * y / L) for x in domain, y in domain]
    cos_profile = vec(cos_profile)
    
    u0 = u_hom .* cos_profile .+ 0.01 * (rand(Nxy) .- 0.5) .* cos_profile
    v0 = v_hom .* cos_profile
    x0_guess = vcat(u0, v0)
    
    # -------- OBLICZENIA DLA MODELU NIELOKALNEGO --------
    println("Computing bifurcation diagram for the NON-LOCAL model...")
    prog_nl = Progress(1000, dt=0.5, desc="  Postęp: ", barlen=40, color=:green)
    
    # POPRAWKA TUTAJ: Drugi argument to bieżące A (Float64), używamy globalnego Nxy
    record_sol_nl(x, A_current; kwargs...) = begin
        current_val = clamp(Int(round(1000 * (A_max - A_current) / (A_max - A_min))), 0, 1000)
        ProgressMeter.update!(prog_nl, current_val)
        return (u_max=maximum(view(x, 1:Nxy)), u_avg=mean(view(x, 1:Nxy)))
    end

    prob_nl = BifurcationProblem(F_system!, x0_guess, params_nonlocal, lens; J=J_system, record_from_solution=record_sol_nl)
    br_nl = continuation(prob_nl, PALC(), opts_br; verbosity = 0)
    ProgressMeter.finish!(prog_nl)
    
    # -------- OBLICZENIA DLA MODELU LOKALNEGO --------
    println("Computing bifurcation diagram for the LOCAL model...")
    prog_l = Progress(1000, dt=0.5, desc="  Postęp: ", barlen=40, color=:blue)
    
    # POPRAWKA TUTAJ: Analogicznie
    record_sol_l(x, A_current; kwargs...) = begin
        current_val = clamp(Int(round(1000 * (A_max - A_current) / (A_max - A_min))), 0, 1000)
        ProgressMeter.update!(prog_l, current_val)
        return (u_max=maximum(view(x, 1:Nxy)), u_avg=mean(view(x, 1:Nxy)))
    end
    
    prob_l = BifurcationProblem(F_system!, x0_guess, params_local, lens; J=J_system, record_from_solution=record_sol_l)
    br_l = continuation(prob_l, PALC(), opts_br; verbosity = 0)
    ProgressMeter.finish!(prog_l)
    
    all_results[kernel_name] = Dict("nonlocal" => br_nl, "local" => br_l)
end
println("\n" * "="^60)
println("--- All computations finished. Ready for plotting. ---")

#%%
# ==============================================================================
# CELL 4: Plotting BIFURCATION DIAGRAMS (1x2 layout)
# ==============================================================================

println("\n--- Plotting bifurcation diagrams... ---")

# Plotting parameters
plot_size_diagrams = (1500, 800)
f_title = 15
f_guide = 15
f_tick = 15
f_legend = 15
line_width = 2.0
main_title_size = 18

plot_array_diagrams = []
kernel_order =["Super-Gaussian (Thin Tails)", "Sub-Gaussian (Fat Tails)"]

for (idx, kernel_name) in enumerate(kernel_order)
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
# CELL 5: Plotting a 2x3 2D Profile Gallery (Heatmaps)
# ==============================================================================

println("\n--- Creating 2D profile galleries (2x3 layouts) using Heatmaps... ---")

function find_closest_solution(br, target_A, Nxy; biomass_target=:high)
    indices = findall(p -> abs(p - target_A) < 0.05, br.param)
    if isempty(indices)
        _, idx = findmin(abs.(br.param .- target_A))
        return br.sol[idx].x[1:Nxy]
    end

    candidate_avg_biomass = br.u_avg[indices]
    if biomass_target == :high
        _, correct_index = findmax(candidate_avg_biomass)
    elseif biomass_target == :low
        _, correct_index = findmin(candidate_avg_biomass)
    else
        _, closest_idx = findmin(abs.(br.param[indices] .- target_A))
        correct_index = closest_idx
    end
    
    return br.sol[indices[correct_index]].x[1:Nxy]
end

plot_size = (1400, 900)
num_cols = 3
f_title = 14; f_guide = 12; f_tick = 10; main_title_size = 18

A_targets =[
    (A=1.5,   biomass=:high), (A=1.2,   biomass=:high),
    (A=1.1,   biomass=:high), (A=1.025, biomass=:high),
    (A=0.774, biomass=:high), (A=0.651, biomass=:high)
]

# Definiujemy struktury do pętli nad modelami
models_to_plot = [
    ("Model Lokalny (Local)", all_results["Super-Gaussian (Thin Tails)"]["local"]),
    ("Model Nielokalny (Thin Tails)", all_results["Super-Gaussian (Thin Tails)"]["nonlocal"]),
    ("Model Nielokalny (Fat Tails)", all_results["Sub-Gaussian (Fat Tails)"]["nonlocal"])
]

# Generujemy osobną galerię 2x3 dla każdego z trzech modeli
for (model_name, br) in models_to_plot
    gallery_plots =[]
    
    for (idx, target) in enumerate(A_targets)
        target_A = round(target.A, digits=2)
        
        # Wyciągamy profil wektora u z odpowiedniej gałęzi
        u_profile = find_closest_solution(br, target_A, Nxy; biomass_target=target.biomass)
        
        # Przekształcamy z wektora 1D z powrotem w macierz NxN dla 2D heatampy
        u_2d = reshape(u_profile, N, N)'
        
        p = heatmap(domain, domain, u_2d,
            title = "A ≈ $target_A",
            titlefontsize = f_title,
            tickfontsize = f_tick,
            guidefontsize = f_guide,
            color = :viridis,               # Używamy standardowej palety dla zagadnień wzorotwórczych
            aspect_ratio = :equal,          # Kwadratowa dziedzina
            right_margin = 5Plots.mm,       # Zostawiamy miejsce na colorbar
            colorbar_title = "Biomasa u"
        )
        
        if idx % num_cols == 1; ylabel!(p, "y"); end
        if idx > num_cols; xlabel!(p, "x"); end
        
        push!(gallery_plots, p)
    end
    
    # Tworzenie pojedynczej galerii 2x3 dla danego wariantu
    final_gallery = plot(gallery_plots...,
        layout = (2, num_cols),
        size = plot_size,
        plot_title = "Galeria wzorów 2D: $model_name",
        plot_titlefontsize = main_title_size,
        left_margin = 5Plots.mm,
        bottom_margin = 5Plots.mm
    )
    
    display(final_gallery)
end

println("\n--- End of script. ---")