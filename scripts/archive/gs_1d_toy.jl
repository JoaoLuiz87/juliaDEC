# ============================================================================
#  gs_1d_toy.jl -- the reconstruction-energy Hodge star in ONE dimension, where every matrix fits on a page
#
#      julia scripts/archive/gs_1d_toy.jl          (writes output/gs_data/toy1d.txt and output/figures/fig_toy1d.png)
#
#  Periodic interval [0, 1).  Primal edges e_k = [x_k, x_{k+1}], length h_k, midpoint m_k.  Each edge has a dual point
#  x̃_k = m_k + δ_k inside it (the 1D twin of the Voronoi edge; δ_k is the offset).
#      primal 1-cochain  α_k = ∫_{e_k} u dx          dual 0-cochain  β_k = u(x̃_k)     (⋆ of the 1-form u dx is u)
#  DEC: H_d = diag(1/h_k), i.e. (H_d α)_k = mean of u over e_k = u(m_k) for linear u: error -δ_k u'.
#
#  The method:  cell j = edge e_j, stencil S_j = {j-1, j, j+1}, basis {1, (x - m_j)/s} on the cell.
#    P_j (3 x 2) circulations of the basis on the stencil, M_j (2 x 2) mass matrix,
#    R_j = P_j⁺ + z_j w_jᵀ  (w_j spans the null space of P_jᵀ, z_j ∈ R²: TWO numbers per cell),
#    H = Σ_j S_jᵀ [R_jᵀ M_j R_j + σ_j (I - P_jR_j)ᵀ D_j (I - P_jR_j)] S_j.
#    Φ_k = the function the reconstructions build from the cochain 1_k (three linear pieces, on cells k-1, k, k+1).
#    Exact on linear u  <=>  ∫Φ_k = 1 and ∫Φ_k (x - m_k) = δ_k  (mass 1, centre at the dual point).
#    The z_j: minimal Σ z_jᵀ M_j z_j subject to these 2N conditions (dense solve; N is small).
# ============================================================================
using LinearAlgebra, Printf, Random, CairoMakie
lines = String[]
logp(s) = (println(s); push!(lines, s))

"""
mesh.  `kind = :alternating` (default): uniform cells, offsets δ_k = (-1)^k θ(x_k) h with the smoothly varying amplitude
θ(x) = 0.15 (1 + 0.5 sin 2πx); Σ h_k δ_k = 0 exactly for even N.  `kind = :random`: non-uniform cells (smooth map) and
random offsets δ_k ≈ ±0.15 h_k, corrected only globally so that Σ h_k δ_k = 0.  `uniform = true, θ`: uniform, δ_k = θ_k h.
"""
function toy_mesh(N; uniform = false, θ = nothing, kind = :alternating, seed = 1)
    rng = MersenneTwister(seed)
    if uniform
        x = collect((0:N-1) ./ N); h = fill(1 / N, N); δ = θ .* h
    elseif kind == :alternating
        x = collect((0:N-1) ./ N); h = fill(1 / N, N)
        δ = [(-1)^k * 0.15 * (1 + 0.5sin(2pi * x[k])) for k in 1:N] .* h
    else
        ξ = (0:N-1) ./ N
        x = collect(ξ .+ 0.06 .* sin.(2pi .* ξ)); h = [mod(x[mod1(k + 1, N)] - x[k], 1.0) for k in 1:N]
        δ = [0.15 * (-1)^k + 0.05 * (2rand(rng) - 1) for k in 1:N] .* h
        δ .-= (sum(h .* δ) / sum(h .^ 2)) .* h             # enforce Σ h_k δ_k = 0 globally only
    end
    m = x .+ h ./ 2
    (x = x, h = h, m = m, δ = δ, N = N)
end

"position of the midpoint of edge q seen from edge j (periodic lift)"
lift(M, q, j) = M.m[q] + round(M.m[j] - M.m[q])

"one cell: stencil, P, M, P⁺, null vector w"
function toy_cell(M, j, s)
    N = M.N; S = [mod1(j - 1, N), j, mod1(j + 1, N)]
    P = [M.h[q] * (lift(M, q, j) - M.m[j]) / s for q in S]
    P = hcat(M.h[S], P)                                    # ∫_{e_q} 1,  ∫_{e_q} (x - m_j)/s
    Mm = [M.h[j] 0.0; 0.0 M.h[j]^3 / (12s^2)]
    w = nullspace(P')[:, 1]
    (S = S, P = P, M = Mm, P⁺ = pinv(P), w = w)
end

"mixed mass N_{j,k} (2 x 2): ∫_{e_j} φ_a ψ_b, φ = cell basis of j, ψ = test basis centred at m_k"
function toy_N(M, j, k, s)
    c = lift(M, j, k) - M.m[k]                              # m_j - m_k in the frame of k
    [M.h[j] M.h[j]*c/s; 0.0 M.h[j]^3/(12s^2)]
end

"build the star; corrected = false gives the least-squares energy star (z = 0)"
function toy_star(M; corrected = true, τ = 1.0)
    N = M.N; s = sum(M.h) / N
    cells = [toy_cell(M, j, s) for j in 1:N]
    # constraints A z = r on z = (z_1, ..., z_N), z_j ∈ R²
    A = zeros(2N, 2N); r = zeros(2N)
    for k in 1:N
        r[2k-1:2k] .= [1.0, M.δ[k] / s]                         # ψ(x̃_k) for ψ = 1, (x - m_k)/s
    end
    for (j, C) in enumerate(cells), (q, k) in enumerate(C.S)
        Nm = toy_N(M, j, k, s)
        r[2k-1:2k] .-= Nm' * C.P⁺[:, q]                         # mismatch of least squares
        A[2k-1:2k, 2j-1:2j] .+= C.w[q] .* Nm'                   # effect of z_j on edge k
    end
    Mbig = zeros(2N, 2N)
    for (j, C) in enumerate(cells)
        Mbig[2j-1:2j, 2j-1:2j] .= C.M
    end
    z = corrected ? (Mbig \ A') * (pinv(A * (Mbig \ A')) * r) : zeros(2N)
    H = zeros(N, N); Rs = Matrix{Float64}[]
    for (j, C) in enumerate(cells)
        R = C.P⁺ + z[2j-1:2j] * C.w'; push!(Rs, R)
        B = R' * C.M * R
        Pr = I - C.P * R; St = Pr' * Diagonal(1 ./ M.h[C.S] .^ 2) * Pr
        tr(St) > 1e-14 && (B .+= (τ * tr(B) / tr(St)) .* St)
        H[C.S, C.S] .+= (B + B') / 2
    end
    (H = H, R = Rs, cells = cells, z = z, s = s, residual = norm(A * z - r) / norm(r))
end

"the reconstruction function Φ_k: list of (cell j, interval, coefficients a + b (x - m_j)/s) in the frame of k"
function toy_Phi(M, star, k)
    pieces = []
    for (j, C) in enumerate(star.cells)
        q = findfirst(==(k), C.S); q === nothing && continue
        c = star.R[j][:, q]; mj = lift(M, j, k)
        push!(pieces, (j = j, a = mj - M.h[j] / 2, b = mj + M.h[j] / 2, c0 = c[1], c1 = c[2] / star.s, mj = mj))
    end
    pieces
end
Φ_moments(M, P, k) = (sum(p.c0 * (p.b - p.a) for p in P),                                   # ∫Φ_k
                      sum(p.c0 * (p.b - p.a) * (p.mj - M.m[k]) + p.c1 * (p.b - p.a)^3 / 12 for p in P))   # ∫Φ_k (x - m_k)

# ---------------------------------------------------------------- 1. the uniform mesh, by hand
logp("1. Uniform mesh, N = 8, h = 1/8, alternating offsets δ_k = ±0.15 h")
U = toy_mesh(8; uniform = true, θ = [0.15 * (-1)^k for k in 1:8])
C = toy_cell(U, 4, sum(U.h) / 8)
logp("   P_4 (rows: edges 3, 4, 5; columns: ∫1, ∫(x-m_4)/h) = " * string(round.(C.P ./ U.h[1]; digits = 6)) * " × h")
logp("   P_4⁺ × h = " * string(round.(C.P⁺ .* U.h[1]; digits = 6)) * ",   w = " * string(round.(C.w .* sqrt(6); digits = 6)) * " / √6")
LS = toy_star(U; corrected = false); CO = toy_star(U)
for (lbl, st) in (("least squares", LS), ("corrected", CO))
    P = toy_Phi(U, st, 4); mass, cen = Φ_moments(U, P, 4)
    logp(@sprintf("   %-14s h·Φ_4 on cells 3, 4, 5 = %s;  mass %.6f, centre - m_4 = %+.6f h  (dual point: %+.6f h)", lbl,
                  join([@sprintf("%.4f %+.4f ξ", p.c0 * U.h[1], p.c1 * U.h[1]^2) for p in sort(P; by = p -> p.mj)], " | "),
                  mass, cen / U.h[1], U.δ[4] / U.h[1]))
end
logp("   predicted correction (δ_k/h)(3ξ, 6ξ, 3ξ) with δ_4/h = " * @sprintf("%.2f: %.4f, %.4f, %.4f", U.δ[4] / U.h[1], 3U.δ[4] / U.h[1], 6U.δ[4] / U.h[1], 3U.δ[4] / U.h[1]))
for (lbl, H) in (("DEC", Diagonal(1 ./ U.h)), ("least squares", LS.H), ("corrected", CO.H))
    logp(@sprintf("   row 4 of h·H, %-14s: %s", lbl, join([@sprintf("%+.4f", v) for v in (Matrix(H)[4, :] .* U.h[1])], " ")))
end

# ---------------------------------------------------------------- 2. checks and convergence on non-uniform meshes
u(x) = sin(2pi * x) + 0.5cos(4pi * x)
U_int(a, b) = (-cos(2pi * b) + cos(2pi * a)) / (2pi) + 0.5(sin(4pi * b) - sin(4pi * a)) / (4pi)
function errors(M, H)
    α = [U_int(M.x[k], M.x[k] + M.h[k]) for k in 1:M.N]; β = u.(M.m .+ M.δ)
    maximum(abs.(H * α - β))
end
function linear_exactness(M, H)
    worst = 0.0
    for k in 1:M.N, (c0, c1) in ((1.0, 0.0), (0.0, 1.0))
        ψint(q) = (lq = lift(M, q, k); c0 * M.h[q] + c1 * M.h[q] * (lq - M.m[k]))      # ∫_{e_q} ψ, ψ = c0 + c1 (x - m_k)
        worst = max(worst, abs(sum(H[k, q] * ψint(q) for q in 1:M.N if H[k, q] != 0) - (c0 + c1 * M.δ[k])))
    end
    worst
end
"largest correction: max_j ‖z_j w_jᵀ‖_M / ‖P_j⁺‖_M"
corr_size(st) = maximum(sqrt(tr((st.R[j] - C.P⁺)' * C.M * (st.R[j] - C.P⁺)) / tr(C.P⁺' * C.M * C.P⁺)) for (j, C) in enumerate(st.cells))
res = Dict{Symbol,Any}()
for kind in (:alternating, :random)
    logp(kind == :alternating ?
         "\n2. Uniform cells, alternating offsets δ_k = (-1)^k θ(x_k) h, θ = 0.15 (1 + 0.5 sin 2πx); u = sin 2πx + 0.5 cos 4πx" :
         "\n2b. Contrast: non-uniform cells, random offsets that cancel only globally (Σ h_k δ_k = 0)")
    logp(@sprintf("%5s %10s %10s %10s %10s %12s %10s %12s", "N", "max δ/h", "DEC", "least sq.", "corrected", "lin. exact", "corr.", "λmin(h·H)"))
    r = []
    for N in (16, 32, 64, 128, 256)
        M = toy_mesh(N; kind = kind); Hd = Diagonal(1 ./ M.h); L = toy_star(M; corrected = false); Cc = toy_star(M)
        e = (errors(M, Hd), errors(M, L.H), errors(M, Cc.H))
        push!(r, e)
        logp(@sprintf("%5d %10.3f %10.3e %10.3e %10.3e %12.1e %10.2f %12.4f", N, maximum(abs.(M.δ ./ M.h)), e..., linear_exactness(M, Cc.H),
                      corr_size(Cc), minimum(eigvals(Symmetric(Cc.H .* (sum(M.h) / N))))))
    end
    logp("   orders:  DEC " * join([@sprintf("%.2f", log2(r[i][1] / r[i+1][1])) for i in 1:4], " ") *
         "   least squares " * join([@sprintf("%.2f", log2(r[i][2] / r[i+1][2])) for i in 1:4], " ") *
         "   corrected " * join([@sprintf("%.2f", log2(r[i][3] / r[i+1][3])) for i in 1:4], " "))
    res[kind] = r
end

# ---------------------------------------------------------------- 3. what if Σ h_k δ_k ≠ 0 ?
M = toy_mesh(32); Mbad = (x = M.x, h = M.h, m = M.m, δ = M.δ .+ 0.05 .* M.h, N = M.N)
logp(@sprintf("\n3. Offsets with Σ h_k δ_k = %.2e ≠ 0: relative residual of the 2N conditions after the best correction = %.2e (with Σ h_k δ_k = 0: %.1e)",
              sum(Mbad.h .* Mbad.δ), toy_star(Mbad).residual, toy_star(M).residual))

# ---------------------------------------------------------------- 4. degree 2 is impossible: the size of the obstruction
logp(@sprintf("\n4. Degree 2: Σ_k ∫_{e_k} (x̃_k - y)² dy = Σ_k h_k (h_k²/12 + δ_k²) = %.3e > 0 on the N = 32 mesh (must be 0 for exactness on quadratics)",
              sum(M.h .* (M.h .^ 2 ./ 12 .+ M.δ .^ 2))))
write(joinpath(@__DIR__, "..", "..", "output", "gs_data", "toy1d.txt"), join(lines, "\n") * "\n")

# ---------------------------------------------------------------- figure
fig = Figure(size = (1500, 470))
ax = Axis(fig[1, 1]; title = "reconstruction function Φ₄ (uniform mesh, δ₄ = +0.15h)", xlabel = "x / h  (edge 4 = [3, 4])", ylabel = "h · Φ₄")
for k in 0:8
    vlines!(ax, [k]; color = (:gray, 0.4), linewidth = 0.8)
end
for (lbl, st, c) in (("least squares", LS, :steelblue), ("corrected", CO, :firebrick))
    for p in toy_Phi(U, st, 4)
        xs = range(p.a, p.b; length = 2)
        lines!(ax, xs ./ U.h[1], (p.c0 .+ p.c1 .* (xs .- p.mj)) .* U.h[1]; color = c, linewidth = 3, label = lbl)
    end
end
mk = U.m[4] / U.h[1]; dk = (U.m[4] + U.δ[4]) / U.h[1]
scatter!(ax, [mk], [0.0]; color = :black, marker = :circle, markersize = 12)
scatter!(ax, [dk], [0.0]; color = :firebrick, marker = :utriangle, markersize = 16)
text!(ax, mk - 0.05, 0.02; text = "m₄", align = (:right, :bottom)); text!(ax, dk + 0.05, 0.02; text = "x̃₄", color = :firebrick, align = (:left, :bottom))
xlims!(ax, 1.6, 6.4); ylims!(ax, -0.25, 0.85)
axislegend(ax; position = :rt, unique = true)
ax2 = Axis(fig[1, 2]; xscale = log10, yscale = log10, xreversed = true, xlabel = "h = 1/N", ylabel = "max_k |(Hα(u))_k - u(x̃_k)|",
           title = "convergence (alternating offsets)")
hs = 1 ./ [16, 32, 64, 128, 256]
for (i, lbl, c, m) in ((1, "DEC (diagonal)", :black, :circle), (2, "energy star, least squares", :steelblue, :diamond),
                       (3, "energy star, corrected", :firebrick, :rect))
    scatterlines!(ax2, hs, [r[i] for r in res[:alternating]]; color = c, marker = m, label = lbl)
end
lines!(ax2, [hs[2], hs[5]], [3e-2, 3e-2 / 8]; color = :gray, linestyle = :dot); text!(ax2, hs[4], 6e-3; text = "h", color = :gray)
lines!(ax2, [hs[2], hs[5]], [2e-3, 2e-3 / 64]; color = :gray); text!(ax2, hs[4], 6e-5; text = "h²", color = :gray)
axislegend(ax2; position = :lb)
CairoMakie.save(joinpath(@__DIR__, "..", "..", "output", "figures", "fig_toy1d.png"), fig; px_per_unit = 2)
