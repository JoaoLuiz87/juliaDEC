# ============================================================================
#  gs_figures.jl -- figures and worked-example numbers of galerkin_star_paper.tex (uses galerkin_star.jl)
#
#      julia scripts/archive/gs_figures.jl        (needs output/gs_data/check_table.txt from gs_check.jl)
#
#  fig_gs_stencil.png  (a) a triangle cell and its 1-ring stencil, (b, c) the reconstruction functions of one edge
#                      (least squares and moment-constrained) over the cells whose stencils contain it
#  fig_gs_kernel.png   spectrum of the multiplier matrix G near zero (mesh2) and of Hd^{-1/2} H Hd^{-1/2} (mesh3)
#  fig_gs_row.png      |H_ef| against distance for every row (mesh4), Galerkin versus lift
#  fig_gs_conv.png     Hodge error per edge for u1, u2 against h (max and rms), four stars
#  fig_gs_poisson.png  Poisson errors against h, four stars
#  gs_data/worked.txt  the numbers of the worked example
# ============================================================================
using VoronoiMeshes, TensorsLite, TensorsLiteGeometry, SmallCollections, LinearAlgebra, NCDatasets
using StaticArrays, SparseArrays, Printf
Base.similar(bc::Base.Broadcast.Broadcasted, ::Type{Union{}}) = Array{Union{}}(undef, length.(axes(bc)))
const ROOT = joinpath(@__DIR__, "..", ".."); cd(ROOT)
include(joinpath(ROOT, "src", "juliaDEC.jl")); include(joinpath(ROOT, "src", "hodge_stars.jl")); include(joinpath(ROOT, "src", "galerkin_star.jl"))
const OUT = joinpath(@__DIR__, "..", "..", "output", "gs_data"); mkpath(OUT)
W = String[]; logw(s) = (println(s); flush(stdout); push!(W, s))
unw(y, x, P) = [y[1] - P[1] * round((y[1] - x[1]) / P[1]), y[2] - P[2] * round((y[2] - x[2]) / P[2])]

# ---------------------------------------------------------------- worked example (mesh3)
mesh = VoronoiMesh("meshes/mesh3.nc"); P = (mesh.x_period, mesh.y_period)
H, info = galerkin_star_matrix(mesh; keep = true)
cells, edata, Rs = info.data.cells, info.data.edges, info.data.R
s = info.s
t0 = argmin([norm(C.xc - [0.5, 0.5]) for C in cells]); C = cells[t0]; R = Rs[t0]
logw(@sprintf("Worked example: mesh3 (%d triangles, %d edges), basis scale s = %.5f, cell T = %d, centroid (%.4f, %.4f)",
              mesh.vertices.n, mesh.edges.n, s, t0, C.xc...))
logw(@sprintf("  stencil n_T = %d edges (3 own + %d); P_T is %d×6, singular values %s", length(C.S), length(C.S) - 3, length(C.S),
              join([@sprintf("%.3f", x) for x in svdvals(C.P)], ", ")))
logw("  first rows of P_T (own edges): " * join([join([@sprintf("%+.4f", x) for x in C.P[q, :]], " ") for q in 1:3], "  |  "))
logw(@sprintf("  M_T eigenvalues %s (|T| = %.5f)", join([@sprintf("%.2e", x) for x in eigvals(Symmetric(C.M))], ", "), C.M[1, 1]))
logw(@sprintf("  ‖P⁺P - I‖ = %.1e, ‖R P - I‖ = %.1e;  dim ker Pᵀ = %d;  ‖R - P⁺‖_M / ‖P⁺‖_M = %.3f", opnorm(C.R0 * C.P - I),
              opnorm(R * C.P - I), size(C.W, 2), sqrt(tr((R - C.R0)' * C.M * (R - C.R0)) / tr(C.R0' * C.M * C.R0))))
e0 = C.S[1]; ed = edata[e0]
# moments of the reconstruction function Φ_e over all cells containing e, least squares and constrained
function moments(e, which)
    acc = zeros(6)
    for (k, Ck) in enumerate(cells)
        q = findfirst(==(e), Ck.S); q === nothing && continue
        Rk = which == :lsq ? Ck.R0 : Rs[k]
        acc .+= Ck.Nm[q]' * Rk[:, q]
    end
    acc
end
mL = moments(e0, :lsq); mC = moments(e0, :con)
ncont = count(Ck -> e0 in Ck.S, cells)
logw(@sprintf("  edge e = %d (first own edge): |e| = %.5f, |ẽ| = %.5f, contained in %d stencils", e0, ed.le, ed.lde, ncont))
logw("    β_e(ψ_i), i = 1..6           : " * join([@sprintf("%+.6f", x) for x in ed.β], " "))
logw("    Σ_T ∫ φ̃_T,e·ψ_i, least squares: " * join([@sprintf("%+.6f", x) for x in mL], " "))
logw("    Σ_T ∫ φ̃_T,e·ψ_i, constrained  : " * join([@sprintf("%+.6f", x) for x in mC], " "))
B = R' * C.M * R; Pr = I - C.P * R; Sm = Pr' * Diagonal(1 ./ C.lens .^ 2) * Pr; σ = tr(B) / tr(Sm); Bt = B + σ * Sm
evB = eigvals(Symmetric(Bt))
logw(@sprintf("  local block B_T: σ_T = %.3e, eigenvalues in [%.2e, %.2e] (positive definite), ‖B_T P_T - Rᵀ M_T‖ = %.1e", σ, evB[1], evB[end],
              opnorm(Bt * C.P - R' * C.M)))
logw(@sprintf("  global: G is %d×%d with %d non-zeros; obstruction %.1e; moment residual %.1e (least squares %.2f)", size(info.data.G)...,
              nnz(info.data.G), info.obstruction, info.moment_residual, info.moment_residual_lsq))
logw(@sprintf("  H: %d×%d, %.1f non-zeros per row; H_ee = %.4f (DEC: %.4f); row exactness of row e: %.1e; C_e = %.3f",
              size(H)..., nnz(H) / size(H, 1), H[e0, e0], ed.lde / ed.le, gs_row_exactness(mesh, H; edges = [e0]),
              gs_row_constants(mesh, H)[e0]))
a = [1.0, 0.7]
αc = reduce_primal_edges(mesh, x -> a).values; βc = reduce_dual_edges(mesh, star_field(x -> a)).values
logw(@sprintf("  constant field a = (1, 0.7): max_e |(H α(a) - β(a))_e| / |ẽ| = %.1e", maximum(abs.(H * αc - βc) ./ edge_lengths(mesh, false))))
write(joinpath(OUT, "worked.txt"), join(W, "\n") * "\n")

using CairoMakie
# ---------------------------------------------------------------- figure: stencil and reconstruction functions
c0 = C.xc
segP(e, ref) = (p = edge_endpoints(mesh, e, true; about = _gs_vec(ref)); (p[1], p[2]))
fig = Figure(size = (1450, 480))
ax = Axis(fig[1, 1]; aspect = DataAspect(), title = "(a) cell T and its stencil S_T ($(length(C.S)) edges)"); hidedecorations!(ax); hidespines!(ax)
for e in 1:mesh.edges.n
    p1, p2 = segP(e, c0); norm((p1 + p2) / 2 - c0) < 0.2 && lines!(ax, [p1[1], p2[1]], [p1[2], p2[2]]; color = (:gray, 0.35), linewidth = 0.8)
end
tri = [unw(p, c0, P) for p in C.tri]
poly!(ax, Point2f[Tuple(p) for p in tri]; color = (:steelblue, 0.45))
for e in C.S
    p1, p2 = segP(e, c0); lines!(ax, [p1[1], p2[1]], [p1[2], p2[2]]; color = :black, linewidth = 2.2)
end
r0 = 0.17; xlims!(ax, c0[1] - r0, c0[1] + r0); ylims!(ax, c0[2] - r0, c0[2] + r0)
# reconstruction functions of edge e0
m0 = ed.m
for (col, which, ttl) in ((2, :lsq, "(b) least squares R_T = P_T⁺"), (3, :con, "(c) constrained R_T = P_T⁺ + Z_T W_Tᵀ"))
    ax = Axis(fig[1, col]; aspect = DataAspect(), title = ttl); hidedecorations!(ax); hidespines!(ax)
    pts = Point2f[]; vecs = Vec2f[]; gsum = zeros(2)
    for (k, Ck) in enumerate(cells)
        q = findfirst(==(e0), Ck.S); q === nothing && continue
        Rk = which == :lsq ? Ck.R0 : Rs[k]
        sh = unw(Ck.xc, m0, P) - Ck.xc
        poly!(ax, Point2f[Tuple(p + sh) for p in Ck.tri]; color = (:steelblue, 0.12), strokecolor = (:gray, 0.6), strokewidth = 0.6)
        for l in ((1 / 3, 1 / 3, 1 / 3), (0.6, 0.2, 0.2), (0.2, 0.6, 0.2), (0.2, 0.2, 0.6))
            x = l[1] * Ck.tri[1] + l[2] * Ck.tri[2] + l[3] * Ck.tri[3]
            push!(pts, Point2f(Tuple(x + sh))); push!(vecs, Vec2f(Tuple(basis_at(x, Ck.xc, true, s) * Rk[:, q])))
        end
        gsum .+= (Ck.Nm[q]' * Rk[:, q])[1:2]           # constant moments: ψ_1 = (1,0), ψ_2 = (0,1)
    end
    sc = 0.3 * s / maximum(norm, vecs)
    arrows2d!(ax, pts, vecs .* sc; color = :black, shaftwidth = 1.1, tipwidth = 6, tiplength = 6)
    lines!(ax, [ed.p1[1], ed.p2[1]], [ed.p1[2], ed.p2[2]]; color = :black, linewidth = 3)
    lines!(ax, [ed.v1[1], ed.v2[1]], [ed.v1[2], ed.v2[2]]; color = :firebrick, linewidth = 3)
    text!(ax, m0[1] - 0.13, m0[2] - 0.15; fontsize = 12,
          text = @sprintf("∫Φₑ = (%.5f, %.5f)\nβₑ(e₁), βₑ(e₂) = (%.5f, %.5f)", gsum..., ed.β[1], ed.β[2]))
    xlims!(ax, m0[1] - 0.15, m0[1] + 0.15); ylims!(ax, m0[2] - 0.17, m0[2] + 0.15)
end
CairoMakie.save(joinpath(@__DIR__, "..", "..", "output", "figures", "fig_gs_stencil.png"), fig; px_per_unit = 2)

# ---------------------------------------------------------------- figure: kernel of G (mesh2), spectrum of H (mesh3)
mesh2 = VoronoiMesh("meshes/mesh2.nc")
H2, info2 = galerkin_star_matrix(mesh2; keep = true, verbose = false)
evG = eigvals(Symmetric(Matrix(info2.data.G)))
Kv = info2.data.Kv
kres = maximum(norm(info2.data.G * Kv[:, k]) / (opnorm(Matrix(info2.data.G)) * norm(Kv[:, k])) for k in 1:size(Kv, 2))
logw(@sprintf("mesh2: G is %d×%d; eigenvalues/λmax: %s ... ; ‖Gλ^κ‖ rel. max over the 12 kernel vectors %.1e",
              size(info2.data.G)..., join([@sprintf("%.1e", x / evG[end]) for x in evG[1:14]], ", "), kres))
Hd3 = Diagonal(edge_lengths(mesh, false) ./ edge_lengths(mesh, true))
Sd = Diagonal(1 ./ sqrt.(diag(Hd3)))
evGal = eigvals(Symmetric(Matrix(Sd * H * Sd)))
evSand = eigvals(Symmetric(Matrix(Sd * sparse(hodge_star_sandwich_matrix(mesh; only_linear = true)) * Sd)))
fig = Figure(size = (1150, 420))
ax = Axis(fig[1, 1]; yscale = log10, xlabel = "k", ylabel = "λ_k(G) / λ_max", title = "multiplier matrix G (mesh2): 12 zero eigenvalues")
scatterlines!(ax, 1:24, max.(abs.(evG[1:24]) ./ evG[end], 1e-18); color = :black)
vlines!(ax, [12.5]; color = :firebrick, linestyle = :dash); text!(ax, 13, 1e-12; text = "dim ker G = 12", color = :firebrick)
ax = Axis(fig[1, 2]; xlabel = "eigenvalue of Hd^{-1/2} H Hd^{-1/2}", ylabel = "count", title = "spectrum relative to DEC (mesh3)")
hist!(ax, evGal; bins = 60, color = (:steelblue, 0.6), label = "Galerkin star")
hist!(ax, evSand; bins = 60, color = (:darkorange, 0.45), label = "sandwich (hodge_stars.jl)")
axislegend(ax; position = :rt)
CairoMakie.save(joinpath(@__DIR__, "..", "..", "output", "figures", "fig_gs_kernel.png"), fig; px_per_unit = 2)
logw(@sprintf("mesh3: eigenvalues of Hd^{-1/2} H Hd^{-1/2}: Galerkin [%.3f, %.3f], sandwich [%.3f, %.3f]", evGal[1], evGal[end], evSand[1], evSand[end]))

# ---------------------------------------------------------------- figure: row structure (mesh4)
mesh4 = VoronoiMesh("meshes/mesh4.nc"); P4 = (mesh4.x_period, mesh4.y_period)
H4, _ = galerkin_star_matrix(mesh4; verbose = false)
HL4 = sparse(hodge_star_lift_matrix(mesh4; only_linear = true))
h4 = sqrt(P4[1] * P4[2] / mesh4.cells.n)
pos = [_pt(mesh4.edges.position[e]) for e in 1:mesh4.edges.n]
function rowscatter(Hm)
    Ht = sparse(transpose(Hm)); d = Float64[]; v = Float64[]
    for e in 1:size(Ht, 2), idx in nzrange(Ht, e)
        f = rowvals(Ht)[idx]; push!(d, norm(unw(pos[f], pos[e], P4) - pos[e]) / h4); push!(v, abs(nonzeros(Ht)[idx]))
    end
    d, v
end
fig = Figure(size = (1100, 420))
for (i, (Hm, ttl)) in enumerate(((H4, "Galerkin star (symmetric)"), (HL4, "lift (hodge_stars.jl, not symmetric)")))
    d, v = rowscatter(Hm)
    ax = Axis(fig[1, i]; yscale = log10, xlabel = "|m_f - m_e| / h", ylabel = "|H_ef|", title = ttl * @sprintf(", %.0f non-zeros/row", nnz(Hm) / size(Hm, 1)))
    scatter!(ax, d, max.(v, 1e-12); color = (:steelblue, 0.08), markersize = 3)
    ylims!(ax, 1e-6, 2)
end
CairoMakie.save(joinpath(@__DIR__, "..", "..", "output", "figures", "fig_gs_row.png"), fig; px_per_unit = 2)

# ---------------------------------------------------------------- convergence figures from the study table
T = [split(l) for l in readlines(joinpath(OUT, "check_table.txt"))[2:end] if !isempty(strip(l))]
col(r, k) = parse(Float64, r[k])       # 2 h, 4 asym, 5 rowP1, 6 u1max, 7 u1rms, 8 u2max, 9 u2rms, 10 C, 11 λmin, 12 μ, 13 pnod, 14 grad, 15 flux
style = (("DEC", :black, :circle), ("sandwich", :darkorange, :utriangle), ("lift", :seagreen, :dtriangle), ("Galerkin", :steelblue, :diamond))
function slopes!(ax, y2, y1)
    hs = [0.03125, 0.015625]
    lines!(ax, hs, [y2, y2 / 4]; color = :gray, linewidth = 2); text!(ax, 0.022, y2 / 2.6; text = "h²", color = :gray, align = (:right, :top))
    lines!(ax, hs, [y1, y1 / 2]; color = :gray, linewidth = 2, linestyle = :dot); text!(ax, 0.022, y1 / 1.3; text = "h", color = :gray, align = (:right, :bottom))
end
fig = Figure(size = (1150, 500))
for (i, (k, ttl)) in enumerate(((6, "u₁ = (sin 2πx cos 2πy, cos 2πx sin 2πy)"), (8, "u₂ = (sin 2πx cos 2πy, -cos 2πx sin 2πy)")))
    ax = Axis(fig[1, i]; xscale = log10, yscale = log10, xreversed = true, xlabel = "h", ylabel = i == 1 ? "|(Hα(u) - β(u))_e| / |ẽ_e|" : "", title = ttl)
    for (lbl, c, mk) in style
        rs = [r for r in T if r[3] == lbl]
        scatterlines!(ax, col.(rs, 2), col.(rs, k); color = c, marker = mk, label = lbl)
        scatterlines!(ax, col.(rs, 2), col.(rs, k + 1); color = c, marker = mk, linestyle = :dash)
    end
    slopes!(ax, 4e-3, 3e-2)
end
Legend(fig[2, 1:2], fig.content[1]; orientation = :horizontal, framevisible = false)
Label(fig[3, 1:2], "solid: maximum over the edges; dashed: r.m.s.; juliaDEC SCVT meshes 2-5"; fontsize = 12)
CairoMakie.save(joinpath(@__DIR__, "..", "..", "output", "figures", "fig_gs_conv.png"), fig; px_per_unit = 2)
fig = Figure(size = (1400, 460))
for (i, (k, ttl)) in enumerate(((13, "nodal error (max, relative)"), (14, "gradient error ‖d₀p_h - α(∇p)‖ (relative)"), (15, "flux error max_e |(Hd₀p_h - β(∇p))_e|/|ẽ| (rel.)")))
    ax = Axis(fig[1, i]; xscale = log10, yscale = log10, xreversed = true, xlabel = "h", title = ttl)
    for (lbl, c, mk) in style
        rs = [r for r in T if r[3] == lbl]
        scatterlines!(ax, col.(rs, 2), col.(rs, k); color = c, marker = mk, label = lbl)
    end
end
Legend(fig[2, 1:3], fig.content[1]; orientation = :horizontal, framevisible = false)
CairoMakie.save(joinpath(@__DIR__, "..", "..", "output", "figures", "fig_gs_poisson.png"), fig; px_per_unit = 2)
write(joinpath(OUT, "worked.txt"), join(W, "\n") * "\n")
println("figures written")
