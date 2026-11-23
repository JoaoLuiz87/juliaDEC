# ============================================================================
#  ss_figures.jl -- figures and worked-example numbers of symmetric_star_review.tex
#
#      julia scripts/archive/ss_figures.jl     (writes fig_ss_*.png and ss_data/worked.txt)
#
#  fig_ss_patch.png  : one patch (ρ = 2) coloured by its shares; singular values of S_P; the shares of one edge
#                      over all the patches that contain it
#  fig_ss_row.png    : one row of H on the mesh, and |H_ef| against distance
#  fig_ss_kernel.png : spectrum of G near zero (lattices n = 10, 20, 40), λ₅(G) against h, shares and row
#                      constants against h for the two objectives (from ss_data/objective.txt)
#  fig_ss_conv.png   : flux consistency error against h, lattice family and SCVT family (ss_data/objective.txt)
# ============================================================================
include(joinpath(@__DIR__, "..", "..", "SymStar.jl")); using .SymStar
include(joinpath(@__DIR__, "ss_mesh.jl"))
using LinearAlgebra, SparseArrays, Printf, CairoMakie
const OUT = joinpath(@__DIR__, "..", "..", "output", "ss_data")
W = String[]; logw(s) = (println(s); push!(W, s))

X, T, period, h = porting_mesh(20); g = edge_geometry(X, T, period); E = length(g.le)
H, info = symmetric_star(g.me, g.t, g.le, g.lde, g.delta, period, h; rho = 2.0, keep = true, verbose = false)
pats, θs, Vs, svs = info.data.patches, info.data.thetas, info.data.V, info.data.svals
c = argmin([hypot(g.me[e, 1] - 0.5, g.me[e, 2] - 0.43) for e in 1:E])          # a centre edge in the middle
P, N, R = pats[c]; θ = θs[c]

# ---------------------------------------------------------------- worked example: one patch
logw(@sprintf("Worked example on the porting lattice (N = 400, E = %d, h = %.5f, ρ = 2), patch of edge c = %d", E, h, c))
logw(@sprintf("  n_P = %d edges; N_P, R_P are %d×6", length(P), length(P)))
i1 = findfirst(==(c), P)
logw("  row of the centre edge:  N = " * join([@sprintf("%.5f", x) for x in N[i1, :]], ", "))
logw("                           R = " * join([@sprintf("%.5f", x) for x in R[i1, :]], ", "))
logw(@sprintf("  δ_c/|e_c| = %.4f, |ẽ_c|/|e_c| = %.4f", g.delta[c] / g.le[c], g.lde[c] / g.le[c]))
s = svs[c]
logw("  singular values of S_P (relative): " * join([@sprintf("%.1e", x / s[1]) for x in s], ", "))
logw(@sprintf("  rank S_P = %d, admissible shares: dim null S_P = %d", size(Vs[c], 2), length(P) - size(Vs[c], 2)))
logw(@sprintf("  shares of this patch: min %.4f, max %.4f, sum %.4f; |S_P θ_P| = %.1e", minimum(θ), maximum(θ), sum(θ), norm(skew_map(N, R) * θ)))
Rt = θ .* R; A = N' * Rt
HP = patch_star(N, R, θ, g.le[P], g.lde[P])
evHP = eigvals(Symmetric(HP))
logw(@sprintf("  A_P = N_Pᵀ diag(θ) R_P: |A - Aᵀ|/|A| = %.1e;  H_P: |H_P - H_Pᵀ| = %.1e, |H_P N_P - diag(θ)R_P| = %.1e",
              norm(A - A') / norm(A), norm(HP - HP'), norm(HP * N - Rt)))
logw(@sprintf("  eigenvalues of H_P: min %.3f, max %.3f (%d negative: patch matrices need not be positive)", evHP[1], evHP[end], count(<(0), evHP)))
# shares of edge c over all patches containing it
memb = [(k, findfirst(==(c), pats[k][1])) for k in 1:E if c in pats[k][1]]
θc = [θs[k][j] for (k, j) in memb]
dist = [hypot(SymStar.wrapd(g.me[k, 1] - g.me[c, 1], period[1]), SymStar.wrapd(g.me[k, 2] - g.me[c, 2], period[2])) / h for (k, _) in memb]
logw(@sprintf("  edge c lies in %d patches; its shares: min %.4f, max %.4f, sum %.12f", length(memb), minimum(θc), maximum(θc), sum(θc)))
# a row of H
rng = nzrange(H, c); rs = rowvals(H)[rng]; hv = nonzeros(H)[rng]
logw(@sprintf("  row c of H: %d non-zeros; H_cc = %.4f (diagonal star: %.4f); Σ_f |H_cf| |f| / |ẽ_c| = C_c = %.3f",
              length(rs), H[c, c], g.lde[c] / g.le[c], sum(abs.(hv) .* g.le[rs]) / g.lde[c]))
for (lbl, q) in (("q = (1, 0)", x -> [1.0, 0.0]), ("q = (0, 1)", x -> [0.0, 1.0]), ("q = (y - y_c, 0)", x -> [x[2] - g.me[c, 2], 0.0]),
                 ("q = (0, x - x_c)", x -> [0.0, x[1] - g.me[c, 1]]))
    # reductions on the copies next to m_c
    a = [begin
             d = [SymStar.wrapd(g.me[f, 1] - g.me[c, 1], period[1]), SymStar.wrapd(g.me[f, 2] - g.me[c, 2], period[2])]
             m = g.me[c, :] + d; tf = g.t[f, :]
             SymStar.seg6(x -> dot(q(x), tf), m - g.le[f] / 2 * tf, m + g.le[f] / 2 * tf)
         end for f in rs]
    b = SymStar.seg6(x -> dot(q(x), g.t[c, :]), g.c1[c, :], g.c2[c, :])
    logw(@sprintf("    %-18s (Hα(q))_c = %+.10f,  β_c(q) = %+.10f", lbl, dot(hv, a), b))
end
write(joinpath(OUT, "worked.txt"), join(W, "\n") * "\n")

# ---------------------------------------------------------------- figure: patch, singular values, shares of one edge
seg(e, ref) = (d = [SymStar.wrapd(g.me[e, 1] - ref[1], period[1]), SymStar.wrapd(g.me[e, 2] - ref[2], period[2])];
               m = ref + d; (m - g.le[e] / 2 * g.t[e, :], m + g.le[e] / 2 * g.t[e, :]))
ref = g.me[c, :]
fig = Figure(size = (1350, 430))
ax = Axis(fig[1, 1]; aspect = DataAspect(), title = "patch of edge c (ρ = 2): shares θ_P,e"); hidedecorations!(ax); hidespines!(ax)
for e in 1:E
    a, b = seg(e, ref); norm((a + b) / 2 - ref) < 3.2h && lines!(ax, [a[1], b[1]], [a[2], b[2]]; color = (:gray, 0.35), linewidth = 0.8)
end
lim = maximum(abs, θ); cm = cgrad(:balance)
for (k, e) in enumerate(P)
    a, b = seg(e, ref)
    lines!(ax, [a[1], b[1]], [a[2], b[2]]; color = θ[k], colorrange = (-lim, lim), colormap = cm, linewidth = 4)
end
a, b = seg(c, ref); lines!(ax, [a[1], b[1]], [a[2], b[2]]; color = :black, linewidth = 2, linestyle = :dash)
arc = [ref + 2h * [cos(t), sin(t)] for t in range(0, 2pi, 100)]
lines!(ax, first.(arc), last.(arc); color = :black, linestyle = :dot)
Colorbar(fig[1, 2]; colorrange = (-lim, lim), colormap = cm, label = "θ_P,e", height = Relative(0.8))
ax2 = Axis(fig[1, 3]; yscale = log10, xlabel = "k", ylabel = "σ_k(S_P) / σ_1", title = "singular values of the skew map S_P")
for k in 1:E
    sk = svs[k] ./ svs[k][1]; sk = max.(sk, 1e-17)
    lines!(ax2, 1:length(sk), sk; color = (:steelblue, 0.03))
end
scatterlines!(ax2, 1:length(s), max.(s ./ s[1], 1e-17); color = :black)
vlines!(ax2, [7.5]; color = :firebrick, linestyle = :dash); text!(ax2, 7.8, 1e-3; text = "rank 7", color = :firebrick)
ax3 = Axis(fig[1, 4]; xlabel = "|m_P - m_c| / h", ylabel = "θ_P,c", title = "shares of edge c in its $(length(memb)) patches")
scatter!(ax3, dist, θc; color = :steelblue); hlines!(ax3, [1 / length(memb)]; color = :gray, linestyle = :dash)
text!(ax3, 0.05, maximum(θc); text = @sprintf("Σ = %.6f\ndashed: 1/%d", sum(θc), length(memb)), align = (:left, :top))
save(joinpath(@__DIR__, "..", "..", "output", "figures", "fig_ss_patch.png"), fig; px_per_unit = 2)

# ---------------------------------------------------------------- figure: one row of H
fig = Figure(size = (1000, 420))
ax = Axis(fig[1, 1]; aspect = DataAspect(), title = "row c of H: |H_cf| (log)"); hidedecorations!(ax); hidespines!(ax)
for e in 1:E
    a, b = seg(e, ref); norm((a + b) / 2 - ref) < 4.5h && lines!(ax, [a[1], b[1]], [a[2], b[2]]; color = (:gray, 0.3), linewidth = 0.8)
end
lv = log10.(abs.(hv) .+ 1e-16); cr = (maximum(lv) - 4, maximum(lv))
for (k, f) in enumerate(rs)
    a, b = seg(f, ref); lines!(ax, [a[1], b[1]], [a[2], b[2]]; color = lv[k], colorrange = cr, colormap = :viridis, linewidth = 3.5)
end
Colorbar(fig[1, 2]; colorrange = cr, colormap = :viridis, label = "log₁₀ |H_cf|", height = Relative(0.8))
ax2 = Axis(fig[1, 3]; yscale = log10, xlabel = "|m_f - m_c| / h", ylabel = "|H_cf|", title = "entries of row c")
dr = [norm((seg(f, ref)[1] + seg(f, ref)[2]) / 2 - ref) / h for f in rs]
scatter!(ax2, dr, abs.(hv) .+ 1e-16; color = :steelblue, markersize = 7)
scatter!(ax2, [0.0], [g.lde[c] / g.le[c]]; color = :firebrick, marker = :star5, markersize = 16, label = "diagonal star")
axislegend(ax2; position = :rt)
save(joinpath(@__DIR__, "..", "..", "output", "figures", "fig_ss_row.png"), fig; px_per_unit = 2)

# ---------------------------------------------------------------- figure: kernel and growth
obj = [split(l) for l in readlines(joinpath(OUT, "objective.txt"))[2:end] if !isempty(strip(l))]
fig = Figure(size = (1350, 420))
ax1 = Axis(fig[1, 1]; yscale = log10, xlabel = "k", ylabel = "λ_k(G) / λ_max", title = "spectrum of G near zero (lattice)")
λ5 = Float64[]; hs = Float64[]
for (n, col) in ((10, :goldenrod), (20, :steelblue), (40, :firebrick))
    Xn, Tn, pn, hn = porting_mesh(n); gn = edge_geometry(Xn, Tn, pn)
    pt = build_patches(gn.me, gn.t, gn.le, gn.lde, gn.delta, pn, hn, 2.0)
    Vn = [SymStar.rowspace(skew_map(Nk, Rk))[1] for (Pk, Nk, Rk) in pt]
    ev = eigvals(Symmetric(Matrix(share_matrix(pt, Vn, length(gn.le)))))
    scatterlines!(ax1, 1:12, max.(abs.(ev[1:12]) ./ ev[end], 1e-17); color = col, label = "n = $n")
    push!(λ5, ev[5] / ev[end]); push!(hs, hn)
end
vlines!(ax1, [4.5]; color = :gray, linestyle = :dash); axislegend(ax1; position = :rb)
ax2 = Axis(fig[1, 2]; xscale = log10, yscale = log10, xlabel = "h", ylabel = "λ₅(G) / λ_max", title = "first non-zero eigenvalue")
scatterlines!(ax2, hs, λ5; color = :black)
lines!(ax2, hs, λ5[end] .* (hs ./ hs[end]) .^ 4; color = :gray, linestyle = :dash); text!(ax2, hs[2], λ5[2] * 3; text = "h⁴", color = :gray)
ax3 = Axis(fig[1, 3]; xscale = log10, yscale = log10, xlabel = "E (edges)", ylabel = "max|θ|  (solid),  max C_e  (dashed)",
           title = "shares and row constants under refinement")
abst = [split(l) for l in readlines(joinpath(OUT, "abstol.txt"))[2:end] if !isempty(strip(l))]
for (fam, mk) in (("lattice", :circle), ("scvt", :rect)), (at, col, nm) in (("0e+00", :firebrick, "exact (guide)"), ("1e-03", :steelblue, "uniform shares"))
    rows = [r for r in abst if r[1] == fam && r[4] == at]
    isempty(rows) && continue
    Ev = parse.(Float64, getindex.(rows, 3))
    scatterlines!(ax3, Ev, parse.(Float64, getindex.(rows, 6)); color = col, marker = mk, label = "$fam, $nm")
    scatterlines!(ax3, Ev, parse.(Float64, getindex.(rows, 7)); color = col, marker = mk, linestyle = :dash)
end
axislegend(ax3; position = :lt, labelsize = 10)
save(joinpath(@__DIR__, "..", "..", "output", "figures", "fig_ss_kernel.png"), fig; px_per_unit = 2)
include(joinpath(@__DIR__, "ss_figures_conv.jl"))
println("figures written")
