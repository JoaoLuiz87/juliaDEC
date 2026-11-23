# ============================================================================
#  hh_consistency.jl -- isotropic SCVT meshes: symmetric stars exact on linear fields, and their O(h²) consistency
#
#      julia scripts/archive/hh_consistency.jl        (writes hh_data/consistency.txt and fig_hh_conv.png)
#
#  Question: on SCVT meshes (offset δ_e ≠ 0) is there a full matrix H, primal 1-cochains → dual 1-cochains,
#    (1) symmetric, (2) exact on linear fields row by row, (3) with Hodge error ε = H α(u) - β(u) = O(h²) for a
#    periodic field?   Fields on the unit torus:
#        u1 = (sin 2πx cos 2πy,  cos 2πx sin 2πy)   (a gradient),
#        u2 = (sin 2πx cos 2πy, -cos 2πx sin 2πy)   (divergence-free).
#  Errors per edge |ε_e| / |ẽ| (the flux per unit dual length; |u| ≤ 1), max and r.m.s. over the edges.
#  Stars:
#    DEC                 diag(|ẽ|/|e|)
#    Galerkin-tri        R_T = P_T⁺ + Z_T W_Tᵀ on triangle cells, 1-ring stencils, moment-constrained (galerkin_hodge.tex)
#    Galerkin-dia        the same on diamond cells, 13-edge stencils (= the juliaDEC sandwich with a corrected reconstruction)
#    shares-exact        the guide's signed patch shares, ρ = 2 (symmetric_star_guide.pdf)
#    shares-relaxed      uniform shares (abstol = 1e-3): symmetric, NOT exact on linear fields (reference)
# ============================================================================
include(joinpath(@__DIR__, "gh_lab.jl"))
include(joinpath(@__DIR__, "..", "..", "SymStar.jl")); using .SymStar
include(joinpath(@__DIR__, "ss_mesh.jl"))
using CairoMakie
const OUT = joinpath(@__DIR__, "..", "..", "output", "hh_data"); mkpath(OUT)
lines = String[]; logp(s) = (println(s); flush(stdout); push!(lines, s))

u1(x) = [sin(2pi * x[1]) * cos(2pi * x[2]), cos(2pi * x[1]) * sin(2pi * x[2])]
u2(x) = [sin(2pi * x[1]) * cos(2pi * x[2]), -cos(2pi * x[1]) * sin(2pi * x[2])]

logp(@sprintf("%-6s %7s %-15s %9s %9s %9s %9s %9s %9s %7s %7s", "mesh", "h", "star", "asym", "row P1", "u1 max", "u1 rms",
              "u2 max", "u2 rms", "max C_e", "λmin"))
results = Dict{String,Vector{NTuple{5,Float64}}}()
for mn in ("mesh2", "mesh3", "mesh4", "mesh5")
    X, T, period, hS = scvt_mesh(mn)
    g = edge_geometry(X, T, period); M = PMesh(X, T, period...)
    @assert maximum(norm(g.me[e, :] - egeo(M, e, M.X[M.E[e][1]]).m) for e in 1:length(g.le)) < 1e-12   # same edges, same order
    geos = [egeo(M, e, M.X[M.E[e][1]]) for e in eachindex(M.E)]
    h = sqrt(1 / length(X)); E = length(g.le)
    a1, b1 = reductions(g, u1); a2, b2 = reductions(g, u2)
    stars = Any[("DEC", spdiagm(g.lde ./ g.le))]
    for (lbl, cells) in (("Galerkin-tri", tri_cells(M)), ("Galerkin-dia", dia_cells(M)))
        D = celldata(M, cells, 1); Rs, _ = constrained_R(M, cells, D, geos, 1)
        push!(stars, (lbl, assemble_H(M, cells, D, Rs)))
    end
    if 4 * 2.0 * hS < 1.0
        push!(stars, ("shares-exact", symmetric_star(g.me, g.t, g.le, g.lde, g.delta, period, hS; rho = 2.0, verbose = false)[1]))
        push!(stars, ("shares-relaxed", symmetric_star(g.me, g.t, g.le, g.lde, g.delta, period, hS; rho = 2.0, abstol = 1e-3, verbose = false)[1]))
    end
    for (lbl, H) in stars
        e1 = abs.(H * a1 - b1) ./ g.lde; e2 = abs.(H * a2 - b2) ./ g.lde
        rp1 = SymStar.check_affine(H, g.me, g.t, g.le, g.lde, g.delta, period, hS)
        C, _ = row_constants(H, g.me, g.t, g.le, g.lde, period, hS)
        λm = try smallest_eigenvalue(H) catch; NaN end
        asym = norm(H - H', 1) / norm(H, 1)
        r = (maximum(e1), sqrt(sum(abs2, e1) / E), maximum(e2), sqrt(sum(abs2, e2) / E), rp1)
        push!(get!(results, lbl, NTuple{5,Float64}[]), r)
        logp(@sprintf("%-6s %7.4f %-15s %9.1e %9.1e %9.2e %9.2e %9.2e %9.2e %7.2f %7.4f", mn, h, lbl, asym, rp1, r[1], r[2], r[3], r[4], maximum(C), λm))
    end
end
logp("\nobserved orders (h halves between levels), u1 max / u1 rms / u2 max / u2 rms:")
for (lbl, rs) in sort(collect(results); by = first)
    length(rs) < 2 && continue
    rate(k) = join([@sprintf("%.2f", log2(rs[i][k] / rs[i+1][k])) for i in 1:length(rs)-1], " ")
    logp(@sprintf("  %-15s  %s  |  %s  |  %s  |  %s", lbl, rate(1), rate(2), rate(3), rate(4)))
end
write(joinpath(OUT, "consistency.txt"), join(lines, "\n") * "\n")

# ---------------------------------------------------------------- figure
hs = [sqrt(1 / n) for n in (64, 256, 1024, 4096)]
fig = Figure(size = (1100, 420))
style = Dict("DEC" => (:black, :circle), "Galerkin-tri" => (:steelblue, :diamond), "Galerkin-dia" => (:firebrick, :rect),
             "shares-exact" => (:darkorange, :utriangle), "shares-relaxed" => (:gray, :dtriangle))
for (i, (k, ttl)) in enumerate(((1, "u₁ = ∇(-cos 2πx cos 2πy)/2π: max_e |ε_e|/|ẽ|"), (3, "u₂ divergence-free: max_e |ε_e|/|ẽ|")))
    ax = Axis(fig[1, i]; xscale = log10, yscale = log10, xlabel = "h", ylabel = "error", title = ttl, xreversed = true)
    for (lbl, rs) in results
        hh = hs[end-length(rs)+1:end]; c, mk = style[lbl]
        scatterlines!(ax, hh, getindex.(rs, k); color = c, marker = mk, label = lbl)
        scatterlines!(ax, hh, getindex.(rs, k + 1); color = c, marker = mk, linestyle = :dash)
    end
    lines!(ax, hs, 3 .* hs .^ 2; color = (:black, 0.4), linestyle = :dot); text!(ax, hs[2], 3hs[2]^2 * 1.6; text = "h²", color = :gray)
    lines!(ax, hs, 0.12 .* hs; color = (:black, 0.4), linestyle = :dashdot); text!(ax, hs[2], 0.12hs[2] * 1.4; text = "h", color = :gray)
    i == 1 && axislegend(ax; position = :rb, labelsize = 10)
end
Label(fig[2, 1:2], "solid: maximum over the edges; dashed: r.m.s.; juliaDEC SCVT meshes 2-5 (N = 64 ... 4096)"; fontsize = 12)
save(joinpath(@__DIR__, "..", "..", "output", "figures", "fig_hh_conv.png"), fig; px_per_unit = 2)
println("done")
