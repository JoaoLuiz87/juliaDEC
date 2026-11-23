# ============================================================================
#  gap_skew_more.jl -- Hypothesis S2 on more centroidal Voronoi meshes: is |s_h(u,v)| ≥ c h^2 with c bounded away from 0?
#
#      julia scripts/barrier/gap_skew_more.jl       (writes output/ts_data/skew_more.{jls,txt} and output/figures/fig_gap_skew_more.png)
#
#  Three families of centroidal Voronoi meshes:
#    ours    the SCVT meshes 2-7 of the paper (unit torus, 4x more cells per level)
#    lloyd   Lloyd-converged variants of ours: mesh4 refined to 4096 generators + Lloyd, mesh5 refined to 16384 + Lloyd,
#            mesh7 + 2048 more Lloyd steps (generators from ts_data/lloyd_*.jls, lloyd_convergence_check.jl)
#    felipe  an independent CVT family (meshes_felipe/mesh_cvt_L1-L6): doubly periodic 84000 x 87295.36, 3x more cells per level.
#            Scaled by 1/84000 to [0,1) x [0,a), a = 1.03923, and rebuilt as the periodic Delaunay mesh of its generators.
#  Fields: the pairs of Fig. gap_skew, written for a domain [0,1) x [0,a): p(x) = P(x1, x2/a), etc.
#    (∇p,∇q), (J∇p,J∇q), (∇p,J∇q), (J∇p,∇q), (u2,d).
#  Reported: c = s_h / (|Ω| h^2), h = sqrt(|Ω|/N), N = number of cells (= 1/(N h^2)... dimensionless on every domain).
# ============================================================================
include(joinpath(@__DIR__, "..", "stars", "three_stars.jl"))
using CairoMakie
empty!(lines)

Pp(y) = sin(TP * y[1]) * sin(TP * y[2]) + 0.5cos(2TP * y[1])
GPp(y) = [TP * cos(TP * y[1]) * sin(TP * y[2]) - TP * sin(2TP * y[1]), TP * sin(TP * y[1]) * cos(TP * y[2])]
GQq(y) = [-TP * sin(TP * y[1]) * cos(2TP * y[2]) + TP * cos(TP * (y[1] + y[2])),
          -2TP * cos(TP * y[1]) * sin(2TP * y[2]) + TP * cos(TP * (y[1] + y[2]))]
rotf(f) = x -> (g = f(x); [-g[2], g[1]])
"the test pairs on [0,1) x [0,a): fields of y = (x1, x2/a); gradients by the chain rule"
function pairs_for(a)
    sc(x) = [x[1], x[2] / a]
    gp = x -> (g = GPp(sc(x)); [g[1], g[2] / a]); gq = x -> (g = GQq(sc(x)); [g[1], g[2] / a])
    u2 = x -> U2(sc(x)); dd = x -> fields.d(sc(x))
    return [("∇p,∇q", gp, gq), ("J∇p,J∇q", rotf(gp), rotf(gq)), ("∇p,J∇q", gp, rotf(gq)), ("J∇p,∇q", rotf(gp), gq), ("u2,d", u2, dd)]
end
function skew_of(mesh, u, v)
    α(w) = reduce_primal_edges(mesh, w).values; β(w) = reduce_dual_edges(mesh, star_field(w)).values
    return dot(α(u), β(v)) - dot(α(v), β(u))
end

meshes = Any[]
for i in 2:7; push!(meshes, ("ours", "mesh$i", () -> VoronoiMesh("meshes/mesh$i.nc"), 1.0)); end
for (lab, f) in (("mesh4 → 4096, Lloyd", "lloyd_mesh4_4096.jls"), ("mesh5 → 16384, Lloyd", "lloyd_mesh5_8192.jls"), ("mesh7 + 2048 Lloyd", "lloyd_mesh7_continue_2048.jls"))
    push!(meshes, ("lloyd", lab, () -> (D = deserialize(joinpath(TSDIR, f)); delaunay_mesh(D.x, D.y)), 1.0))
end
const LXF, LYF = 84000.0, 87295.3607014714
for i in 1:6
    push!(meshes, ("felipe", "cvt L$i", () -> begin
        m = VoronoiMesh("meshes_felipe/mesh_cvt_L$i.nc")
        xs = [p.x for p in m.cells.position] ./ LXF; ys = [p.y for p in m.cells.position] ./ LXF
        delaunay_mesh(xs, ys; lx = 1.0, ly = LYF / LXF)
    end, LYF / LXF))
end

f_out = joinpath(TSDIR, "skew_more.jls")
R = isfile(f_out) ? deserialize(f_out) : Dict{Any,Any}()
logp("Hypothesis S2 on more centroidal Voronoi meshes: c = s_h/(|Ω| h^2), h = sqrt(|Ω|/N)")
for (fam, lab, build, a) in meshes
    if !haskey(R, (fam, lab))
        mesh = build(); N = mesh.cells.n; Ω = a; h = sqrt(Ω / N)
        vals = [(nm, skew_of(mesh, u, v) / (Ω * h^2)) for (nm, u, v) in pairs_for(a)]
        nv5 = count(==(5), mesh.cells.nEdges); nv7 = count(==(7), mesh.cells.nEdges)
        R[(fam, lab)] = (N = N, h = h, vals = vals, v5 = nv5, v7 = nv7)
        serialize(f_out, R)
    end
    r = R[(fam, lab)]
    logp(@sprintf("%-7s %-22s N %6d  h %.5f  5/7-cells %3d/%3d   ", fam, lab, r.N, r.h, r.v5, r.v7) *
         join([@sprintf("%s %+.4f  ", nm, c) for (nm, c) in r.vals]) * @sprintf("  min|c| %.4f", minimum(abs(c) for (_, c) in r.vals)))
    flush(stdout)
end
write(joinpath(TSDIR, "skew_more.txt"), join(unique(lines), "\n") * "\n")

# ---------------------------------------------------------------- figure: |c| against h, one panel per family
f = Figure(size = (1500, 520))
cols = [colorant"#2b6cb0", colorant"#c23b3a", colorant"#2f8f4e", colorant"#d08a1e", colorant"#111111"]
mks = [:circle, :rect, :utriangle, :diamond, :star5]
titles = Dict("ours" => "(a) the SCVT meshes 2–7 of this paper", "lloyd" => "(b) Lloyd-converged variants of meshes 4, 5, 7",
              "felipe" => "(c) an independent CVT family (non-square torus, 3× cells per level)")
for (k, fam) in enumerate(("ours", "lloyd", "felipe"))
    keysf = [(fm, lb) for (fm, lb, _, _) in meshes if fm == fam && haskey(R, (fm, lb))]
    hs = [R[kk].h for kk in keysf]
    hts = [1 / 8, 1 / 16, 1 / 32, 1 / 64, 1 / 128, 1 / 256]
    ax = Axis(f[1, k]; xscale = log10, yscale = log10, xreversed = true, xlabel = "h = sqrt(|Ω|/N)", title = titles[fam], titlesize = 14,
              xticks = (hts, ["1/8", "1/16", "1/32", "1/64", "1/128", "1/256"]),
              ylabel = k == 1 ? "|s_h(u,v)| / (|Ω| h²)" : "", yticks = (10.0 .^ (-3:1), ["10⁻³", "10⁻²", "10⁻¹", "1", "10"]))
    xlims!(ax, 0.16, 0.0032)
    for (ip, (nm, _, _)) in enumerate(pairs_for(1.0))
        ys = [abs(R[kk].vals[ip][2]) for kk in keysf]
        scatterlines!(ax, hs, ys; color = cols[ip], marker = mks[ip], markersize = 11, linewidth = 1.8, label = nm)
    end
    ylims!(ax, 1e-3, 30.0)
    k == 1 && axislegend(ax; position = :lb, framevisible = false, labelsize = 12)
end
CairoMakie.save(joinpath(@__DIR__, "..", "..", "output", "figures", "fig_gap_skew_more.png"), f; px_per_unit = 2)
