# ============================================================================
#  story_offset.jl -- the offset of the SCVT meshes (hodge_story.tex, Chapter 2)  ->  fig_st_offset.png
#
#      julia scripts/barrier/story_offset.jl
#
#  (a) one Delaunay edge e, its triangles T-(e), T+(e), their circumcentres c- = m_e + s- n_e, c+ = m_e + s+ n_e, the Voronoi edge
#      from c- to c+ (|e~| = s+ - s-) and the offset delta_e = (s- + s+)/2 of its midpoint (notation of Appendix A).
#  (b) mesh 5: every Delaunay edge coloured by |delta_e|/h; generators with 5 and 7 neighbours marked.
# ============================================================================
ENV["LOC_MAIN"] = "0"; include(joinpath(@__DIR__, "locality_check.jl"))
include(joinpath(@__DIR__, "..", "shared", "thesis_plot_helpers.jl"))
using Statistics

circumcentre(a, b, c) = begin
    d = 2 * (a[1] * (b[2] - c[2]) + b[1] * (c[2] - a[2]) + c[1] * (a[2] - b[2]))
    [(sum(abs2, a) * (b[2] - c[2]) + sum(abs2, b) * (c[2] - a[2]) + sum(abs2, c) * (a[2] - b[2])) / d,
     (sum(abs2, a) * (c[1] - b[1]) + sum(abs2, b) * (a[1] - c[1]) + sum(abs2, c) * (b[1] - a[1])) / d]
end
RED, BLUE = colorant"#c23b3a", colorant"#1c5cab"
f = Figure(size = (1500, 700), fontsize = 17)

# ---------------------------------------------------------------- (a) the geometry of one edge
let a = [0.0, 0.0], b = [1.0, 0.0], pm = [0.30, -0.62], pp = [0.72, 0.78]
    cm, cp = circumcentre(a, b, pm), circumcentre(a, b, pp)
    me = (a + b) / 2; mt = (cm + cp) / 2; sm, sp = cm[2], cp[2]
    ax = Axis(f[1, 1]; aspect = DataAspect(), title = "(a) the Voronoi edge ẽ is not centred on e", titlealign = :left, titlesize = 19)
    hidedecorations!(ax); hidespines!(ax)
    poly!(ax, [Point2f(a...), Point2f(b...), Point2f(pm...)]; color = (:gray, 0.07), strokecolor = :gray, strokewidth = 1.2)
    poly!(ax, [Point2f(a...), Point2f(b...), Point2f(pp...)]; color = (:gray, 0.07), strokecolor = :gray, strokewidth = 1.2)
    lines!(ax, [a[1], b[1]], [a[2], b[2]]; color = :black, linewidth = 3.5)
    lines!(ax, [cm[1], cp[1]], [cm[2], cp[2]]; color = RED, linewidth = 3.5)
    scatter!(ax, [me[1]], [me[2]]; color = :black, markersize = 12)
    scatter!(ax, [mt[1]], [mt[2]]; color = RED, markersize = 12)
    scatter!(ax, [cm[1], cp[1]], [cm[2], cp[2]]; color = RED, marker = :diamond, markersize = 14)
    text!(ax, me[1], me[2] - 0.02; text = "  m_e", align = (:left, :top), fontsize = 19)
    text!(ax, cp[1], cp[2]; text = "  c₊ = m_e + s₊ n_e", align = (:left, :center), fontsize = 18, color = RED)
    text!(ax, cm[1], cm[2]; text = "  c₋ = m_e + s₋ n_e  (s₋ < 0)", align = (:left, :center), fontsize = 18, color = RED)
    # the offset, drawn as a bracket to the right of the Voronoi edge
    xo = mt[1] + 0.06
    lines!(ax, [xo, xo], [me[2], mt[2]]; color = RED, linewidth = 2)
    lines!(ax, [xo - 0.02, xo + 0.02], [mt[2], mt[2]]; color = RED, linewidth = 2); lines!(ax, [xo - 0.02, xo + 0.02], [me[2], me[2]]; color = RED, linewidth = 2)
    lines!(ax, [xo + 0.02, 1.08], [mt[2], 0.16]; color = (RED, 0.6), linewidth = 1, linestyle = :dot)
    text!(ax, 1.09, 0.16; text = "offset δ_e = ½(s₋ + s₊):\nm̃_e = m_e + δ_e n_e", align = (:left, :center), fontsize = 17, color = RED)
    text!(ax, cp[1] - 0.04, (cp[2] + mt[2]) / 2; text = "ẽ   ", align = (:right, :center), fontsize = 21, color = RED)
    text!(ax, 0.16, -0.03; text = "e", fontsize = 22, align = (:center, :top))
    text!(ax, 0.42, -0.42; text = "T₋(e)", fontsize = 18); text!(ax, 0.55, 0.42; text = "T₊(e)", fontsize = 18)
    arrows2d!(ax, [Point2f(0.07, 0.0)], [Vec2f(0.22, 0.0)]; color = :black, shaftwidth = 2, tipwidth = 10, tiplength = 10)
    arrows2d!(ax, [Point2f(0.07, 0.0)], [Vec2f(0.0, 0.22)]; color = :black, shaftwidth = 2, tipwidth = 10, tiplength = 10)
    text!(ax, 0.30, 0.0; text = "t_e", align = (:left, :bottom), fontsize = 18); text!(ax, 0.07, 0.24; text = "n_e = J t_e", align = (:center, :bottom), fontsize = 18)
    text!(ax, -0.1, -0.88; text = "|ẽ| = s₊ − s₋.  If c₋ and c₊ are symmetric about e (lattices), δ_e = 0.", align = (:left, :center), fontsize = 16, color = INK2)
    xlims!(ax, -0.15, 1.95); ylims!(ax, -0.95, 0.92)
end

# ---------------------------------------------------------------- (b) the offsets of SCVT mesh 5
let mesh = VoronoiMesh("meshes/mesh5.nc")
    h = meshsize(mesh); n = mesh.edges.n
    vc = argmin([norm(_pt(p) .- [0.5, 0.5]) for p in mesh.cells.position]); xcv = mesh.cells.position[vc]; xc = _pt(xcv)
    geo = edge_geometry(mesh, xcv)
    δ = [abs(dot(E.mt .- E.m, E.n)) / h for E in geo]
    deg = zeros(Int, mesh.cells.n)
    for e in 1:n, c in mesh.edges.cells[e]; deg[c] += 1; end
    println(@sprintf("mesh5 offsets |δ|/h: median %.4f, 90%% %.4f, max %.4f; cells with 5 / 6 / 7 sides: %d / %d / %d", median(δ), quantile(δ, 0.9),
                     maximum(δ), count(==(5), deg), count(==(6), deg), count(==(7), deg)))
    ax = Axis(f[1, 2]; aspect = DataAspect(), title = @sprintf("(b) SCVT mesh 5: |δ_e|/h on every edge (median %.3f, max %.3f)", median(δ), maximum(δ)),
              titlealign = :left, titlesize = 19)
    hidedecorations!(ax)
    cmax = 0.06
    for k in sortperm(δ)
        E = geo[k]; a = E.m .- (E.le / 2) .* E.t; b = E.m .+ (E.le / 2) .* E.t
        lines!(ax, [a[1], b[1]], [a[2], b[2]]; color = min(δ[k], cmax), colormap = Reverse(:magma), colorrange = (0, cmax), linewidth = 1.3)
    end
    pos = [_pt(closest(xcv, p, 1.0, 1.0)) for p in mesh.cells.position]
    for (d, col, lab) in ((5, BLUE, "5 neighbours"), (7, colorant"#2f8f4e", "7 neighbours"))
        P = [Point2f(pos[c]...) for c in 1:mesh.cells.n if deg[c] == d]
        scatter!(ax, P; color = col, markersize = 7, label = lab)
    end
    xlims!(ax, xc[1] - 0.5, xc[1] + 0.5); ylims!(ax, xc[2] - 0.5, xc[2] + 0.5)
    axislegend(ax; position = :rb, framevisible = true, backgroundcolor = (:white, 0.85), labelsize = 14)
    Colorbar(f[1, 3]; colormap = Reverse(:magma), limits = (0, cmax), label = "|δ_e| / h", width = 13)
end
colsize!(f.layout, 1, Relative(0.47))
CairoMakie.save(thesis_out("fig_st_offset.png"), f; px_per_unit = 2)
println("offset figure written")
