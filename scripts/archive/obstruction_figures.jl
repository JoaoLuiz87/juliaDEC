# ============================================================================
#  obstruction_figures.jl  -- figures for hodge_symmetry_consistency.tex
#
#      julia scripts/archive/obstruction_figures.jl            (all figures)
#      julia obstruction_figures.jl quick                           (skip the slow ones)
# ============================================================================
using VoronoiMeshes, TensorsLite, TensorsLiteGeometry, SmallCollections, LinearAlgebra, NCDatasets
using StaticArrays, SparseArrays, QuadGK, Printf, CairoMakie
Base.similar(bc::Base.Broadcast.Broadcasted, ::Type{Union{}}) = Array{Union{}}(undef, length.(axes(bc)))
const ROOT = joinpath(@__DIR__, "..", "..")
cd(ROOT)
include(joinpath(ROOT, "src", "juliaDEC.jl")); include(joinpath(ROOT, "src", "hodge_stars.jl")); include(joinpath(ROOT, "symmetric_star.jl"))
const OUT = joinpath(ROOT, "report")
const QUICK = "quick" in ARGS

# ---------------------------------------------------------------- style
const INK = colorant"#2b2b29"; const MUTED = colorant"#8a8a85"; const GRID = colorant"#e6e5e1"
const BLUE = colorant"#2a78d6"; const ORANGE = colorant"#eb6834"; const AQUA = colorant"#1baf7a"
const LIGHTBLUE = colorant"#9ec5f4"; const LIGHTGRAY = colorant"#cfcec9"
const DIV = cgrad([colorant"#1c5cab", colorant"#f0efec", colorant"#c23b3a"])
const FAMCOL = Dict("regular" => BLUE, "SCVT" => ORANGE, "perturbed" => AQUA)
const FAMMRK = Dict("regular" => :circle, "SCVT" => :utriangle, "perturbed" => :rect)
const FAMS = [("regular",   ["mesh2_regular", "mesh3_regular", "mesh4_regular", "mesh5_regular"]),
              ("SCVT",      ["mesh2", "mesh3", "mesh4", "mesh5"]),
              ("perturbed", ["pert2", "pert3", "pert4", "pert5"])]
set_theme!(Theme(fontsize = 13, textcolor = INK,
    Axis = (xgridcolor = GRID, ygridcolor = GRID, spinewidth = 0.8, leftspinecolor = MUTED,
            bottomspinecolor = MUTED, topspinevisible = false, rightspinevisible = false,
            xtickcolor = MUTED, ytickcolor = MUTED, titlefont = :regular, titlesize = 13),
    Legend = (framevisible = false, labelsize = 12)))
savefig(name, f) = (CairoMakie.save(joinpath(OUT, name), f, px_per_unit = 2); println("  wrote ", name))
meanh(m) = sum(edge_lengths(m, true)) / m.edges.n
function arrow!(ax, p, d; color = INK, lw = 2, head = 11)
    q = p .+ d
    lines!(ax, [p[1], q[1]], [p[2], q[2]], color = color, linewidth = lw)
    scatter!(ax, [q[1]], [q[2]], marker = :utriangle, markersize = head, color = color,
             rotation = atan(d[2], d[1]) - pi/2)
end

# geometry of one edge, in the frame unwrapped about the edge
function edgegeom(m, e)
    c1, c2 = edge_endpoints(m, e, true); v1, v2 = edge_endpoints(m, e, false)
    t = (c2 - c1) / norm(c2 - c1); n = [-t[2], t[1]]
    me = (c1 + c2) / 2; mt = (v1 + v2) / 2
    return (c1 = c1, c2 = c2, v1 = v1, v2 = v2, t = t, n = n, me = me, mt = mt,
            delta = dot(mt - me, n), le = norm(c2 - c1), lde = norm(v2 - v1))
end
deltas(m) = [(g = edgegeom(m, e); g.delta / g.lde) for e in 1:m.edges.n]
defects(m) = ([_pt(m.cells.position[c]) for c in 1:m.cells.n if length(m.cells.edges[c]) == 5],
              [_pt(m.cells.position[c]) for c in 1:m.cells.n if length(m.cells.edges[c]) == 7])

# the two smooth periodic test fields used throughout
const U = x -> [sin(2pi*x[2]), cos(2pi*x[1])]
const V = x -> [cos(2pi*(x[1] + x[2])), sin(2pi*(x[1] - 2x[2]))]
avg_t(w, p1, p2, t) = sum(wq * dot(w(p1 + q*(p2 - p1)), t) for (q, wq) in _GL3)
# per-edge contribution c_e to s_h(u,v) = sum_e c_e
function skew_contrib(m, u, v)
    c = zeros(m.edges.n)
    for e in 1:m.edges.n
        g = edgegeom(m, e)
        bu, bv = avg_t(u, g.c1, g.c2, g.t), avg_t(v, g.c1, g.c2, g.t)
        tu, tv = avg_t(u, g.v1, g.v2, g.t), avg_t(v, g.v1, g.v2, g.t)
        c[e] = g.le * g.lde * (bv * tu - bu * tv)
    end
    return c
end
# periodic Gaussian smoothing of point weights c at (px,py), evaluated on grid xs × xs,
# truncated at 3σ (each point only touches nearby grid nodes)
function splat_density(c::Vector{Float64}, px::Vector{Float64}, py::Vector{Float64}, xs::Vector{Float64}, σ::Float64)
    ng = length(xs); dx = xs[2] - xs[1]; D = zeros(ng, ng); w = ceil(Int, 3σ / dx)
    for e in eachindex(c)
        i0 = round(Int, px[e] / dx); j0 = round(Int, py[e] / dx)
        for di in -w:w, dj in -w:w
            i = mod(i0 + di, ng - 1) + 1; j = mod(j0 + dj, ng - 1) + 1
            ddx = xs[i] - px[e]; ddx -= round(ddx); ddy = xs[j] - py[e]; ddy -= round(ddy)
            D[i, j] += c[e] * exp(-(ddx^2 + ddy^2) / (2σ^2))
        end
    end
    D[end, :] .= D[1, :]; D[:, end] .= D[:, 1]          # periodic copy of the first row/column
    return D ./ (2pi * σ^2)
end
function s_rel(m, u, v)
    Ru, Rv = reduce_primal_edges(m, u).values, reduce_primal_edges(m, v).values
    Su, Sv = reduce_dual_edges(m, star_field(u)).values, reduce_dual_edges(m, star_field(v)).values
    return abs(dot(Rv, Su) - dot(Ru, Sv)) / sqrt(abs(dot(Ru, Su) * dot(Rv, Sv)))
end

# ============================================================================
# Fig 1: the geometry of one edge (real SCVT mesh)
# ============================================================================
let m = VoronoiMesh("meshes/mesh3.nc")
    # an edge with a clearly visible offset AND a dual edge of reasonable length
    cand = [e for e in 1:m.edges.n if all(0.25 .< _pt(m.edges.position[e]) .< 0.75) &&
            edgegeom(m, e).lde >= 0.5 * edgegeom(m, e).le]
    e = cand[argmax([abs(edgegeom(m, k).delta) / edgegeom(m, k).le for k in cand])]
    g = edgegeom(m, e); h = g.le; x0 = m.edges.position[e]; sg = sign(g.delta)
    f = Figure(size = (1150, 560))
    ax = Axis(f[1, 1], aspect = DataAspect(), title = "a primal edge e and its dual edge ẽ (SCVT mesh3)")
    hidedecorations!(ax); hidespines!(ax)
    for k in 1:m.edges.n
        q = closest(x0, m.edges.position[k], 1.0, 1.0)
        norm(_pt(q) - g.me) < 2.6h || continue
        a1, a2 = edge_endpoints(m, k, true; about = x0); b1, b2 = edge_endpoints(m, k, false; about = x0)
        lines!(ax, [a1[1], a2[1]], [a1[2], a2[2]], color = LIGHTGRAY, linewidth = 1.2)
        lines!(ax, [b1[1], b2[1]], [b1[2], b2[2]], color = LIGHTBLUE, linewidth = 1.2)
    end
    poly!(ax, Point2f[g.c1, g.v2, g.c2, g.v1], color = (ORANGE, 0.12), strokewidth = 0)
    lines!(ax, [g.c1[1], g.c2[1]], [g.c1[2], g.c2[2]], color = INK, linewidth = 3.5)
    lines!(ax, [g.v1[1], g.v2[1]], [g.v1[2], g.v2[2]], color = ORANGE, linewidth = 3.5)
    scatter!(ax, [g.c1[1], g.c2[1]], [g.c1[2], g.c2[2]], color = INK, markersize = 11)
    scatter!(ax, [g.v1[1], g.v2[1]], [g.v1[2], g.v2[2]], color = ORANGE, markersize = 11)
    scatter!(ax, [g.me[1]], [g.me[2]], color = :white, strokecolor = INK, strokewidth = 2, markersize = 10)
    scatter!(ax, [g.mt[1]], [g.mt[2]], color = :white, strokecolor = ORANGE, strokewidth = 2, markersize = 10)
    text!(ax, g.c1..., text = "xₐ", align = (:center, :top), offset = (0, -8), fontsize = 14)
    text!(ax, g.c2..., text = "x_b", align = (:center, :bottom), offset = (0, 8), fontsize = 14)
    text!(ax, g.v1..., text = "c₁", align = (:right, :center), offset = (-8, 0), fontsize = 14, color = ORANGE)
    text!(ax, g.v2..., text = "c₂", align = (:left, :center), offset = (8, 0), fontsize = 14, color = ORANGE)
    text!(ax, (g.me .+ 0.30h .* g.t)..., text = "e", fontsize = 16, align = (:left, :center), offset = (8, 0))
    text!(ax, (g.v2 .- 0.25 .* (g.v2 - g.v1))..., text = "ẽ", fontsize = 16, color = ORANGE, align = (:center, :bottom), offset = (0, 6))
    p0 = g.c1 .- 0.55h .* g.n .* sg .+ 0.05h .* g.t
    arrow!(ax, p0, 0.35h .* g.t; color = MUTED, lw = 1.8)
    arrow!(ax, p0, 0.35h .* g.n; color = MUTED, lw = 1.8)
    text!(ax, (p0 .+ 0.38h .* g.t)..., text = "tₑ", color = MUTED, fontsize = 13, offset = (6, 0))
    text!(ax, (p0 .+ 0.38h .* g.n)..., text = "nₑ", color = MUTED, fontsize = 13, offset = (6, 0))
    pad = 0.9h
    xs_ = [g.c1[1], g.c2[1], g.v1[1], g.v2[1]]; ys_ = [g.c1[2], g.c2[2], g.v1[2], g.v2[2]]
    xlims!(ax, minimum(xs_) - pad, maximum(xs_) + pad); ylims!(ax, minimum(ys_) - pad, maximum(ys_) + pad)

    ax2 = Axis(f[1, 2], aspect = DataAspect(), title = @sprintf("zoom around e ∩ ẽ:  δₑ = %+.3f |ẽ|", g.delta / g.lde))
    hidedecorations!(ax2); hidespines!(ax2)
    c = (g.me .+ g.mt) ./ 2; r = max(2.2abs(g.delta), 0.15g.lde)
    lines!(ax2, [g.c1[1], g.c2[1]], [g.c1[2], g.c2[2]], color = INK, linewidth = 3.5)
    lines!(ax2, [g.v1[1], g.v2[1]], [g.v1[2], g.v2[2]], color = ORANGE, linewidth = 3.5)
    lines!(ax2, [g.me[1], g.mt[1]], [g.me[2], g.mt[2]], color = colorant"#c23b3a", linewidth = 4)
    scatter!(ax2, [g.me[1]], [g.me[2]], color = :white, strokecolor = INK, strokewidth = 2.5, markersize = 16)
    scatter!(ax2, [g.mt[1]], [g.mt[2]], color = :white, strokecolor = ORANGE, strokewidth = 2.5, markersize = 16)
    text!(ax2, (g.me .- 0.45r .* g.n .* sg .+ 0.35r .* g.t)..., text = "mₑ: midpoint of e\n= intersection e ∩ ẽ",
          fontsize = 13, align = (:left, :center))
    text!(ax2, (g.mt .+ 0.12r .* g.n .* sg .+ 0.12r .* g.t)..., text = "m̃ₑ: midpoint of ẽ", fontsize = 13,
          color = ORANGE, align = (:left, :center))
    text!(ax2, (c .- 0.30r .* g.t)..., text = "δₑ nₑ", fontsize = 15, color = colorant"#c23b3a", align = (:right, :center))
    xlims!(ax2, c[1] - 1.3r, c[1] + 1.3r); ylims!(ax2, c[2] - 1.3r, c[2] + 1.3r)
    Label(f[2, :], "grey: Delaunay (primal) edges · light blue: Voronoi (dual) edges · shaded: the diamond conv(e ∪ ẽ). " *
          "ẽ always passes through the midpoint of e (it lies on the perpendicular bisector), but its own midpoint is shifted by δₑ along nₑ.",
          fontsize = 11, color = MUTED, halign = :left, justification = :left, tellwidth = false)
    colsize!(f.layout, 1, Relative(0.55))
    savefig("fig_geom_edge.png", f)
end

# ============================================================================
# Fig 2: what a_h and s_h measure (1D picture along the normal line)
# ============================================================================
let f = Figure(size = (1100, 400))
    fu(x) = 0.6 + 0.9sin(2.2x + 0.3); fv(x) = 0.3 + 0.8cos(1.7x - 0.4)
    δ = 0.35; xs = range(-1.2, 1.2, length = 300)
    ax = Axis(f[1, 1], xlabel = "position along nₑ  (e sits at 0, the centre of ẽ at δₑ)",
              ylabel = "tangential component (· tₑ)", title = "each field is read at two places")
    vspan!(ax, [δ - 0.55], [δ + 0.55], color = (ORANGE, 0.08))
    vlines!(ax, [0.0], color = INK, linewidth = 1.5); vlines!(ax, [δ], color = ORANGE, linewidth = 1.5, linestyle = :dash)
    lines!(ax, xs, fu.(xs), color = BLUE, linewidth = 2.5, label = "u · tₑ")
    lines!(ax, xs, fv.(xs), color = AQUA, linewidth = 2.5, label = "v · tₑ")
    for (fn, c) in ((fu, BLUE), (fv, AQUA))
        scatter!(ax, [0.0, δ], [fn(0.0), fn(δ)], color = c, markersize = 11, strokecolor = :white, strokewidth = 1.5)
    end
    text!(ax, 0.0, -0.55, text = "on e (circulation)", align = (:center, :top), fontsize = 11)
    text!(ax, δ, -0.55, text = "on ẽ (flux)", align = (:left, :top), fontsize = 11, color = ORANGE, offset = (4, 0))
    ylims!(ax, -0.75, 1.65); axislegend(ax, position = :lt)
    ax2 = Axis(f[1, 2], title = "a(u,v) and a(v,u) use the two fields at swapped places")
    hidedecorations!(ax2); hidespines!(ax2)
    txt = """
    a(u, v)  ≈  |e||ẽ| · v(0) · u(δ)
    a(v, u)  ≈  |e||ẽ| · u(0) · v(δ)

    s(u, v) = a(u, v) − a(v, u)
            ≈  |e||ẽ| · δ · ( v u′ − u v′ )

    • δ = 0  →  no mismatch, s = 0
    • u, v constant  →  u′ = v′ = 0, s = 0
    • otherwise s ≠ 0, of size δ × gradient
    """
    text!(ax2, 0.02, 0.95, text = txt, space = :relative, align = (:left, :top), fontsize = 14, font = :regular)
    savefig("fig_sh_intuition.png", f)
end

# ============================================================================
# Fig 3: the midpoint offsets on three meshes, with defects marked
# ============================================================================
let f = Figure(size = (1150, 460))
    panels = [("regular (mesh3_regular)", "mesh3_regular"), ("SCVT (mesh3)", "mesh3"), ("perturbed lattice (pert3)", "pert3")]
    meshes = [VoronoiMesh("meshes/$mn.nc") for (_, mn) in panels]
    lim = maximum(maximum(abs, deltas(m)) for m in meshes)        # no clipping: scale covers the largest offset
    for (i, (lab, mn)) in enumerate(panels)
        m = meshes[i]; d = deltas(m); p5, p7 = defects(m)
        ttl = @sprintf("%s\n|δₑ|/|ẽ|: rms %.3f, max %.3f\n%d pentagons, %d heptagons",
                       lab, sqrt(sum(d.^2)/length(d)), maximum(abs, d), length(p5), length(p7))
        ax = Axis(f[1, i], aspect = DataAspect(), title = ttl)
        hidedecorations!(ax); hidespines!(ax)
        segs = Point2f[]; cols = Float64[]
        for e in 1:m.edges.n
            g = edgegeom(m, e)
            push!(segs, Point2f(g.c1...), Point2f(g.c2...)); push!(cols, d[e], d[e])
        end
        linesegments!(ax, segs, color = cols, colormap = DIV, colorrange = (-lim, lim), linewidth = 2.2)
        isempty(p5) || scatter!(ax, first.(p5), last.(p5), marker = :diamond, color = INK, markersize = 11, strokecolor = :white, strokewidth = 1)
        isempty(p7) || scatter!(ax, first.(p7), last.(p7), marker = :star5, color = INK, markersize = 13, strokecolor = :white, strokewidth = 1)
        xlims!(ax, 0, 1); ylims!(ax, 0, 1)
    end
    Colorbar(f[1, 4], colormap = DIV, limits = (-lim, lim), label = "δₑ / |ẽ|  (Delaunay edges coloured)")
    Label(f[2, :], "◆ pentagonal Voronoi cell   ★ heptagonal Voronoi cell.  Similar rms offset on both irregular meshes, but on the SCVT mesh\n" *
          "the offsets concentrate around the pentagon–heptagon defects; on the perturbed lattice they are spread everywhere.",
          fontsize = 11, color = MUTED, halign = :left, justification = :left, tellwidth = false)
    savefig("fig_delta_maps.png", f)
end

# ============================================================================
# Fig 4: where s_h lives: smoothed density of the per-edge contributions
# ============================================================================
let f = Figure(size = (1150, 820))
    σ = 0.06; ng = 121; xs = collect(range(0, 1, length = ng)); ε = 1e-3
    ut(w, x, t) = dot(w(x), t)
    d2(w, x, t, d) = (ut(w, x .+ ε .* d, t) - 2ut(w, x, t) + ut(w, x .- ε .* d, t)) / ε^2
    DW = Dict{String,Matrix{Float64}}(); Dd = Dict{String,Matrix{Float64}}(); SW = Dict(); Sd = Dict(); pts = Dict()
    for (lab, mn) in [("regular", "mesh5_regular"), ("SCVT", "mesh5"), ("perturbed", "pert5")]
        m = VoronoiMesh("meshes/$mn.nc"); c = skew_contrib(m, U, V); cW = zeros(m.edges.n)
        for e in 1:m.edges.n
            g = edgegeom(m, e); x = g.me
            uT, vT = ut(U, x, g.t), ut(V, x, g.t)
            cW[e] = g.le * g.lde * (g.le^2 / 24 * (uT * d2(V, x, g.t, g.t) - vT * d2(U, x, g.t, g.t)) +
                                    g.lde^2 / 24 * (vT * d2(U, x, g.t, g.n) - uT * d2(V, x, g.t, g.n)))
        end
        cd = c .- cW
        px = [m.edges.position[e].x for e in 1:m.edges.n]; py = [m.edges.position[e].y for e in 1:m.edges.n]
        DW[lab] = splat_density(cW, px, py, xs, σ); Dd[lab] = splat_density(cd, px, py, xs, σ)
        SW[lab] = sum(cW); Sd[lab] = sum(cd); pts[lab] = defects(m)
        @printf("  %-9s  sum c = %+.2e   sum c_W = %+.2e   sum c_delta = %+.2e   max|D_W| = %.2e   max|D_delta| = %.2e\n",
                lab, sum(c), SW[lab], Sd[lab], maximum(abs, DW[lab]), maximum(abs, Dd[lab]))
    end
    limW = maximum(maximum(abs, D) for D in values(DW)); limd = maximum(maximum(abs, D) for D in values(Dd))
    for (row, D, S, lim, what) in ((1, DW, SW, limW, "Wronskian part"), (2, Dd, Sd, limd, "δ part"))
        for (i, lab) in enumerate(["regular", "SCVT", "perturbed"])
            ax = Axis(f[row, i], aspect = DataAspect(), title = @sprintf("%s, %s:  Σ = %+.1e", lab, what, S[lab]))
            hidedecorations!(ax); hidespines!(ax)
            heatmap!(ax, xs, xs, D[lab], colormap = DIV, colorrange = (-lim, lim))
            p5, p7 = pts[lab]
            if row == 2
                isempty(p5) || scatter!(ax, first.(p5), last.(p5), marker = :diamond, color = INK, markersize = 9, strokecolor = :white, strokewidth = 1)
                isempty(p7) || scatter!(ax, first.(p7), last.(p7), marker = :star5, color = INK, markersize = 10, strokecolor = :white, strokewidth = 1)
            end
            xlims!(ax, 0, 1); ylims!(ax, 0, 1)
        end
        Colorbar(f[row, 4], colormap = DIV, limits = (-lim, lim), label = "smoothed density")
    end
    Label(f[3, :], "s_h(u,v) = Σₑ cₑ split into the δ-independent Wronskian terms (top) and the remainder, driven by δₑ (bottom). " *
          "\nSame u, v as elsewhere; Gaussian smoothing of width 0.06; the two rows have different colour scales.  ◆ pentagons, ★ heptagons.",
          fontsize = 11, color = MUTED, halign = :left, justification = :left, tellwidth = false)
    savefig("fig_skew_density.png", f)
end

# ============================================================================
# Fig 5: s_h under refinement
# ============================================================================
let f = Figure(size = (640, 430))
    ax = Axis(f[1, 1], xscale = log10, yscale = log10, xlabel = "h", xreversed = true,
              ylabel = "|s_h(u,v)| / (‖u‖ ‖v‖)", title = "the mesh quantity s_h under refinement")
    allh = Float64[]
    for (fam, mns) in FAMS
        hs = Float64[]; ss = Float64[]
        for mn in mns
            m = VoronoiMesh("meshes/$mn.nc"); push!(hs, meanh(m)); push!(ss, s_rel(m, U, V))
        end
        append!(allh, hs)
        lines!(ax, hs, ss, color = FAMCOL[fam], linewidth = 2.5, label = fam)
        scatter!(ax, hs, ss, color = FAMCOL[fam], marker = FAMMRK[fam], markersize = 10, strokecolor = :white, strokewidth = 1)
        text!(ax, hs[end], ss[end], text = fam, color = INK, fontsize = 11, align = (:left, :center), offset = (8, 0))
    end
    hh = sort(unique(allh)); h0 = maximum(hh)
    lines!(ax, hh, 2e-3 .* (hh ./ h0).^2, color = MUTED, linestyle = :dash, label = "slope 2")
    lines!(ax, hh, 2e-3 .* (hh ./ h0).^4, color = MUTED, linestyle = :dot, label = "slope 4")
    xlims!(ax, 1.1h0, minimum(hh) / 1.9)
    axislegend(ax, position = :lc)
    savefig("fig_sh_rates.png", f)
end

# ============================================================================
# Fig 6: the two-mode model
# ============================================================================
let f = Figure(size = (1000, 380))
    λ1, λ2 = 1.0, 1.2; κs = range(0, 0.25, length = 400)
    ev = [eigvals([λ1 k; -k λ2]) for k in κs]
    ax1 = Axis(f[1, 1], xlabel = "coupling κ from the antisymmetric part", ylabel = "Re λ",
               title = "real parts: the split pair is pulled together")
    ax2 = Axis(f[1, 2], xlabel = "coupling κ from the antisymmetric part", ylabel = "|Im λ|",
               title = "imaginary parts: zero until κ = Δ/2")
    for k in 1:2
        lines!(ax1, κs, [real(e[k]) for e in ev], color = BLUE, linewidth = 2.5)
    end
    lines!(ax2, κs, [abs(imag(e[1])) for e in ev], color = ORANGE, linewidth = 2.5)
    for ax in (ax1, ax2)
        vlines!(ax, [0.1], color = MUTED, linestyle = :dash)
    end
    text!(ax1, 0.1, 1.18, text = "κ = Δ/2", fontsize = 11, color = MUTED, align = (:left, :top), offset = (6, 0))
    text!(ax2, 0.1, 0.2, text = "κ = Δ/2: the pair collides\nand leaves the real axis", fontsize = 11, align = (:left, :top), offset = (6, 0))
    Label(f[2, :], "Matrix [λ₁ κ; −κ λ₂] with λ₁ = 1, λ₂ = 1.2 (splitting Δ = 0.2). Eigenvalues (λ₁+λ₂)/2 ± √(Δ²/4 − κ²).",
          fontsize = 11, color = MUTED, halign = :left, justification = :left, tellwidth = false)
    savefig("fig_2x2.png", f)
end

QUICK && (println("quick mode: skipping blow-up and Darcy figures"); exit())

# ============================================================================
# Fig 7: Theorem 4.2 in action -- coefficients of the exact symmetric P2 star
# ============================================================================
let f = Figure(size = (1100, 400))
    ax1 = Axis(f[1, 1], xscale = log10, yscale = log10, xlabel = "h", xreversed = true,
               ylabel = "Λ(H) = maxₑ Σₘ |Hₑₘ|", title = "(a) coefficient size of the exact symmetric P₂ star")
    ax2 = Axis(f[1, 2], xscale = log10, yscale = log10, xlabel = "h", xreversed = true,
               ylabel = "‖H − sym H_lift‖ / ‖sym H_lift‖", title = "(b) how far it must move from the lift")
    for (fam, mns) in FAMS
        hs = Float64[]; Λs = Float64[]; Λl = Float64[]; ds = Float64[]
        for mn in mns
            m = VoronoiMesh("meshes/$mn.nc"); Hl = hodge_star_lift_matrix(m; only_linear = false)
            r = hodge_star_symmetric_matrix(m; only_linear = false, level = 5, consistency = 2, Href = Hl)
            push!(hs, meanh(m)); push!(Λs, opnorm(r.H, Inf)); push!(Λl, opnorm(Hl, Inf)); push!(ds, r.dist)
            @printf("  blow-up %-10s %-14s Lambda(sym)=%.3f Lambda(lift)=%.3f dist=%.3f feas=%.1e\n", fam, mn, Λs[end], Λl[end], ds[end], r.feas)
        end
        lines!(ax1, hs, Λs, color = FAMCOL[fam], linewidth = 2.5, label = "$fam: symmetric")
        scatter!(ax1, hs, Λs, color = FAMCOL[fam], marker = FAMMRK[fam], markersize = 10, strokecolor = :white, strokewidth = 1)
        lines!(ax1, hs, Λl, color = FAMCOL[fam], linewidth = 1.5, linestyle = :dash, label = "$fam: lift")
        lines!(ax2, hs, ds, color = FAMCOL[fam], linewidth = 2.5, label = fam)
        scatter!(ax2, hs, ds, color = FAMCOL[fam], marker = FAMMRK[fam], markersize = 10, strokecolor = :white, strokewidth = 1)
    end
    Legend(f[1, 3], ax1)
    Label(f[2, :], "Exactly symmetric, row-wise exact on [P₂]², widest pattern (level 5). Dashed: the (non-symmetric) lift, for reference.",
          fontsize = 11, color = MUTED, halign = :left, justification = :left, tellwidth = false)
    savefig("fig_blowup.png", f)
end

# ============================================================================
# Fig 8: Darcy convergence, lift vs symmetric, per family
# ============================================================================
pex(x)  = sin(2pi*x[1])*sin(2pi*x[2]) + 0.3cos(2pi*(x[1] - 2x[2]))
gpex(x) = [2pi*cos(2pi*x[1])*sin(2pi*x[2]) - 0.6pi*sin(2pi*(x[1]-2x[2])),
           2pi*sin(2pi*x[1])*cos(2pi*x[2]) + 1.2pi*sin(2pi*(x[1]-2x[2]))]
lapp(x) = -8pi^2*sin(2pi*x[1])*sin(2pi*x[2]) - 6pi^2*cos(2pi*(x[1]-2x[2]))
uex(x)  = (g = gpex(x); [g[2], -g[1]])
function darcy(m, H)
    D1 = sparse(Float64.(d1_matrix_primal(m))); G = sparse(Float64.(d0_matrix_dual(m)))
    nE, nT = m.edges.n, m.vertices.n
    fex = reduce_primal_edges(m, uex).values
    pT  = [pex(_pt(m.vertices.position[t])) for t in 1:nT]
    phi = discretize_scalar_field_triangles_ho(m, x -> -lapp(x)).values; phi .-= sum(phi)/nT
    K = [sparse(H) -G; D1 spzeros(nT, nT)]; keep = setdiff(1:nE+nT, [nE+1])
    sol = zeros(nE+nT); sol[keep] = K[keep, keep] \ [zeros(nE); phi][keep]
    f = sol[1:nE]; p = sol[nE+1:end]
    w = edge_lengths(m, false) ./ edge_lengths(m, true); A = m.vertices.area
    pp = p .- sum(A .* p)/sum(A); pe = pT .- sum(A .* pT)/sum(A)
    return sqrt(sum(w .* (f .- fex).^2) / sum(w .* fex.^2)), sqrt(sum(A .* (pp .- pe).^2) / sum(A .* pe.^2))
end
let f = Figure(size = (1150, 640))
    for (j, (fam, mns)) in enumerate(FAMS)
        hs = Float64[]; R = Dict(k => Float64[] for k in ("lf", "lp", "sf", "sp"))
        for mn in mns
            m = VoronoiMesh("meshes/$mn.nc"); Hl = hodge_star_lift_matrix(m; only_linear = false)
            r = fam == "regular" ?
                hodge_star_symmetric_matrix(m; only_linear = false, level = 4, consistency = 2, Href = Hl) :
                hodge_star_symmetric_matrix(m; only_linear = false, level = 4, consistency = 2, hard_degree = 1, mu_soft = 1e-4, Href = Hl)
            Hs = r.H + polynomial_blind_stabilisation(m; only_linear = false, level = 3)
            a, b = darcy(m, Hl); c, d = darcy(m, Hs)
            push!(hs, meanh(m)); push!(R["lf"], a); push!(R["lp"], b); push!(R["sf"], c); push!(R["sp"], d)
        end
        for (i, (kl, ks, what)) in enumerate([("lf", "sf", "flux"), ("lp", "sp", "pressure")])
            ax = Axis(f[i, j], xscale = log10, yscale = log10, xlabel = i == 2 ? "h" : "", xreversed = true,
                      ylabel = j == 1 ? "relative $what error" : "", title = i == 1 ? fam : "")
            lines!(ax, hs, R[kl], color = BLUE, linewidth = 2.5, label = "lift [P₂]² (non-symmetric)")
            scatter!(ax, hs, R[kl], color = BLUE, marker = :circle, markersize = 9, strokecolor = :white, strokewidth = 1)
            lines!(ax, hs, R[ks], color = ORANGE, linewidth = 2.5, label = "symmetric star")
            scatter!(ax, hs, R[ks], color = ORANGE, marker = :utriangle, markersize = 10, strokecolor = :white, strokewidth = 1)
            ref = R[kl][1]
            lines!(ax, hs, ref .* (hs ./ hs[1]).^2, color = MUTED, linestyle = :dash)
            lines!(ax, hs, ref .* (hs ./ hs[1]).^4, color = MUTED, linestyle = :dot)
            (i == 1 && j == 1) && axislegend(ax, position = :lb)
        end
    end
    Label(f[3, :], "Mixed Darcy problem. Dashed: slope 2, dotted: slope 4. Symmetric star: exact on [P₂]² for the regular family, " *
          "exact on [P₁]² with [P₂]² by penalty for the others, plus polynomial-blind stabilisation.",
          fontsize = 11, color = MUTED, halign = :left, justification = :left, tellwidth = false)
    savefig("fig_darcy_families.png", f)
end
println("done")
