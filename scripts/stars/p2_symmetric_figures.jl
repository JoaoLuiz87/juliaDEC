# ============================================================================
#  p2_symmetric_figures.jl -- the figures of Sections 2.2 and 2.5 of three_stars_report.tex.
#
#      julia scripts/stars/p2_symmetric_figures.jl      (after p2_symmetric_check.jl; also writes ts_data/stencils.txt)
#
#  fig_ts_stencils.png  the four stencils of edge 17 of mesh 3, and the two [P2]^2 fields the 13-edge stencil cannot see
#  fig_ts_p2sym.png     [P2]^2 stars on these stencils, symmetric (the H_P1 recipe) against row-wise (the H_P2 recipe)
# ============================================================================
include(joinpath(@__DIR__, "three_stars.jl"))
using CairoMakie
empty!(lines)
outp(name) = joinpath(@__DIR__, name)
const DIV = cgrad([colorant"#1c5cab", colorant"#f0efec", colorant"#c23b3a"])
const SCOL = Dict(:t2 => colorant"#7a7a7a", 3 => colorant"#c23b3a", :t3 => colorant"#2b6cb0", 4 => colorant"#2f8f4e")
const SMK = Dict(:t2 => :circle, 3 => :rect, :t3 => :diamond, 4 => :utriangle)

# ---------------------------------------------------------------- 1. stencils and the fields the 13-edge stencil misses
let mesh = VoronoiMesh("meshes/mesh3.nc"), e = 17
    xe = mesh.edges.position[e]; x0 = _pt(xe); s = edge_lengths(mesh, true)[e]
    seg(j) = edge_endpoints(mesh, j, true; about = xe)
    near = [j for j in 1:mesh.edges.n if norm(_pt(closest(xe, mesh.edges.position[j], 1.0, 1.0)) .- x0) < 7s]
    q1, q2 = edge_endpoints(mesh, e, false; about = xe)
    verts(S) = collect(values(Dict(round.(p; digits = 10) => p for j in S for p in seg(j))))     # unwrapped vertices, deduplicated
    L = 3.3s
    f = Figure(size = (1350, 900))
    function base(pos, title)
        ax = Axis(pos; aspect = DataAspect(), title = title, titlesize = 14); hidedecorations!(ax)
        for j in near
            a, b = seg(j); lines!(ax, [a[1], b[1]], [a[2], b[2]]; color = (:gray, 0.3), linewidth = 0.8)
        end
        xlims!(ax, x0[1] - L, x0[1] + L); ylims!(ax, x0[2] - L, x0[2] + L)
        return ax
    end
    function stencil!(ax, S, col)
        for j in S
            a, b = seg(j); lines!(ax, [a[1], b[1]], [a[2], b[2]]; color = col, linewidth = 3.5)
        end
        V = verts(S); scatter!(ax, [p[1] for p in V], [p[2] for p in V]; color = :black, markersize = 8)
        a, b = seg(e); lines!(ax, [a[1], b[1]], [a[2], b[2]]; color = :black, linewidth = 2.5)
        lines!(ax, [q1[1], q2[1]], [q1[2], q2[2]]; color = :black, linewidth = 2, linestyle = :dash)
    end
    labels = Dict(:t2 => "(a) 13-edge stencil\n(get_neighbors_2level_primal)", 3 => "(b) 3-level stencil S_e, used by H_P1 and H_P2\n(get_neighbors_3level_primal)",
                  :t3 => "(c) three layers of triangles", 4 => "(d) 4-ring\n(13-edge stencils of the 13 edges)")
    for (i, st) in enumerate(STENCILS)
        S = ring(mesh, e, st); V = verts(S)
        ax = base(f[(i - 1) ÷ 3 + 1, (i - 1) % 3 + 1], labels[st] * @sprintf(":  E = %d edges, V = %d vertices", length(S), length(V)))
        stencil!(ax, S, SCOL[st])
        logp(@sprintf("stencil %-28s of edge %d (mesh 3): E = %d, V = %d, cycles = %d", SNAME[st], e, length(S), length(V), length(S) - length(V) + 1))
    end
    # the cubics that vanish at the 8 vertices of the 13-edge stencil: their gradients have zero circulation on every
    # stencil edge, but not zero flux across e~
    S = ring(mesh, e, :t2); V = verts(S)
    mono(X, Y) = [1.0, X, Y, X^2, X * Y, Y^2, X^3, X^2 * Y, X * Y^2, Y^3]
    sc(p) = ((p[1] - x0[1]) / s, (p[2] - x0[2]) / s)
    Z = nullspace(reduce(vcat, [mono(sc(p)...)' for p in V]))
    logp(@sprintf("13-edge stencil: %d vertices, the cubics vanishing at all of them form a space of dimension %d", length(V), size(Z, 2)))
    GL5X = [-0.9061798459386640, -0.5384693101056831, 0.0, 0.5384693101056831, 0.9061798459386640]
    GL5W = [0.2369268850561891, 0.4786286704993665, 0.5688888888888889, 0.4786286704993665, 0.2369268850561891]
    te = (seg(e)[2] .- seg(e)[1]) ./ s
    for k in 1:size(Z, 2)
        c = Z[:, k] ./ maximum(abs, Z[:, k])
        φ(p) = dot(c, mono(sc(p)...))
        ∇φ(p) = (h_ = 1e-6s; [(φ(p .+ [h_, 0]) - φ(p .- [h_, 0])) / 2h_, (φ(p .+ [0, h_]) - φ(p .- [0, h_])) / 2h_])
        circ = maximum(abs(φ(seg(j)[2]) - φ(seg(j)[1])) for j in S)                          # = ∫_j ∇φ·t exactly
        flux = norm(q2 .- q1) * sum(w / 2 * dot(∇φ(q1 .+ (ξ + 1) / 2 .* (q2 .- q1)), te) for (ξ, w) in zip(GL5X, GL5W))
        gmax = maximum(norm(∇φ(x0 .+ s .* [cos(θ), sin(θ)] .* r)) for θ in range(0, 2π; length = 60), r in (0.5, 1.0, 1.5))
        logp(@sprintf("   field ∇φ_%d: max circulation on the 13 edges %.1e, flux across e~ %.3e = %.3f |e~| max|∇φ| near e", k, circ, flux,
                      flux / (norm(q2 .- q1) * gmax)))
        ax = base(f[2, k + 1], "($(Char('d' + k))) φ$(k == 1 ? "₁" : "₂"), a cubic that vanishes at the 8 vertices of (a):\n" *
                  "∇φ has zero circulation on all 13 edges, but a flux across ẽ")
        xs = range(x0[1] - L, x0[1] + L; length = 160); ys = range(x0[2] - L, x0[2] + L; length = 160)
        F = [φ([x, y]) for x in xs, y in ys]; m = maximum(abs, F[abs.(xs .- x0[1]) .< 2.5s, abs.(ys .- x0[2]) .< 2.5s])
        heatmap!(ax, xs, ys, F; colormap = DIV, colorrange = (-m, m))
        contour!(ax, xs, ys, F; levels = [0.0], color = :black, linewidth = 1)
        stencil!(ax, S, (SCOL[:t2], 0.9))
    end
    CairoMakie.save(outp("fig_ts_stencils.png"), f; px_per_unit = 2)
end

# ---------------------------------------------------------------- 2. symmetric against row-wise [P2]^2 stars
#   Coloured curves: the eight stars exact on [P2]^2.  Black curves: H_DEC and H_P1 itself (exact on [P1]^2), measured the
#   same way (ts_data/p2_symmetric_refs.jls, from p2_symmetric_refs.jl), so that the dashed curves are not read as H_P1.
let R = deserialize(joinpath(TSDIR, "p2_symmetric.jls")), f = Figure(size = (1450, 590))
    RF = deserialize(joinpath(TSDIR, "p2_symmetric_refs.jls"))
    ms = ("mesh2", "mesh3", "mesh4", "mesh5", "mesh6")
    hs = [R[(m, :h)] for m in ms]; xt = (hs, ["1/8", "1/16", "1/32", "1/64", "1/128"])
    FNAME = Dict(:t2 => "13-edge (13 entries/row)", 3 => "3-ring (19)", :t3 => "3 triangle layers (27)", 4 => "4-ring (45)")   # names of Table symp2
    REFSTYLE = Dict("H_DEC" => (linestyle = :dashdot, marker = :circle, markercolor = :white), "H_P1" => (linestyle = :solid, marker = :xcross, markercolor = :black))
    function panel(pos, title, ylabel, fld; slopes = ())
        ax = Axis(pos; xscale = log10, yscale = log10, xreversed = true, xticks = xt, xlabel = "h", ylabel = ylabel, title = title, titlesize = 14,
                  yticks = (10.0 .^ (-8:2:4), ["10⁻⁸", "10⁻⁶", "10⁻⁴", "10⁻²", "1", "10²", "10⁴"]))
        for st in STENCILS, recipe in (:row, :sym)
            pts = [(R[(m, :h)], getfield(R[(m, st, recipe)], fld)) for m in ms if haskey(R, (m, st, recipe))]
            scatterlines!(ax, first.(pts), last.(pts); color = SCOL[st], marker = SMK[st], markersize = 10,
                          linestyle = recipe == :sym ? :dash : :solid, strokecolor = SCOL[st], strokewidth = 1.5,
                          markercolor = recipe == :sym ? :white : SCOL[st])
        end
        for name in ("H_DEC", "H_P1")
            st_ = REFSTYLE[name]
            scatterlines!(ax, hs, [getfield(RF[(m, name)], fld) for m in ms]; color = :black, linewidth = 2.2, linestyle = st_.linestyle,
                          marker = st_.marker, markersize = 11, markercolor = st_.markercolor, strokecolor = :black, strokewidth = 1.5)
        end
        for (k, y0) in slopes
            lines!(ax, [hs[3], hs[5]], y0 .* ([hs[3], hs[5]] ./ hs[3]) .^ k; color = :gray50, linewidth = 1, linestyle = :dot)
            text!(ax, hs[5], y0 * (hs[5] / hs[3])^k; text = "  h" * Dict(1 => "", 2 => "²", 3 => "³", 4 => "⁴")[k], fontsize = 12, align = (:left, :center))
        end
        return ax
    end
    panel(f[1, 1], "(a) Hodge error, field d (rms)", "r₂(d) = rms of |ε_e|/|ẽ|, relative", :hodge; slopes = ((2, 3e-4), (3, 4e-5)))
    panel(f[1, 2], "(b) size of the star Λ", "Λ = max_e Σ_j |H_ej||e_j| / |ẽ_e|", :Λ)
    panel(f[1, 3], "(c) Poisson, nodal L² error", "‖p_h − p‖_A / ‖p‖_A", :poisson; slopes = ((2, 1e-3), (4, 1e-4)))
    elems = [[LineElement(color = SCOL[st]), MarkerElement(color = SCOL[st], marker = SMK[st], markersize = 10)] for st in STENCILS]
    Legend(f[2, 1:3], elems, [FNAME[st] for st in STENCILS], "stencil of the stars exact on [P2]²"; orientation = :horizontal,
           framevisible = false, labelsize = 13, titlesize = 13)
    starel = [LineElement(color = :black, linestyle = :solid),
              LineElement(color = :black, linestyle = :dash),
              [LineElement(color = :black, linestyle = :dashdot, linewidth = 2.2), MarkerElement(color = :white, marker = :circle, markersize = 11, strokecolor = :black, strokewidth = 1.5)],
              [LineElement(color = :black, linestyle = :solid, linewidth = 2.2), MarkerElement(color = :black, marker = :xcross, markersize = 11)]]
    Legend(f[3, 1:3], starel, ["exact on [P2]², row by row (on the 3-ring: H_P2)", "exact on [P2]², symmetric least change",
                               "H_DEC", "H_P1 (symmetric, exact on [P1]², 3-ring)"]; orientation = :horizontal, framevisible = false, labelsize = 13)
    CairoMakie.save(outp("fig_ts_p2sym.png"), f; px_per_unit = 2)
end
write(joinpath(TSDIR, "stencils.txt"), join(lines, "\n") * "\n")
