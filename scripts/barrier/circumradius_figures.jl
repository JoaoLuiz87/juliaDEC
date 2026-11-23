# ============================================================================
#  circumradius_figures.jl  -- the offset as a dual gradient of half the squared
#  circumradius (section "Geometry" and section "Related work" of
#  hodge_symmetry_consistency.tex)
#
#      julia scripts/barrier/circumradius_figures.jl
#
#  Checks |e~| delta_e = (D1' rho)_e with rho_T = R_T^2/2 on every mesh used in
#  the note, and draws fig_circumradius.png.
# ============================================================================
using VoronoiMeshes, TensorsLite, TensorsLiteGeometry, SmallCollections, LinearAlgebra, NCDatasets
using StaticArrays, SparseArrays, Printf, CairoMakie
Base.similar(bc::Base.Broadcast.Broadcasted, ::Type{Union{}}) = Array{Union{}}(undef, length.(axes(bc)))
const ROOT = joinpath(@__DIR__, "..", "..")
cd(ROOT)
include(joinpath(ROOT, "src", "juliaDEC.jl")); include(joinpath(ROOT, "src", "hodge_stars.jl"))
const OUT = joinpath(ROOT, "report")

const INK = colorant"#2b2b29"; const MUTED = colorant"#8a8a85"; const GRID = colorant"#e6e5e1"
const BLUE = colorant"#2a78d6"; const ORANGE = colorant"#eb6834"; const AQUA = colorant"#1baf7a"
const DIV = cgrad([colorant"#1c5cab", colorant"#f0efec", colorant"#c23b3a"])
const FAMS = [("regular",   ["mesh2_regular", "mesh3_regular", "mesh4_regular", "mesh5_regular"]),
              ("SCVT",      ["mesh2", "mesh3", "mesh4", "mesh5"]),
              ("perturbed", ["pert2", "pert3", "pert4", "pert5"])]
const FAMCOL = Dict("regular" => BLUE, "SCVT" => ORANGE, "perturbed" => AQUA)
const FAMMRK = Dict("regular" => :circle, "SCVT" => :utriangle, "perturbed" => :rect)
set_theme!(Theme(fontsize = 13, textcolor = INK,
    Axis = (xgridcolor = GRID, ygridcolor = GRID, spinewidth = 0.8, leftspinecolor = MUTED,
            bottomspinecolor = MUTED, topspinevisible = false, rightspinevisible = false,
            xtickcolor = MUTED, ytickcolor = MUTED, titlefont = :regular, titlesize = 13),
    Legend = (framevisible = false, labelsize = 12)))
savefig(name, f) = (CairoMakie.save(joinpath(OUT, name), f, px_per_unit = 2); println("  wrote ", name))
meanh(m) = sum(edge_lengths(m, true)) / m.edges.n

# circumradius of every triangle (= distance from its circumcentre to each of its three generators)
function circumradii(m)
    R = zeros(m.vertices.n); spread = 0.0
    for T in 1:m.vertices.n
        v = m.vertices.position[T]
        r = [norm(_pt(closest(v, m.cells.position[c], m.x_period, m.y_period)) - _pt(v)) for c in m.vertices.cells[T]]
        R[T] = r[1]; spread = max(spread, maximum(r) - minimum(r))
    end
    return R, spread
end
# offset delta_e (n_e = R t_e) and dual length |e~|
function offsets(m)
    d = zeros(m.edges.n); L = zeros(m.edges.n)
    for e in 1:m.edges.n
        c1, c2 = edge_endpoints(m, e, true); v1, v2 = edge_endpoints(m, e, false)
        t = (c2 - c1) / norm(c2 - c1); n = [-t[2], t[1]]
        d[e] = dot((v1 + v2) / 2 - (c1 + c2) / 2, n); L[e] = norm(v2 - v1)
    end
    return d, L
end
defects(m) = ([_pt(m.cells.position[c]) for c in 1:m.cells.n if length(m.cells.edges[c]) == 5],
              [_pt(m.cells.position[c]) for c in 1:m.cells.n if length(m.cells.edges[c]) == 7])
function tri_polys(m)
    polys = Vector{Vector{Point2f}}(); owner = Int[]
    for T in 1:m.vertices.n
        v = m.vertices.position[T]
        P = [Point2f(_pt(closest(v, m.cells.position[c], m.x_period, m.y_period))...) for c in m.vertices.cells[T]]
        for sx in -1:1, sy in -1:1
            Q = [p .+ Point2f(sx, sy) for p in P]
            xs = first.(Q); ys = last.(Q)
            (maximum(xs) < 0 || minimum(xs) > 1 || maximum(ys) < 0 || minimum(ys) > 1) && continue
            push!(polys, Q); push!(owner, T)
        end
    end
    return polys, owner
end

# ============================================================================
# (1) the identity |e~| delta = D1' rho on every mesh, and the statistics of delta
# ============================================================================
println("identity |e~| delta_e = (D1' rho)_e,  rho_T = R_T^2 / 2")
stats = Dict{String, Vector{NTuple{3, Float64}}}()
for (fam, mns) in FAMS
    stats[fam] = NTuple{3, Float64}[]
    for mn in mns
        m = VoronoiMesh("meshes/$mn.nc")
        R, spread = circumradii(m); d, L = offsets(m)
        D1 = sparse(Float64.(d1_matrix_primal(m))); G = sparse(Float64.(d0_matrix_dual(m)))
        rho = R .^ 2 ./ 2
        err = maximum(abs, D1' * rho .- L .* d); errG = maximum(abs, G * rho .- L .* d)
        r = abs.(d) ./ L
        push!(stats[fam], (meanh(m), maximum(r), sqrt(sum(L .* m.edges.n .* r.^2) / sum(L) / m.edges.n)))
        @printf("  %-14s h=%.4f  max|D1'rho - |e~|delta| = %.1e (rel %.1e)  [dual-incidence form %.1e]  R spread %.1e  max|delta|/|e~| %.3f  rms %.3f  |D1 d0| = %d\n",
                mn, meanh(m), err, err / maximum(abs, L .* d .+ eps()), errG, spread, maximum(r),
                sqrt(sum(r.^2) / length(r)), Int(maximum(abs, D1 * sparse(Float64.(d0_matrix_primal(m))))))
    end
end

# ============================================================================
# (2) figure: the potential rho = R^2/2 on SCVT and perturbed meshes, and the size of delta/|e~|
# ============================================================================
let f = Figure(size = (1250, 470))
    panels = [("(a) SCVT (mesh4)", "mesh4"), ("(b) perturbed lattice (pert4)", "pert4")]
    vals = []; ms = []
    for (_, mn) in panels
        m = VoronoiMesh("meshes/$mn.nc"); R, _ = circumradii(m)
        push!(ms, m); push!(vals, R .^ 2 ./ (sum(R .^ 2) / length(R)) .- 1)
    end
    lim = maximum(maximum(abs, v) for v in vals)
    for (i, (ttl, _)) in enumerate(panels)
        m = ms[i]; v = vals[i]; polys, owner = tri_polys(m)
        ax = Axis(f[1, i], aspect = DataAspect(),
                  title = @sprintf("%s\nR² / mean(R²) − 1 per triangle, max |·| %.3f", ttl, maximum(abs, v)))
        hidedecorations!(ax); hidespines!(ax)
        poly!(ax, polys, color = v[owner], colormap = DIV, colorrange = (-lim, lim), strokewidth = 0)
        p5, p7 = defects(m)
        isempty(p5) || scatter!(ax, first.(p5), last.(p5), marker = :diamond, color = INK, markersize = 9, strokecolor = :white, strokewidth = 1)
        isempty(p7) || scatter!(ax, first.(p7), last.(p7), marker = :star5, color = INK, markersize = 10, strokecolor = :white, strokewidth = 1)
        xlims!(ax, 0, 1); ylims!(ax, 0, 1)
    end
    Colorbar(f[1, 3], colormap = DIV, limits = (-lim, lim), width = 10, ticklabelsize = 10)
    ax = Axis(f[1, 4], xscale = log10, yscale = log10, xlabel = "h", xreversed = true,
              ylabel = "|δₑ| / |ẽ|", title = "(c) size of the offsets under refinement")
    for fam in ("SCVT", "perturbed")
        s = stats[fam]; hs = first.(s)
        lines!(ax, hs, [x[2] for x in s], color = FAMCOL[fam], linewidth = 2.5, label = "$fam, max")
        scatter!(ax, hs, [x[2] for x in s], color = FAMCOL[fam], marker = FAMMRK[fam], markersize = 10, strokecolor = :white, strokewidth = 1)
        lines!(ax, hs, [x[3] for x in s], color = FAMCOL[fam], linewidth = 2, linestyle = :dash, label = "$fam, rms")
    end
    hs = first.(stats["SCVT"])
    lines!(ax, hs, 0.1 .* (hs ./ hs[1]), color = MUTED, linestyle = :dot, linewidth = 1.5, label = "slope 1 (δ = O(h²))")
    ylims!(ax, 10^-2.9, 10^-0.35)
    axislegend(ax, position = :lb, labelsize = 10)
    colgap!(f.layout, 10)
    Label(f[2, :], "(a), (b): Delaunay triangles coloured by the relative deviation of their squared circumradius; one colour scale.  " *
          "The offset is the jump of R²/2 across each Delaunay edge divided by |ẽ|.\n" *
          "◆ pentagons, ★ heptagons.  (c) max and rms (weighted by |ẽ|) of |δₑ|/|ẽ|; on the regular family it is below 10⁻⁹ and is not shown.",
          fontsize = 11, color = MUTED, halign = :left, justification = :left, tellwidth = false)
    savefig("fig_circumradius.png", f)
end
println("done")
