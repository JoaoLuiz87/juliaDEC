# ============================================================================
#  peixoto_figures.jl  -- figures for the section "Relation to Peixoto (2016)"
#  of hodge_symmetry_consistency.tex
#
#      julia scripts/archive/peixoto_figures.jl
#
#  Needs the symmetric-star construction. By default it is read from
#  juliaDEC/symmetric_star.jl; set the environment variable SYMSTAR to use
#  another copy.
# ============================================================================
using VoronoiMeshes, TensorsLite, TensorsLiteGeometry, SmallCollections, LinearAlgebra, NCDatasets
using StaticArrays, SparseArrays, QuadGK, Printf, CairoMakie
Base.similar(bc::Base.Broadcast.Broadcasted, ::Type{Union{}}) = Array{Union{}}(undef, length.(axes(bc)))
const ROOT = joinpath(@__DIR__, "..", "..")
cd(ROOT)
include(joinpath(ROOT, "src", "juliaDEC.jl")); include(joinpath(ROOT, "src", "hodge_stars.jl"))
include(get(ENV, "SYMSTAR", joinpath(ROOT, "symmetric_star.jl")))
const OUT = joinpath(ROOT, "report")

const INK = colorant"#2b2b29"; const MUTED = colorant"#8a8a85"; const GRID = colorant"#e6e5e1"
const BLUE = colorant"#2a78d6"; const ORANGE = colorant"#eb6834"; const AQUA = colorant"#1baf7a"
const DIV = cgrad([colorant"#1c5cab", colorant"#f0efec", colorant"#c23b3a"])
set_theme!(Theme(fontsize = 13, textcolor = INK,
    Axis = (xgridcolor = GRID, ygridcolor = GRID, spinewidth = 0.8, leftspinecolor = MUTED,
            bottomspinecolor = MUTED, topspinevisible = false, rightspinevisible = false,
            xtickcolor = MUTED, ytickcolor = MUTED, titlefont = :regular, titlesize = 13),
    Legend = (framevisible = false, labelsize = 12)))
savefig(name, f) = (CairoMakie.save(joinpath(OUT, name), f, px_per_unit = 2); println("  wrote ", name))
meanh(m) = sum(edge_lengths(m, true)) / m.edges.n

pf(x)   = sin(2pi*x[1]) * sin(2pi*x[2]) + 0.3cos(2pi*(x[1] - 2x[2]))
lapf(x) = -8pi^2*sin(2pi*x[1])*sin(2pi*x[2]) - 6pi^2*cos(2pi*(x[1] - 2x[2]))

diag_star(m) = spdiagm(0 => edge_lengths(m, false) ./ edge_lengths(m, true))
# relative pointwise truncation error of (1/|V_c|) d0' H d0 against the cell average of -Lap p
function truncation(m, H)
    d0  = sparse(Float64.(d0_matrix_primal(m)))
    p   = [pf(_pt(m.cells.position[c])) for c in 1:m.cells.n]
    avg = discretize_scalar_field_cells(m, lapf).values ./ m.cells.area
    return ((d0' * (H * (d0 * p))) ./ m.cells.area .+ avg) ./ maximum(abs, avg)
end
# local TRSK kinetic energy K_c = 1/(4|V_c|) sum alpha_e (H0 alpha)_e for a constant field, relative error
function ke_error(m, a)
    err = zeros(m.cells.n)
    for c in 1:m.cells.n
        K = 0.0
        for e in Int.(m.cells.edges[c])
            c1, c2 = edge_endpoints(m, e, true); v1, v2 = edge_endpoints(m, e, false)
            t = (c2 - c1) / norm(c2 - c1)
            K += norm(c2 - c1) * norm(v2 - v1) * dot(a, t)^2
        end
        err[c] = (K / (4m.cells.area[c]) - dot(a, a)/2) / (dot(a, a)/2)
    end
    return err
end
symmetric_soft(m, Hl) = hodge_star_symmetric_matrix(m; only_linear = false, level = 4, consistency = 2,
                            hard_degree = 1, mu_soft = 1e-4, Href = Hl).H +
                        polynomial_blind_stabilisation(m; only_linear = false, level = 3)

# Voronoi cells as polygons, with periodic copies for cells that cross the seam
function cell_polys(m)
    polys = Vector{Vector{Point2f}}(); owner = Int[]
    for c in 1:m.cells.n
        xc = m.cells.position[c]
        P = [Point2f(_pt(closest(xc, m.vertices.position[v], 1.0, 1.0))...) for v in m.cells.vertices[c]]
        for sx in -1:1, sy in -1:1
            Q = [p .+ Point2f(sx, sy) for p in P]
            xs = first.(Q); ys = last.(Q)
            (maximum(xs) < 0 || minimum(xs) > 1 || maximum(ys) < 0 || minimum(ys) > 1) && continue
            push!(polys, Q); push!(owner, c)
        end
    end
    return polys, owner
end
defects(m) = ([_pt(m.cells.position[c]) for c in 1:m.cells.n if length(m.cells.edges[c]) == 5],
              [_pt(m.cells.position[c]) for c in 1:m.cells.n if length(m.cells.edges[c]) == 7])
function mark_defects!(ax, m)
    p5, p7 = defects(m)
    isempty(p5) || scatter!(ax, first.(p5), last.(p5), marker = :diamond, color = INK, markersize = 9, strokecolor = :white, strokewidth = 1)
    isempty(p7) || scatter!(ax, first.(p7), last.(p7), marker = :star5, color = INK, markersize = 10, strokecolor = :white, strokewidth = 1)
end

# ============================================================================
# Fig A: where the lowest-order errors live (SCVT mesh4)
# ============================================================================
let m = VoronoiMesh("meshes/mesh4.nc")
    Hl = hodge_star_lift_matrix(m; only_linear = false)
    td = truncation(m, diag_star(m)); ts = truncation(m, symmetric_soft(m, Hl))
    ke = ke_error(m, [0.8, -0.35])
    polys, owner = cell_polys(m)
    @printf("  mesh4: max|tau| diag %.2e  symmetric %.2e   max|K error| %.2e\n", maximum(abs, td), maximum(abs, ts), maximum(abs, ke))
    f = Figure(size = (1200, 470))
    lim = maximum(abs, td)
    lk = maximum(abs, ke)
    # columns: (a) (b) [shared colour bar] (c) [colour bar]
    for (col, vals, ttl, l) in [(1, td, @sprintf("(a) Laplacian truncation, diagonal star\nmax %.1e", maximum(abs, td)), lim),
                                (2, ts, @sprintf("(b) same, symmetric star (P₁ exact + soft P₂)\nmax %.1e", maximum(abs, ts)), lim),
                                (4, ke, @sprintf("(c) local kinetic energy error, constant field\nmax %.1e (global error 0)", lk), lk)]
        ax = Axis(f[1, col], aspect = DataAspect(), title = ttl)
        hidedecorations!(ax); hidespines!(ax)
        poly!(ax, polys, color = vals[owner], colormap = DIV, colorrange = (-l, l), strokewidth = 0)
        mark_defects!(ax, m)
        xlims!(ax, 0, 1); ylims!(ax, 0, 1)
    end
    Colorbar(f[1, 3], colormap = DIV, limits = (-lim, lim), width = 10, ticklabelsize = 10)
    Colorbar(f[1, 5], colormap = DIV, limits = (-lk, lk), width = 10, ticklabelsize = 10)
    colgap!(f.layout, 8)
    Label(f[2, :], "SCVT mesh4. (a), (b): (1/|V_c|)(d₀ᵀHd₀p)_c + mean of Δp over V_c, relative to max|Δp|, for p = sin2πx sin2πy + 0.3cos2π(x−2y); " *
          "(a) and (b) share one colour scale.\n(c): TRiSK kinetic energy (1/4|V_c|)Σ αₑ(H₀α)ₑ for u = (0.8, −0.35), relative error.  " *
          "◆ pentagons, ★ heptagons.", fontsize = 11, color = MUTED, halign = :left, justification = :left, tellwidth = false)
    savefig("fig_peixoto_maps.png", f)
end

"maps" in ARGS && (println("maps only"); exit())

# ============================================================================
# Fig B: max-norm truncation of the Laplacian under refinement
# ============================================================================
let f = Figure(size = (1150, 420))
    fams = [("SCVT", ["mesh2", "mesh3", "mesh4", "mesh5"]), ("perturbed", ["pert2", "pert3", "pert4", "pert5"]),
            ("regular", ["mesh2_regular", "mesh3_regular", "mesh4_regular", "mesh5_regular"])]
    specs = [("diagonal (TRiSK)", MUTED, :circle, :solid), ("symmetric, P₁ exact", BLUE, :rect, :solid),
             ("symmetric, P₁ exact + soft P₂", ORANGE, :utriangle, :solid), ("lift P₂ (non-symmetric)", AQUA, :diamond, :dash)]
    for (j, (fam, mns)) in enumerate(fams)
        hs = Float64[]; R = [Float64[] for _ in specs]
        for mn in mns
            m = VoronoiMesh("meshes/$mn.nc"); push!(hs, meanh(m))
            Hl = hodge_star_lift_matrix(m; only_linear = false)
            Hs = [diag_star(m),
                  hodge_star_symmetric_matrix(m; only_linear = true, level = 3, consistency = 1).H +
                      polynomial_blind_stabilisation(m; only_linear = true, level = 3),
                  symmetric_soft(m, Hl), Hl]
            for (k, H) in enumerate(Hs); push!(R[k], maximum(abs, truncation(m, H))); end
            @printf("  %-14s %s\n", mn, join([@sprintf("%.2e", r[end]) for r in R], "  "))
        end
        ax = Axis(f[1, j], xscale = log10, yscale = log10, xlabel = "h", xreversed = true,
                  ylabel = j == 1 ? "max_c |τ_c| / max |Δp|" : "", title = fam)
        for (k, (lab, c, mk, ls)) in enumerate(specs)
            lines!(ax, hs, R[k], color = c, linewidth = 2.5, linestyle = ls, label = lab)
            scatter!(ax, hs, R[k], color = c, marker = mk, markersize = 10, strokecolor = :white, strokewidth = 1)
        end
        lines!(ax, hs, 0.5 .* R[2][1] .* (hs ./ hs[1]), color = MUTED, linestyle = :dot, linewidth = 1)
        lines!(ax, hs, 0.5 .* R[3][1] .* (hs ./ hs[1]).^2, color = MUTED, linestyle = :dash, linewidth = 1)
        j == 1 && axislegend(ax, position = :lb, labelsize = 10)
    end
    Label(f[2, :], "Pointwise (max-norm) truncation error of the DEC Laplacian.  Grey dotted: slope 1; grey dashed: slope 2.  " *
          "A flat curve means the operator is inconsistent in the max norm.", fontsize = 11, color = MUTED, halign = :left, justification = :left, tellwidth = false)
    savefig("fig_peixoto_truncation.png", f)
end
println("done")
