# ============================================================================
#  thesis_figures.jl -- solution plots and transport convergence for hodge_thesis.tex
#
#      julia -t 4 scripts/elliptic/thesis_figures.jl
#        -> fig_th_darcy.png        Darcy on mesh 3: pressure on the triangles, ONE velocity arrow per triangle at its
#                                   circumcentre (velocity recovered from the three edge fluxes, as in Hirani et al.)
#           fig_th_density.png      transport of a density (deformational flow, mesh 5): t = 0, T/2, T and the errors at T
#           fig_th_oneform.png      transport of a 1-form (deformational flow, mesh 4): one arrow per triangle at its
#                                   circumcentre (field recovered from the three edge circulations), colour = vorticity
#           fig_th_transport.png    convergence of the transport tests (density and 1-form)
# ============================================================================
include(joinpath(@__DIR__, "..", "transport", "extrusion_oneform.jl"))
using CairoMakie, Printf, Serialization
const INK, INK2, GRID = colorant"#0b0b0b", colorant"#52514e", colorant"#e4e3df"
const BLUE, ORANGE, AQUA, YELLOW, GRAY = colorant"#2a78d6", colorant"#eb6834", colorant"#1baf7a", colorant"#eda100", colorant"#6b6b6b"
const DIVM = cgrad([colorant"#1c5cab", colorant"#f0efec", colorant"#c23b3a"])
outp(n) = joinpath(@__DIR__, n)
function periodic_copies(polys)
    out = Vector{Vector{Point2f}}(); own = Int[]
    for (i, P) in enumerate(polys)
        push!(out, P); push!(own, i)
        any(p -> p[1] < 0 || p[1] > 1 || p[2] < 0 || p[2] > 1, P) || continue
        for sx in -1:1, sy in -1:1
            (sx, sy) == (0, 0) && continue
            push!(out, [p .+ Point2f(sx, sy) for p in P]); push!(own, i)
        end
    end
    return out, own
end
cellpolys(mesh) = [Point2f[(p[1], p[2]) for p in get_cells_points(mesh, c)] for c in 1:mesh.cells.n]
tripolys(tris) = [Point2f[(p[1], p[2]) for p in T.pts] for T in tris]
ccpoints(tris) = [Point2f(mod(T.cc[1], 1.0), mod(T.cc[2], 1.0)) for T in tris]
function panel(pos, polys, own, v, title; cmap = :viridis, range = nothing, pts = nothing, dirs = nothing, ls = 1.0, acol = :white)
    ax = Axis(pos; aspect = DataAspect(), title = title, titlesize = 17, titlealign = :left); hidedecorations!(ax)
    r = range === nothing ? extrema(v) : range
    poly!(ax, polys; color = v[own], colormap = cmap, colorrange = r, strokewidth = 0)
    pts === nothing || arrows2d!(ax, pts, dirs; lengthscale = ls, color = acol, shaftwidth = 1.0, tipwidth = 4, tiplength = 4)
    xlims!(ax, 0, 1); ylims!(ax, 0, 1); return ax
end
"Vector constant on each triangle from the three edge circulations b_e = (x_head − x_tail)·w."
function triangle_vector(mesh, tris, b)
    V = zeros(length(tris), 2)
    for (t, T) in enumerate(tris)
        M = zeros(3, 2); r = zeros(3)
        for (k, e) in enumerate(T.edges)
            q1, q2 = edge_endpoints(mesh, e, true; about = T.ccv); M[k, :] = q2 .- q1; r[k] = b[e]
        end
        V[t, :] = M \ r
    end
    return V
end
vecs(V) = [Vec2f(V[t, 1], V[t, 2]) for t in 1:size(V, 1)]

# ---------------------------------------------------------------------------- Darcy, mesh 3
let mn = "mesh3"
    mesh = VoronoiMesh("meshes/$mn.nc"); S = cached_stars(mn; mesh); tris, D1, _ = triangles_ccw(mesh); At = [T.area for T in tris]
    ex = darcy_smooth(mesh, tris); polys, own = periodic_copies(tripolys(tris)); cc = ccpoints(tris)
    Vex = reduce(vcat, [VD(T.cc)' for T in tris]); shift(q) = q .- sum(At .* q) / sum(At)
    f = Figure(size = (1500, 520), fontsize = 16)
    panel(f[1, 1], polys, own, shift(ex.p), "exact p and v"; range = (-1, 1), pts = cc, dirs = vecs(Vex), ls = 0.006)
    for (j, s) in enumerate(("H_DEC", "H_P2"))
        fh, ph = solve_darcy(S[s], D1, ex.G, ex.Φ, At); V = triangle_velocity(mesh, tris, fh)
        ef = maximum(abs.(fh .- ex.f) ./ edge_lengths(mesh, true)) / TP
        panel(f[1, j+1], polys, own, shift(ph), @sprintf("%s (max flux error %.1e)", s, ef); range = (-1, 1), pts = cc, dirs = vecs(V), ls = 0.006)
    end
    Colorbar(f[1, 4]; colormap = :viridis, limits = (-1, 1), width = 12, label = "pressure")
    CairoMakie.save(outp("fig_th_darcy.png"), f; px_per_unit = 2)
end

# ---------------------------------------------------------------------------- density transport, mesh 5
let mn = "mesh5", test = TESTS.deformational
    G = lie_setup(mn; cfv = (), upw = ()); extrusion_setup!(G; p_rho = 3, rings = 2, constrained = true)
    mesh = G["mesh"]; A = G["A"]; h = G["h"]; polys, own = periodic_copies(cellpolys(mesh))
    ρ0 = cell_averages(mesh, test.ρ0) .* A; ex = exact_at_T(mesh, test); rhs = extrusion_rhs(G, "H_P2", test)
    ρh = rk4(rhs, ρ0, test.T / 2, h / 8) ./ A; ρT = rk4(rhs, ρ0, test.T, h / 8) ./ A
    G2 = lie_setup(mn; cfv = (), upw = ()); extrusion_setup!(G2); ρ2 = rk4(extrusion_rhs(G2, "H_P2", test), ρ0, test.T, h / 8) ./ A
    f = Figure(size = (1500, 360), fontsize = 15)
    for (j, (v, t)) in enumerate(((ρ0 ./ A, "t = 0"), (ρh, "t = T/2"), (ρT, "t = T (cubic)")))
        panel(f[1, j], polys, own, v, t; range = (0, 1))
    end
    Colorbar(f[1, 4]; colormap = :viridis, limits = (0, 1), width = 12, label = "density")
    for (j, (v, t)) in enumerate(((ρ2 .- ex, "error at T, quadratic"), (ρT .- ex, "error at T, cubic")))
        m = maximum(abs, v)
        panel(f[1, j+4], polys, own, v, @sprintf("%s: max %.1e", t, m); cmap = DIVM, range = (-m, m))
    end
    CairoMakie.save(outp("fig_th_density.png"), f; px_per_unit = 2)
end

# ---------------------------------------------------------------------------- 1-form transport, mesh 4
let mn = "mesh4", test = TESTS.deformational
    G = lie_setup(mn; cfv = (), upw = ()); extrusion_setup!(G); oneform_setup!(G)
    mesh = G["mesh"]; h = G["h"]; tris, _, _ = triangles_ccw(mesh); polys, own = periodic_copies(tripolys(tris)); cc = ccpoints(tris)
    β0 = x -> test.ρ0(x) .* [1.0, 0.5]; b0 = edge_integrals(mesh, β0); rhs = oneform_rhs(G, test)
    bh = rk4(rhs, b0, test.T / 2, h / 8); bT = rk4(rhs, b0, test.T, h / 8)
    ζ(b) = G["σT"] .* (G["D1p"] * b) ./ G["AT"]; zr = maximum(abs, ζ(b0))
    f = Figure(size = (1500, 560), fontsize = 16)
    for (j, (b, t)) in enumerate(((b0, "t = 0 (also the exact field at T)"), (bh, "t = T/2"), (bT, "t = T")))
        panel(f[1, j], polys, own, ζ(b), t; cmap = DIVM, range = (-zr, zr), pts = cc, dirs = vecs(triangle_vector(mesh, tris, b)), ls = 0.028, acol = INK)
    end
    Colorbar(f[1, 4]; colormap = DIVM, limits = (-zr, zr), width = 12, label = "vorticity D₁b / |T|")
    CairoMakie.save(outp("fig_th_oneform.png"), f; px_per_unit = 2)
end

# ---------------------------------------------------------------------------- transport convergence
let X = deserialize(joinpath(TSDIR, "extrusion.jls")), K = deserialize(joinpath(TSDIR, "extrusion_cubic.jls")), O = deserialize(joinpath(TSDIR, "extrusion_oneform.jls"))
    f = Figure(size = (1500, 520), fontsize = 17)
    series = (("density, quadratic, H_P2", BLUE, :circle, (m, t) -> haskey(X, (m, t, "H_P2", :ext)) ? X[(m, t, "H_P2", :ext)].linf : NaN),
              ("density, cubic, H_DEC", GRAY, :xcross, (m, t) -> haskey(K, (m, t, "H_DEC", "EXT3")) ? K[(m, t, "H_DEC", "EXT3")].linf : NaN),
              ("density, cubic, H_P2", ORANGE, :rect, (m, t) -> haskey(K, (m, t, "H_P2", "EXT3")) ? K[(m, t, "H_P2", "EXT3")].linf : NaN),
              ("1-form b", AQUA, :utriangle, (m, t) -> haskey(O, (m, t, "rec", :oneform)) ? O[(m, t, "rec", :oneform)].linf : NaN),
              ("1-form, vorticity", YELLOW, :diamond, (m, t) -> haskey(O, (m, t, "rec", :oneform)) ? O[(m, t, "rec", :oneform)].zlinf : NaN))
    ms = ("mesh3", "mesh4", "mesh5", "mesh6", "mesh7"); hs = [1 / 16, 1 / 32, 1 / 64, 1 / 128, 1 / 256]
    local ax
    for (j, t) in enumerate(("translation", "deformational", "cellular"))
        ax = Axis(f[1, j]; xscale = log10, yscale = log10, xreversed = true, xticks = (hs, ["1/16", "1/32", "1/64", "1/128", "1/256"]), xlabel = "h",
                  ylabel = j == 1 ? "relative max error at T" : "", title = t, titlealign = :left, xgridcolor = GRID, ygridcolor = GRID)
        for (lab, col, mk, get) in series
            y = [get(m, t) for m in ms]; ok = .!isnan.(y)
            scatterlines!(ax, hs[ok], y[ok]; color = col, marker = mk, linewidth = 2.5, markersize = 12, label = lab)
        end
        for (k, c0) in ((3, 2e-1), (4, 2e-2))
            lines!(ax, [hs[2], hs[end]], c0 .* ([hs[2], hs[end]] ./ hs[2]) .^ k; color = INK2, linestyle = :dot)
            text!(ax, hs[end], c0 * (hs[end] / hs[2])^k; text = " h" * (k == 3 ? "³" : "⁴"), align = (:left, :center), fontsize = 15, color = INK2)
        end
    end
    Legend(f[2, 1:3], ax; orientation = :horizontal, framevisible = false)
    CairoMakie.save(outp("fig_th_transport.png"), f; px_per_unit = 2)
end
println("thesis figures written")
