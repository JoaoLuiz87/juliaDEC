# ============================================================================
#  gh_figures.jl -- figures of galerkin_hodge.tex
#
#      julia scripts/archive/gh_figures.jl
#
#  fig_gh_cells.png    : the two cell types and their stencils (triangle + 1-ring, diamond + 13 edges)
#  fig_gh_dual.png     : the dual object Φ_e of one edge, least squares versus moment-constrained (triangle cells, SCVT)
#  fig_gh_conv.png     : (a) Hodge error of a smooth field, (b) Poisson gradient error on SCVT meshes 2-5,
#                        (c) size of the correction on mapped meshes, exact versus regularised
#  Reads gh_data/main_results.txt and gh_data/reg_sweep.txt; computes DEC and the least-squares stars on mesh5.
# ============================================================================
include(joinpath(@__DIR__, "gh_lab.jl"))
using CairoMakie
const DATA = joinpath(@__DIR__, "..", "..", "output", "gh_data")

# ---------------------------------------------------------------- cells and stencils
let M = latmesh(8, "random"; amp = 0.04)
    fig = Figure(size = (900, 430))
    for (col, (title, C)) in enumerate((("triangle cell and its 1-ring stencil", tri_cells(M)[71]),
                                        ("diamond cell and its 13-edge stencil", dia_cells(M)[110])))
        ax = Axis(fig[1, col]; aspect = DataAspect(), title = title)
        hidedecorations!(ax); hidespines!(ax)
        for (a, b) in M.E
            xa = unwrap(M, M.X[a], C.xc); xb = unwrap(M, M.X[b], xa)
            norm((xa + xb) / 2 - C.xc) < 0.33 && lines!(ax, [xa[1], xb[1]], [xa[2], xb[2]]; color = (:gray, 0.4), linewidth = 0.8)
        end
        for (p1, p2, p3, sg) in C.regs
            poly!(ax, Point2f[Tuple(p1), Tuple(p2), Tuple(p3)]; color = (sg > 0 ? :steelblue : :steelblue, 0.35))
        end
        for (xa, xb) in C.segs
            lines!(ax, [xa[1], xb[1]], [xa[2], xb[2]]; color = :black, linewidth = 2.2)
        end
        if col == 2
            g = egeo(M, 110, M.X[M.E[110][1]]); sh = C.xc - g.m
            lines!(ax, [g.c1[1], g.c2[1]] .+ sh[1], [g.c1[2], g.c2[2]] .+ sh[2]; color = :firebrick, linewidth = 2.5)
        end
        r = 0.27; xlims!(ax, C.xc[1] - r, C.xc[1] + r); ylims!(ax, C.xc[2] - r, C.xc[2] + r)
    end
    save(joinpath(@__DIR__, "..", "..", "output", "figures", "fig_gh_cells.png"), fig; px_per_unit = 2)
end

# ---------------------------------------------------------------- the dual object of one edge
let M = readmesh(joinpath(DATA, "mesh3.txt"))
    geos = [egeo(M, e, M.X[M.E[e][1]]) for e in eachindex(M.E)]
    cells = tri_cells(M); D = celldata(M, cells, 1)
    Rc, _ = constrained_R(M, cells, D, geos, 1)
    e0 = argmin([norm(g.m - [0.5, 0.5]) for g in geos]); g0 = geos[e0]
    fig = Figure(size = (980, 470))
    for (col, (title, Rs)) in enumerate((("least squares R = P⁺", [d.R0 for d in D]), ("moment-constrained R = P⁺ + ZWᵀ", Rc)))
        ax = Axis(fig[1, col]; aspect = DataAspect(), title = title); hidedecorations!(ax); hidespines!(ax)
        pts = Point2f[]; vec = Vec2f[]; gsum = zeros(2)
        for (C, d, R) in zip(cells, D, Rs)
            q = findfirst(==(e0), C.S); q === nothing && continue
            off = unwrap(M, C.xc, g0.m) - C.xc                      # chart shift: put the cell next to e0
            (p1, p2, p3, _) = C.regs[1]
            for l in ((1 / 3, 1 / 3, 1 / 3), (0.6, 0.2, 0.2), (0.2, 0.6, 0.2), (0.2, 0.2, 0.6))
                x = l[1] * p1 + l[2] * p2 + l[3] * p3
                φ = basis(x, C.xc, M.h, 1) * R[:, q]
                push!(pts, Point2f(Tuple(x + off))); push!(vec, Vec2f(Tuple(φ)))
            end
            gsum .+= cellint(C, x -> basis(x, C.xc, M.h, 1) * R[:, q])
            poly!(ax, Point2f[Tuple(p1 + off), Tuple(p2 + off), Tuple(p3 + off)]; color = (:steelblue, 0.12), strokecolor = (:gray, 0.5), strokewidth = 0.6)
        end
        sc = 0.35 * M.h / maximum(norm, vec)
        arrows2d!(ax, pts, vec .* sc; color = :black, shaftwidth = 1.2, tipwidth = 7, tiplength = 7)
        lines!(ax, [g0.xa[1], g0.xb[1]], [g0.xa[2], g0.xb[2]]; color = :black, linewidth = 3)
        lines!(ax, [g0.c1[1], g0.c2[1]], [g0.c1[2], g0.c2[2]]; color = :firebrick, linewidth = 3)
        b = g0.Ld * g0.t
        text!(ax, g0.m[1] - 0.16, g0.m[2] - 0.205; fontsize = 13,
              text = @sprintf("∫Φₑ = (%.4f, %.4f)\n|ẽ| tₑ = (%.4f, %.4f)", gsum..., b...))
        r = 0.17; xlims!(ax, g0.m[1] - r, g0.m[1] + r); ylims!(ax, g0.m[2] - r - 0.04, g0.m[2] + r)
    end
    save(joinpath(@__DIR__, "..", "..", "output", "figures", "fig_gh_dual.png"), fig; px_per_unit = 2)
end

# ---------------------------------------------------------------- convergence and coefficient size
function readtab(f)
    rows = [split(l) for l in readlines(f)[2:end] if !isempty(strip(l))]
    rows
end
main = readtab(joinpath(DATA, "main_results.txt")); sweep = readtab(joinpath(DATA, "reg_sweep.txt"))
scvt(m) = startswith(m, "scvt")
# main: mesh h star r0 r1 r2 flux Poisson patch Λ λmin λmax moment dev
series = Dict{String,Vector{NTuple{3,Float64}}}()
for r in main
    scvt(r[1]) || continue
    push!(get!(series, r[3], NTuple{3,Float64}[]), (parse(Float64, r[2]), parse(Float64, r[7]), parse(Float64, r[8])))
end
# mesh5: DEC and least-squares stars computed here, constrained stars from the sweep (ε = 1e-13)
let M = readmesh(joinpath(DATA, "mesh5.txt"))
    geos = [egeo(M, e, M.X[M.E[e][1]]) for e in eachindex(M.E)]
    u(x) = [sin(2pi * x[2]) + 0.5cos(2pi * (x[1] + x[2])), cos(2pi * x[1])]
    p(x) = sin(2pi * x[1]) * cos(2pi * x[2])
    gp(x) = [2pi * cos(2pi * x[1]) * cos(2pi * x[2]), -2pi * sin(2pi * x[1]) * sin(2pi * x[2])]
    h = maximum(g.L for g in geos)
    H = spdiagm([g.Ld / g.L for g in geos]); push!(series["DEC"], (h, flux_err(M, geos, H, u), poisson(M, geos, H, p, gp)[1]))
    for (lbl, cells) in (("tri-pinv", tri_cells(M)), ("dia-pinv", dia_cells(M)))
        D = celldata(M, cells, 1); H = assemble_H(M, cells, D, [d.R0 for d in D])
        push!(series[lbl], (h, flux_err(M, geos, H, u), poisson(M, geos, H, p, gp)[1]))
    end
end
for r in sweep
    (r[1] == "scvt-mesh5" && r[4] == "1e-13") || continue
    push!(series[r[3] * "-cons"], (parse(Float64, r[2]), parse(Float64, r[10]), parse(Float64, r[11])))
end
style = Dict("DEC" => (:black, :circle, "DEC (diagonal)"), "dia-pinv" => (:darkorange, :utriangle, "diamond, least squares (juliaDEC sandwich)"),
             "tri-pinv" => (:goldenrod, :dtriangle, "triangle, least squares"), "dia-cons" => (:firebrick, :rect, "diamond, constrained"),
             "tri-cons" => (:steelblue, :diamond, "triangle, constrained"))
fig = Figure(size = (1250, 400))
ax1 = Axis(fig[1, 1]; xscale = log10, yscale = log10, xlabel = "h", ylabel = "relative error", title = "(a) Hodge error, smooth field (SCVT)")
ax2 = Axis(fig[1, 2]; xscale = log10, yscale = log10, xlabel = "h", title = "(b) Poisson gradient error (SCVT)")
for (lbl, (c, mk, name)) in style
    haskey(series, lbl) || continue
    s = sort(series[lbl])
    scatterlines!(ax1, first.(s), getindex.(s, 2); color = c, marker = mk, label = name)
    scatterlines!(ax2, first.(s), getindex.(s, 3); color = c, marker = mk)
end
hh = [0.02, 0.16]
for ax in (ax1, ax2)
    lines!(ax, hh, 1.2 .* hh .^ 2 ./ hh[2]^2 .* 0.35; color = :gray, linestyle = :dash)
    text!(ax, 0.05, 0.03; text = "h²", color = :gray)
end
# (c) correction size on mapped meshes: dev versus h for ε = 1e-13 (exact) and ε = 1e-8
ax3 = Axis(fig[1, 3]; xscale = log10, yscale = log10, xlabel = "h", ylabel = "max ‖Z W'‖_M / ‖P⁺‖_M", title = "(c) size of the correction (mapped)")
for (cell, c) in (("tri", :steelblue), ("dia", :firebrick)), (eps, ls) in (("1e-13", :solid), ("1e-06", :dash))
    s = sort([(parse(Float64, r[2]), parse(Float64, r[8])) for r in sweep if startswith(r[1], "mapped") && r[3] == cell && r[4] == eps])
    isempty(s) && continue
    scatterlines!(ax3, first.(s), last.(s); color = c, linestyle = ls, label = "$cell, ε = $eps")
end
axislegend(ax3; position = :rt, labelsize = 10)
Legend(fig[2, 1:2], ax1; orientation = :horizontal, nbanks = 2, labelsize = 11, framevisible = false)
save(joinpath(@__DIR__, "..", "..", "output", "figures", "fig_gh_conv.png"), fig; px_per_unit = 2)
println("figures written")
