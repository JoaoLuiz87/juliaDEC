# thesis_plot_helpers.jl -- shared plotting helpers for the thesis figures (include after the numerical libraries)
using CairoMakie, Printf
const INK, INK2, GRID = colorant"#0b0b0b", colorant"#52514e", colorant"#e4e3df"
const C_DEC, C_P1, C_P2 = colorant"#4d4d4d", colorant"#2a78d6", colorant"#eb6834"
const AQUA, YELLOW, VIOLET = colorant"#1baf7a", colorant"#eda100", colorant"#6250d6"
const DIVM = cgrad([colorant"#1c5cab", colorant"#f0efec", colorant"#c23b3a"])
const STARCOL = Dict("H_DEC" => C_DEC, "H_P1" => C_P1, "H_P2" => C_P2)
const STARMK = Dict("H_DEC" => :circle, "H_P1" => :diamond, "H_P2" => :rect)
thesis_out(n) = joinpath(@__DIR__, "..", "..", "output", "figures", n)
axstyle_th = (xgridcolor = GRID, ygridcolor = GRID, topspinevisible = false, rightspinevisible = false, titlealign = :left)

"Polygons crossing the periodic boundary are duplicated so that the unit square is covered. Returns (polys, owner)."
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
vecs(V) = [Vec2f(V[t, 1], V[t, 2]) for t in 1:size(V, 1)]

"One map panel: polygons coloured by v; optional arrows (one per polygon) at pts."
function map_panel(pos, polys, own, v, title; cmap = :viridis, range = nothing, pts = nothing, dirs = nothing, ls = 1.0, acol = :white,
                   titlesize = 15)
    ax = Axis(pos; aspect = DataAspect(), title = title, titlesize = titlesize, titlealign = :left); hidedecorations!(ax)
    r = range === nothing ? extrema(v) : range
    r[1] == r[2] && (r = (r[1] - 1e-300, r[2] + 1e-300))
    poly!(ax, polys; color = v[own], colormap = cmap, colorrange = r, strokewidth = 0)
    pts === nothing || arrows2d!(ax, pts, dirs; lengthscale = ls, color = acol, shaftwidth = 1.0, tipwidth = 4, tiplength = 4)
    xlims!(ax, 0, 1); ylims!(ax, 0, 1); return ax
end
"Error panel with a symmetric diverging range and the maximum in the title."
err_panel(pos, polys, own, e, name; titlesize = 15) =
    (m = max(maximum(abs, e), 1e-300); map_panel(pos, polys, own, e, @sprintf("%s, max %.1e", name, m); cmap = DIVM, range = (-m, m), titlesize = titlesize))

"Vector constant on each triangle matching its three edge values b_e = (x_head − x_tail)·w (least squares)."
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

"log-log convergence axis in h with reference slopes drawn from the second point."
function conv_axis(pos, hs, title; ylabel = "", slopes = (2, 4), anchor = nothing, xt = nothing)
    xt === nothing && (xt = (hs, ["1/" * string(round(Int, 1 / h)) for h in hs]))
    ax = Axis(pos; xscale = log10, yscale = log10, xreversed = true, xticks = xt, xlabel = "h", ylabel = ylabel, title = title, axstyle_th...)
    if anchor !== nothing
        for k in slopes
            xx = [hs[2], hs[end]]; yy = anchor .* (xx ./ xx[1]) .^ k
            lines!(ax, xx, yy; color = INK2, linestyle = :dot)
            text!(ax, xx[2], yy[2]; text = " h" * Dict(1 => "", 2 => "²", 3 => "³", 4 => "⁴")[k], align = (:left, :center), fontsize = 14, color = INK2)
        end
    end
    return ax
end
