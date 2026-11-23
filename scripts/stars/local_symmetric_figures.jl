# ============================================================================
#  local_symmetric_figures.jl -- figures of hodge_properties_note.tex, Sections "How far local and symmetric go" and
#  "Symmetry" (spectrum)
#
#      julia scripts/stars/local_symmetric_figures.jl
#          -> output/figures/fig_hp_sweeps.png     (a) symmetric iterates: max Hodge error against the number of sweeps
#                                          (b) exact iterates: asymmetry against the number of sweeps
#             output/figures/fig_hp_spectrum.png   non-real eigenvalues of the Laplacian against h: size and share
#  Data: ts_data/local_symmetric.jls, p2_min_asym.jls, p2_swept_spectrum.jls, p2_spectrum.jls
# ============================================================================
isdefined(Main, :cached_stars) || include(joinpath(@__DIR__, "stars_cache.jl"))
using CairoMakie, Printf, Serialization
const TS = joinpath(@__DIR__, "..", "..", "output", "ts_data")
L = deserialize(joinpath(TS, "local_symmetric.jls")); A = deserialize(joinpath(TS, "p2_min_asym.jls"))
W = deserialize(joinpath(TS, "p2_swept_spectrum.jls")); P = deserialize(joinpath(TS, "p2_spectrum.jls"))
# colours: one hue per family (blue: exact on P1, orange: exact on P2), a light-to-dark ramp for the ordered meshes
const INK, INK2, GRID = colorant"#0b0b0b", colorant"#52514e", colorant"#e4e3df"
const BLUE, ORANGE, AQUA, YELLOW = colorant"#2a78d6", colorant"#eb6834", colorant"#1baf7a", colorant"#eda100"
const RAMP = Dict("mesh4" => colorant"#86b6ef", "mesh5" => colorant"#3987e5", "mesh6" => colorant"#104281")
xs(k) = k == 0 ? -1.0 : log2(k)
axstyle = (xgridcolor = GRID, ygridcolor = GRID, topspinevisible = false, rightspinevisible = false, titlealign = :left, titlesize = 21,
           xlabelcolor = INK2, ylabelcolor = INK2, xticklabelcolor = INK2, yticklabelcolor = INK2, titlecolor = INK)

# ---------------------------------------------------------------------------- Figure 1
f = Figure(size = (1180, 500), fontsize = 20, backgroundcolor = :white)
ks = sort(unique(k for (m, t, k) in keys(L) if t == :ap))
ax = Axis(f[1, 1]; yscale = log10, yticks = ([1, 3, 10, 30, 100], ["1", "3", "10", "30", "100"]), xticks = (xs.([0, 1, 4, 16, 64, 256]), string.([0, 1, 4, 16, 64, 256])),
          xlabel = "sweeps k  (0 = H_DEC)", ylabel = "max error ÷ that of H_P1", title = "(a) symmetric after k sweeps", axstyle...)
hlines!(ax, [1.0]; color = INK2, linewidth = 1.5, linestyle = :dash)
text!(ax, xs(0), 1.0; text = "H_P1", align = (:left, :bottom), offset = (0, 3), color = INK2, fontsize = 18)
for mn in ("mesh4", "mesh5", "mesh6")
    mesh = VoronoiMesh("meshes/$mn.nc"); ref = hodge_error(mesh, cached_stars(mn; mesh)["H_P1"], fields.d, supn.d).inf
    y = [L[(mn, :ap, k)].rinf / ref for k in ks]
    scatterlines!(ax, xs.(ks), y; color = RAMP[mn], linewidth = 2.5, markersize = 10, label = "h = 1/$(round(Int, sqrt(mesh.cells.n)))")
end
xlims!(ax, -1.4, xs(512) + 0.6)
axislegend(ax; position = :rt, framevisible = false, labelsize = 18)

ax2 = Axis(f[1, 2]; yscale = log10, yticks = ([1e-4, 1e-3, 1e-2], ["0.01%", "0.1%", "1%"]), xticks = (xs.([0, 1, 4, 16, 64, 256, 1024]), string.([0, 1, 4, 16, 64, 256, 1024])),
           xlabel = "sweeps s  (0 = row-by-row star)", ylabel = "asymmetry  ‖H − Hᵀ‖ ÷ ‖H‖", title = "(b) exact after s sweeps", axstyle...)
for (k, col, lab, mk) in ((2, ORANGE, "exact on [P2]²", :circle), (1, BLUE, "exact on [P1]²", :rect))
    for (mn, ls, lw) in (("mesh5", :solid, 2.0), ("mesh4", :dot, 1.5))
        its = sort([it for (m, kk, it) in keys(A) if m == mn && kk == k])
        y = [A[(mn, k, it)].asymF for it in its]
        scatterlines!(ax2, xs.(its), y; color = col, linewidth = mn == "mesh5" ? 2.5 : 1.8, linestyle = ls, markersize = mn == "mesh5" ? 10 : 6, marker = mk,
                      label = mn == "mesh5" ? lab : nothing)
        mn == "mesh5" && text!(ax2, xs(its[end]), y[end]; text = k == 2 ? " P2" : " P1", align = (:left, :center), color = INK, fontsize = 18)
    end
end
xlims!(ax2, -1.4, xs(1024) + 1.5)
axislegend(ax2, "h = 1/64  (dotted: 1/32)"; position = :lb, framevisible = false, labelsize = 18, titlesize = 17)
CairoMakie.save(joinpath(@__DIR__, "..", "..", "output", "figures", "fig_hp_sweeps.png"), f; px_per_unit = 2)

# ---------------------------------------------------------------------------- Figure 2
MS = ("mesh2", "mesh3", "mesh4", "mesh5", "mesh6"); lev = Dict(m => i for (i, m) in enumerate(MS))
swept(mn, k, it) = haskey(W, (mn, k, it)) ? W[(mn, k, it)] : (haskey(A, (mn, k, it)) && haskey(A[(mn, k, it)], :im) ? A[(mn, k, it)] : nothing)
series = (("H_P2", ORANGE, :circle, mn -> haskey(P, (mn, "H_P2")) ? P[(mn, "H_P2")] : nothing),
          ("H_P2 after 1 sweep", AQUA, :utriangle, mn -> swept(mn, 2, 1)),
          ("H_P2 after 4 sweeps", YELLOW, :diamond, mn -> swept(mn, 2, 4)),
          ("local star exact on [P1]²", BLUE, :rect, mn -> swept(mn, 1, 0)))
g = Figure(size = (1180, 480), fontsize = 20, backgroundcolor = :white)
xt = (1:5, ["1/8", "1/16", "1/32", "1/64", "1/128"])
a1 = Axis(g[1, 1]; yscale = log10, yticks = ([1e-5, 1e-4, 1e-3], ["10⁻⁵", "10⁻⁴", "10⁻³"]), xticks = xt, xlabel = "mesh size h", ylabel = "max |Im λ| ÷ λmax",
          title = "(a) size of the non-real part", axstyle...)
a2 = Axis(g[1, 2]; xticks = xt, xlabel = "mesh size h", ylabel = "non-real eigenvalues (%)", title = "(b) share of non-real eigenvalues", axstyle...)
for (lab, col, mk, get) in series
    pts = [(lev[mn], get(mn)) for mn in MS if get(mn) !== nothing]
    x1 = [p[1] for p in pts if p[2].im > 0]; y1 = [p[2].im for p in pts if p[2].im > 0]
    scatterlines!(a1, x1, y1; color = col, marker = mk, linewidth = 2.5, markersize = 12, label = lab)
    scatterlines!(a2, [p[1] for p in pts], [100p[2].frac for p in pts]; color = col, marker = mk, linewidth = 2.5, markersize = 12, label = lab)
    isempty(x1) || (lab == "local star exact on [P1]²" ?
        text!(a1, x1[1], y1[1]; text = "P1 local  ", align = (:right, :center), color = INK, fontsize = 18) :
        text!(a1, x1[end], y1[end]; text = "  " * (lab == "H_P2" ? "H_P2" : replace(lab, "H_P2 after " => "")), align = (:left, :center), color = INK, fontsize = 18))
end
xlims!(a1, 0.7, 6.9); xlims!(a2, 0.7, 5.4)
axislegend(a2; position = :lt, framevisible = false, labelsize = 17)
CairoMakie.save(joinpath(@__DIR__, "..", "..", "output", "figures", "fig_hp_spectrum.png"), g; px_per_unit = 2)
println("wrote fig_hp_sweeps.png, fig_hp_spectrum.png")
