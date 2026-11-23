# ============================================================================
#  homogeneous_figures.jl -- figure of homogeneous_note.tex (after homogeneous_skew_check.jl and homogeneous_p2_check.jl)
#
#      julia scripts/barrier/homogeneous_figures.jl          ->  output/figures/fig_homog.png
#
#  (a) the 5- and 7-valent generators of SCVT meshes 4-6; (b) |s_h| against h on four families; (c) the smallest symmetric
#  star exact on [P2]^2 on the replicated family and on the SCVT meshes: consistency error and Λ.
# ============================================================================
include(joinpath(@__DIR__, "..", "stars", "three_stars.jl"))
using CairoMakie
const HS = deserialize(joinpath(TSDIR, "homogeneous_skew.jls"))
const HP = deserialize(joinpath(TSDIR, "homogeneous_p2.jls"))
const SK = deserialize(joinpath(TSDIR, "skew_split.jls"))
const PS = deserialize(joinpath(TSDIR, "p2_symmetric.jls"))
const BLUE, RED, GREEN, GRAY, ORANGE = colorant"#2b6cb0", colorant"#c23b3a", colorant"#2f8f4e", colorant"#6b6b6b", colorant"#d08a1c"
sup(k) = Dict(1 => "", 2 => "²", 3 => "³", 4 => "⁴")[k]

f = Figure(size = (1500, 980))
# (a) defects
for (i, k) in enumerate((4, 5, 6))
    D = HS[("defects", k)]
    ax = Axis(f[1, i]; aspect = DataAspect(), title = "($(Char('a' + i - 1))) SCVT mesh $k: generators with 5 (blue) and 7 (red) neighbours",
              titlesize = 13); hidedecorations!(ax)
    for (val, col) in ((5, BLUE), (7, RED))
        P = [p for (p, v) in zip(D.pos, D.nb) if v == val]
        scatter!(ax, [p[1] for p in P], [p[2] for p in P]; color = col, markersize = 8)
    end
    lines!(ax, [0, 1, 1, 0, 0], [0, 0, 1, 1, 0]; color = :black, linewidth = 0.8)
    xlims!(ax, -0.02, 1.02); ylims!(ax, -0.02, 1.02)
end
# (b) skew against h
ax = Axis(f[2, 1]; xscale = log10, yscale = log10, xreversed = true, xticks = ([1/4, 1/8, 1/16, 1/32, 1/64, 1/128], ["1/4", "1/8", "1/16", "1/32", "1/64", "1/128"]), xlabel = "h", ylabel = "|s_h(u, v)|", titlesize = 13,
          title = "(d) the skew on four mesh families", yticks = (10.0 .^ (-16:2:0), ["10⁻¹⁶", "10⁻¹⁴", "10⁻¹²", "10⁻¹⁰", "10⁻⁸", "10⁻⁶", "10⁻⁴", "10⁻²", "1"]))
sc = sort([(SK[k].h, abs(SK[k].ud)) for k in keys(SK) if k[1] == "SCVT"], by = first, rev = true)
scatterlines!(ax, first.(sc), last.(sc); color = RED, marker = :utriangle, label = "SCVT meshes, (u₂, d)")
bc = [(HS[("bicrystal", n)].h, abs(HS[("bicrystal", n)].vals[1])) for n in (4, 8, 16, 32, 64)]
scatterlines!(ax, first.(bc), last.(bc); color = ORANGE, marker = :diamond, label = "bicrystal, (u₂, d)")
rp = [(HS[("mesh2", n, 1)].h, abs(HS[("mesh2", n, 1)].s)) for n in (1, 2, 4, 8, 16)]
scatterlines!(ax, first.(rp), last.(rp); color = BLUE, marker = :circle, label = "replicated SCVT mesh 2, plane wave k₁")
rp3 = [(HS[("mesh3", n, 1)].h, abs(HS[("mesh3", n, 1)].s)) for n in (1, 2, 4, 8)]
scatterlines!(ax, first.(rp3), last.(rp3); color = BLUE, marker = :rect, linestyle = :dash, label = "replicated SCVT mesh 3, plane wave k₁")
for (k, y0, x0) in ((2, 3e-3, 0.12), (3, 1.5e-7, 0.12))
    xx = [x0, x0 / 8]; lines!(ax, xx, y0 .* (xx ./ x0) .^ k; color = :black, linestyle = :dot)
    text!(ax, xx[2], y0 * (xx[2] / x0)^k; text = "  h" * sup(k), fontsize = 13, align = (:left, :center))
end
axislegend(ax; position = :lb, labelsize = 11, framevisible = false)
# (c) symmetric [P2]^2 star: replicated family against SCVT
ax2 = Axis(f[2, 2]; xscale = log10, yscale = log10, xreversed = true, xticks = ([1/4, 1/8, 1/16, 1/32, 1/64, 1/128], ["1/4", "1/8", "1/16", "1/32", "1/64", "1/128"]), xlabel = "h", ylabel = "r₂(d)", titlesize = 13,
           title = "(e) symmetric star exact on [P₂]² (4-ring): consistency error",
           yticks = (10.0 .^ (-7:1:-2), ["10⁻⁷", "10⁻⁶", "10⁻⁵", "10⁻⁴", "10⁻³", "10⁻²"]))
ax3 = Axis(f[2, 3]; xscale = log10, yscale = log10, xreversed = true, xticks = ([1/4, 1/8, 1/16, 1/32, 1/64, 1/128], ["1/4", "1/8", "1/16", "1/32", "1/64", "1/128"]), xlabel = "h", ylabel = "Λ", titlesize = 13,
           title = "(f) its size Λ", yticks = ([1, 2, 5, 10, 20], ["1", "2", "5", "10", "20"]))
rep = [(HP[(n, "sym P2 (4-ring)")].h, HP[(n, "sym P2 (4-ring)")].d2, HP[(n, "sym P2 (4-ring)")].Λ) for n in (1, 2, 4, 8)]
scv = [(PS[(m, :h)], PS[(m, 4, :sym)].hodge, PS[(m, 4, :sym)].Λ) for m in ("mesh2", "mesh3", "mesh4", "mesh5")]
p1r = [(HP[(n, "H_P1")].h, HP[(n, "H_P1")].d2) for n in (1, 2, 4, 8)]
for (A, lab, col, mk) in ((rep, "replicated SCVT mesh 2", BLUE, :circle), (scv, "SCVT meshes 2–5", RED, :utriangle))
    scatterlines!(ax2, [a[1] for a in A], [a[2] for a in A]; color = col, marker = mk, label = lab)
    scatterlines!(ax3, [a[1] for a in A], [a[3] for a in A]; color = col, marker = mk, label = lab)
end
scatterlines!(ax2, first.(p1r), last.(p1r); color = GRAY, marker = :diamond, linestyle = :dash, label = "H_P1, replicated (for reference)")
for (k, y0) in ((2, 6e-3), (3, 1.2e-3))
    xx = [0.125, 0.0156]; lines!(ax2, xx, y0 .* (xx ./ 0.125) .^ k; color = :black, linestyle = :dot)
    text!(ax2, xx[2], y0 * (xx[2] / 0.125)^k; text = "  h" * sup(k), fontsize = 13, align = (:left, :center))
end
axislegend(ax2; position = :lb, labelsize = 11, framevisible = false)
axislegend(ax3; position = :lt, labelsize = 11, framevisible = false)
rowsize!(f.layout, 1, Relative(0.48))
CairoMakie.save(joinpath(@__DIR__, "..", "..", "output", "figures", "fig_homog.png"), f; px_per_unit = 2)
