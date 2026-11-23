# gs_p2_figures.jl -- fig_p2_conv.png from gs_data/p2_conv_table.txt: DEC, the degree-1 and the degree-2 stars
#      julia scripts/archive/gs_p2_figures.jl
using CairoMakie
rows = [split(l) for l in readlines(joinpath(@__DIR__, "..", "..", "output", "gs_data", "p2_conv_table.txt")) if !startswith(l, "#")]
# columns: 1 mesh, 2 h, 3 star, 4 rowP1, 5 rowP2, 6 corr, 7 Cmax, 8 mu, 9 u1max, 10 u1rms, 11 u2max, 12 u2rms, 13 pnodal, 14 flux
style = (("DEC", :black, :circle, "DEC (diagonal)"),
         ("degree1", :steelblue, :diamond, "degree 1: [P1]² reconstruction, exact on linear fields"),
         ("degree2", :firebrick, :rect, "degree 2: [P2]² reconstruction, quadratic moments in least squares"))
fig = Figure(size = (1500, 520))
panels = ((9, "Hodge error, u₁ (gradient field)", "|ε_e| / |ẽ|"), (11, "Hodge error, u₂ (divergence-free field)", ""),
          (13, "Poisson, nodal error (relative max)", ""))
for (i, (k, ttl, yl)) in enumerate(panels)
    ax = Axis(fig[1, i]; xscale = log10, yscale = log10, xlabel = "h = N^{-1/2}", ylabel = yl, title = ttl, xreversed = true)
    for (lbl, c, mk, _) in style
        rs = [r for r in rows if r[3] == lbl]; h = parse.(Float64, getindex.(rs, 2))
        scatterlines!(ax, h, parse.(Float64, getindex.(rs, k)); color = c, marker = mk)
        k < 13 && scatterlines!(ax, h, parse.(Float64, getindex.(rs, k + 1)); color = c, marker = mk, linestyle = :dash)
    end
    h0, h1 = 0.0625, 0.015625
    lines!(ax, [h0, h1], [8e-3, 8e-3 / 16]; color = :gray, linewidth = 2); text!(ax, 0.03, 1.5e-3; text = "h²", color = :gray)
    k < 13 && (lines!(ax, [h0, h1], [6e-2, 6e-2 / 4]; color = :gray, linewidth = 2, linestyle = :dot);
               text!(ax, 0.03, 3.6e-2; text = "h", color = :gray))
end
Legend(fig[2, 1:3], [MarkerElement(color = c, marker = mk) for (_, c, mk, _) in style], [n for (_, _, _, n) in style];
       orientation = :horizontal, framevisible = false, labelsize = 13)
Label(fig[3, 1:3], "solid: maximum over the edges   dashed: r.m.s.   juliaDEC SCVT meshes 2-5 (N = 64, 256, 1024, 4096)"; fontsize = 12)
CairoMakie.save(joinpath(@__DIR__, "..", "..", "output", "figures", "fig_p2_conv.png"), fig; px_per_unit = 2)
