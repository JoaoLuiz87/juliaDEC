# hh_figure.jl -- fig_hh_conv.png from hh_data/consistency.txt: DEC versus the symmetric, linear-exact Galerkin stars
using CairoMakie
rows = [split(l) for l in readlines(joinpath(@__DIR__, "..", "..", "output", "hh_data", "consistency.txt"))[2:end] if occursin(r"^mesh\d", l)]
# columns: 1 mesh, 2 h, 3 star, 4 asym, 5 row P1, 6 u1 max, 7 u1 rms, 8 u2 max, 9 u2 rms, 10 max C_e, 11 λmin
style = (("DEC", :black, :circle, "DEC (diagonal): not exact on linear fields"),
         ("Galerkin-tri", :steelblue, :diamond, "Galerkin, triangle cells: symmetric, exact on linear fields"),
         ("Galerkin-dia", :firebrick, :rect, "Galerkin, diamond cells: symmetric, exact on linear fields"))
fig = Figure(size = (1150, 500))
for (i, (k, ttl)) in enumerate(((6, "u₁ = (sin 2πx cos 2πy, cos 2πx sin 2πy)"), (8, "u₂ = (sin 2πx cos 2πy, -cos 2πx sin 2πy)")))
    ax = Axis(fig[1, i]; xscale = log10, yscale = log10, xlabel = "h = N^{-1/2}", ylabel = i == 1 ? "|ε_e| / |ẽ|" : "",
              title = ttl, xreversed = true)
    for (lbl, c, mk, name) in style
        rs = [r for r in rows if r[3] == lbl]
        h = parse.(Float64, getindex.(rs, 2))
        scatterlines!(ax, h, parse.(Float64, getindex.(rs, k)); color = c, marker = mk, label = name)
        scatterlines!(ax, h, parse.(Float64, getindex.(rs, k + 1)); color = c, marker = mk, linestyle = :dash)
    end
    # slope guides at the right end
    h0, h1 = 0.03125, 0.015625
    lines!(ax, [h0, h1], [4e-3, 1e-3]; color = :gray, linewidth = 2); text!(ax, 0.022, 2.4e-3; text = "h²", color = :gray, align = (:right, :top))
    lines!(ax, [h0, h1], [2.4e-2, 1.2e-2]; color = :gray, linewidth = 2, linestyle = :dot); text!(ax, 0.022, 2.0e-2; text = "h", color = :gray, align = (:right, :bottom))
end
Legend(fig[2, 1:2], fig.content[1]; orientation = :horizontal, nbanks = 3, framevisible = false, labelsize = 12)
Label(fig[3, 1:2], "solid: maximum over the edges   dashed: r.m.s.   juliaDEC SCVT meshes 2-5 (N = 64, 256, 1024, 4096)"; fontsize = 12)
save(joinpath(@__DIR__, "..", "..", "output", "figures", "fig_hh_conv.png"), fig; px_per_unit = 2)
