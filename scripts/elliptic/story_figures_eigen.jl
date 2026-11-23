# story_figures_eigen.jl -- Laplace eigenvalues for H_DEC, H_P1, H_P2 (from ts_data/story_eigen.jls)  ->  fig_st_eigen.png
#      julia scripts/elliptic/story_figures_eigen.jl
include(joinpath(@__DIR__, "..", "stars", "stars_cache.jl"))
include(joinpath(@__DIR__, "..", "shared", "thesis_plot_helpers.jl"))
using Serialization
RE = deserialize(joinpath(TSDIR, "story_eigen.jls"))
const S3 = ("H_DEC", "H_P1", "H_P2")
MS = ("mesh2", "mesh3", "mesh4", "mesh5")
let f = Figure(size = (1500, 470), fontsize = 15)
    ax1 = Axis(f[1, 1]; yscale = log10, xlabel = "index j", ylabel = "|λ_j,h − λ_j| / λ_j", title = "(a) mesh 5, corrected M0", axstyle_th...)
    for s in S3
        R = RE[("mesh5", s, :C)]
        scatterlines!(ax1, 1:length(R.err), max.(R.err, 1e-12); color = STARCOL[s], marker = STARMK[s], markersize = 6, linewidth = 1.5, label = s)
    end
    axislegend(ax1; position = :rb, framevisible = false)
    hs = [RE[(m, "H_DEC", :C)].h for m in MS]
    ax2 = conv_axis(f[1, 2], hs, "(b) max error of λ_1, …, λ_20"; ylabel = "relative error", slopes = (2, 4), anchor = 0.5 * maximum(RE[(MS[2], "H_DEC", :C)].err[1:20]))
    for s in S3, (mk, ls) in ((:C, :solid), (:A, :dash))
        scatterlines!(ax2, hs, [maximum(RE[(m, s, mk)].err[1:min(20, length(RE[(m, s, mk)].err))]) for m in MS]; color = STARCOL[s], marker = STARMK[s],
                      linestyle = ls, markersize = 10, linewidth = 2, label = "$s, " * (mk == :C ? "corrected M0" : "M0 = A"))
    end
    axislegend(ax2; position = :lb, framevisible = false, labelsize = 12)
    ax3 = Axis(f[1, 3]; xscale = log10, yscale = log10, xreversed = true, xlabel = "h", ylabel = "max |Im λ| / λ_max",
               title = "(c) complex eigenvalues of H_P2", axstyle_th..., xticks = (hs, ["1/" * string(round(Int, 1 / h)) for h in hs]))
    for (mk, ls) in ((:C, :solid), (:A, :dash))
        scatterlines!(ax3, hs, [max(RE[(m, "H_P2", mk)].imax, 1e-17) for m in MS]; color = C_P2, marker = :rect, linestyle = ls, markersize = 10,
                      linewidth = 2, label = mk == :C ? "corrected M0" : "M0 = A")
    end
    axislegend(ax3; position = :rb, framevisible = false)
    CairoMakie.save(thesis_out("fig_st_eigen.png"), f; px_per_unit = 2)
end
for m in MS, s in S3, mk in (:A, :C)
    R = RE[(m, s, mk)]
    println(@sprintf("%s %-5s %s  max err λ1..20 %.2e   λ1..100 %.2e   imax %.1e   low-mode max |Im|/Re %.1e", m, s, mk, maximum(R.err[1:min(20, end)]), maximum(R.err), R.imax, maximum(R.im ./ R.re)))
end
