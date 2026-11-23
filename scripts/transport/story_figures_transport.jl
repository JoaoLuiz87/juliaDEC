# story_figures_transport.jl -- density transport with H_DEC, H_P1, H_P2 (from ts_data/story_transport.jls)  ->  fig_st_transport.png
#      julia scripts/transport/story_figures_transport.jl
include(joinpath(@__DIR__, "..", "stars", "stars_cache.jl"))
include(joinpath(@__DIR__, "..", "shared", "thesis_plot_helpers.jl"))
using Serialization
RT = deserialize(joinpath(TSDIR, "story_transport.jls"))
const S3 = ("H_DEC", "H_P1", "H_P2")
MS = [m for m in ("mesh4", "mesh5", "mesh6", "mesh7") if haskey(RT, (m, :uniform, "H_P2"))]
let mesh = VoronoiMesh("meshes/mesh6.nc"), (polys, own) = periodic_copies(cellpolys(mesh))
    f = Figure(size = (1500, 1180), fontsize = 15)
    for (i, (test, lab)) in enumerate(((:uniform, "uniform density, ρ − 1 at T"), (:deformational, "two hills, error at T")))
        E = Dict(s => RT[("mesh6", test, s, :final)] .- RT[("mesh6", test, :exact)] for s in S3); m = maximum(maximum(abs, E[s]) for s in S3)
        for (j, s) in enumerate(S3)
            map_panel(f[i, j], polys, own, E[s], @sprintf("%s, %s, max %.1e", lab, s, maximum(abs, E[s])); cmap = DIVM, range = (-m, m), titlesize = 14)
        end
        Colorbar(f[i, 4]; colormap = DIVM, limits = (-m, m), width = 12, tickformat = v -> [@sprintf("%.0e", x) for x in v])
    end
    hs = [RT[(m, :h)] for m in MS]; local ax
    g = f[3, 1:4] = GridLayout()
    for (j, (test, lab)) in enumerate(((:uniform, "uniform density"), (:deformational, "two hills, deformational"), (:divergent, "two hills, divergent")))
        ax = conv_axis(g[1, j], hs, lab; ylabel = j == 1 ? "relative max error at T" : "", anchor = 0.5 * RT[(MS[min(2, end)], test, "H_DEC")].linf, slopes = (2, 4))
        for s in S3
            scatterlines!(ax, hs, [RT[(m, test, s)].linf for m in MS]; color = STARCOL[s], marker = STARMK[s], markersize = 11, linewidth = 2.2, label = s)
        end
    end
    axislegend(ax; position = :lb, framevisible = false)
    rowsize!(f.layout, 3, Relative(0.32))
    CairoMakie.save(thesis_out("fig_st_transport.png"), f; px_per_unit = 2)
end
for m in MS, t in (:uniform, :deformational, :divergent)
    println(@sprintf("%s %-13s ", m, t) * join([@sprintf("%s %.2e  ", s, RT[(m, t, s)].linf) for s in S3]))
end
