# ss_figures_conv.jl -- fig_ss_conv.png from ss_data/{abstol,objective,aniso}.txt (included by ss_figures.jl; runs alone too)
using CairoMakie, Printf
let OUT = joinpath(@__DIR__, "..", "..", "output", "ss_data")
    rd(f) = [split(l) for l in readlines(joinpath(OUT, f))[2:end] if !isempty(strip(l))]
    abst = rd("abstol.txt"); obj = rd("objective.txt")
    # abstol.txt: 3 E, 4 abstol, 11 flux max, 14 flux H⁻¹;  objective.txt DEC rows: 7 flux max, 10 flux H⁻¹
    fig = Figure(size = (1500, 430))
    panels = (("lattice", 11, 7, "flux consistency, porting lattice refined", "max_e |(Hα(u) - β(u))_e| / |ẽ|"),
              ("scvt", 11, 7, "flux consistency, juliaDEC SCVT 3-5", ""),
              ("scvt", 14, 10, "Poisson flux error ‖H d₀p_h - β‖_{H⁻¹} (rel.), SCVT", ""))
    for (i, (fam, ka, kd, title, yl)) in enumerate(panels)
        ax = Axis(fig[1, i]; xscale = log10, yscale = log10, xlabel = "E (edges)", ylabel = yl, title = title)
        for (at, col, lbl) in (("0e+00", :firebrick, "exact star (guide)"), ("1e-03", :steelblue, "uniform shares, symmetrised (not exact)"))
            rows = [r for r in abst if r[1] == fam && r[4] == at]
            scatterlines!(ax, parse.(Float64, getindex.(rows, 3)), parse.(Float64, getindex.(rows, ka)); color = col, label = lbl)
        end
        Es = Set(r[3] for r in abst if r[1] == fam); rows = [r for r in obj if r[1] == fam && r[4] == "DEC" && r[3] in Es]
        scatterlines!(ax, parse.(Float64, getindex.(rows, 3)), parse.(Float64, getindex.(rows, kd)); color = :black, label = "diagonal star (DEC)")
        i == 1 && axislegend(ax; position = :lb, labelsize = 10)
    end
    if isfile(joinpath(OUT, "aniso.txt"))
        an = rd("aniso.txt")
        ax = Axis(fig[1, 4]; xscale = log10, yscale = log10, xlabel = "E (edges)", title = "anisotropic K: Darcy flux error (H⁻¹), SCVT")
        for (key, col, lbl) in (("DEC_K", :black, "diagonal K-star"), ("exact", :firebrick, "exact K-star"), ("uniform", :steelblue, "uniform shares"))
            rows = [r for r in an if r[1] == "scvt" && r[4] == key]
            isempty(rows) && continue
            k0 = key == "DEC_K" ? 4 : 5
            scatterlines!(ax, parse.(Float64, getindex.(rows, 3)), parse.(Float64, getindex.(rows, k0 + 6)); color = col, label = lbl)
        end
        axislegend(ax; position = :lb, labelsize = 10)
    end
    save(joinpath(@__DIR__, "..", "..", "output", "figures", "fig_ss_conv.png"), fig; px_per_unit = 2)
end
