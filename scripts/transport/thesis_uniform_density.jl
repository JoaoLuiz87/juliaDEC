# thesis_uniform_density.jl -- a uniform density in the deformational flow: the only error is that of the swept area (Hv), so it
# isolates the star.   julia -t 4 scripts/transport/thesis_uniform_density.jl  ->  fig_th_uniform.png, ts_data/thesis_uniform.txt
include(joinpath(@__DIR__, "extrusion_oneform.jl"))
include(joinpath(@__DIR__, "..", "shared", "thesis_plot_helpers.jl"))
OUTU = String[]; sayu(s) = (println(s); flush(stdout); push!(OUTU, s))
let mn = "mesh5"
    G = lie_setup(mn; cfv = (), upw = ()); extrusion_setup!(G; p_rho = 3, rings = 2, constrained = true)
    mesh = G["mesh"]; A = G["A"]; h = G["h"]; polys, own = periodic_copies(cellpolys(mesh))
    test = TESTS[:deformational]; ρ0 = A .* 1.0
    f = Figure(size = (1500, 420), fontsize = 15)
    for (j, s) in enumerate(("H_DEC", "H_P2"))
        ρh = rk4(extrusion_rhs(G, s, test), ρ0, test.T / 2, h / 8) ./ A .- 1
        ρT = rk4(extrusion_rhs(G, s, test), ρ0, test.T, h / 8) ./ A .- 1
        err_panel(f[1, 2j-1], polys, own, ρh, "$s, t = T/2: ρ − 1"; titlesize = 14)
        err_panel(f[1, 2j], polys, own, ρT, "$s, t = T: ρ − 1"; titlesize = 14)
        sayu(@sprintf("uniform density, deformational flow, %s, %s: max|ρ − 1| at T/2 %.3e, at T %.3e", mn, s, maximum(abs, ρh), maximum(abs, ρT)))
    end
    Label(f[2, 1:4], "a uniform density stays uniform only if the swept areas Hv have zero discrete divergence; each panel has its own colour range";
          fontsize = 14, color = INK2)
    CairoMakie.save(thesis_out("fig_th_uniform.png"), f; px_per_unit = 2)
end
write(joinpath(TSDIR, "thesis_uniform.txt"), join(OUTU, "\n") * "\n")
