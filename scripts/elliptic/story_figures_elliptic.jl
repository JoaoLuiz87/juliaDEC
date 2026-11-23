# story_figures_elliptic.jl -- Poisson and Darcy errors of H_DEC, H_P1, H_P2 on ONE colour scale per row (from ts_data/thesis_tests.jls)
#      julia scripts/elliptic/story_figures_elliptic.jl  ->  fig_st_poisson.png, fig_st_darcy.png
include(joinpath(@__DIR__, "..", "stars", "stars_cache.jl"))
include(joinpath(@__DIR__, "..", "shared", "thesis_plot_helpers.jl"))
using Serialization
RT = deserialize(joinpath(TSDIR, "thesis_tests.jls"))
const S3 = ("H_DEC", "H_P1", "H_P2")
VMAG = cgrad([colorant"#f0efec", colorant"#e9b97a", colorant"#c46a1f", colorant"#5a2a0a"])

let mesh = VoronoiMesh("meshes/mesh4.nc"), (polys, own) = periodic_copies(cellpolys(mesh))
    names = ("P1 smooth", "P2 localised", "P3 several scales")
    f = Figure(size = (1500, 1100), fontsize = 15)
    for i in 1:3
        pe = RT[("mesh4", :Pex, i)]
        map_panel(f[i, 1], polys, own, pe, "exact, " * names[i]; range = extrema(pe))
        E = Dict(s => RT[("mesh4", :Psol, i, s)] .- pe for s in S3); m = maximum(maximum(abs, E[s]) for s in S3)
        for (j, s) in enumerate(S3)
            map_panel(f[i, 1+j], polys, own, E[s], @sprintf("error %s, max %.1e", s, maximum(abs, E[s])); cmap = DIVM, range = (-m, m))
        end
        Colorbar(f[i, 5]; colormap = DIVM, limits = (-m, m), width = 12, tickformat = v -> [@sprintf("%.0e", x) for x in v])
    end
    Label(f[4, 1:5], "Poisson on mesh 4 (h = 1/32): the three errors of a row share one colour scale"; fontsize = 14, color = INK2)
    CairoMakie.save(thesis_out("fig_st_poisson.png"), f; px_per_unit = 2)
end

"bump, its gradient and Laplacian (as in thesis_tests.jl)"
function bumpder(x; x0, σ)
    a = 1 / (TP^2 * σ^2); X = TP * (x[1] - x0[1]); Y = TP * (x[2] - x0[2]); b = exp(a * (cos(X) + cos(Y) - 2))
    return b, b .* [-a * TP * sin(X), -a * TP * sin(Y)], b * (a^2 * TP^2 * (sin(X)^2 + sin(Y)^2) - a * TP^2 * (cos(X) + cos(Y)))
end
gsw(x) = TP .* [cos(TP * x[1]) * sin(TP * x[2]), -sin(TP * x[1]) * cos(TP * x[2])]
const DT3 = ((p = PD, v = VD, g = x -> [0.0, 0.0], φ = PHID), (p = PD, v = x -> VD(x) .+ gsw(x), g = gsw, φ = PHID),
             (p = x -> bumpder(x; x0 = (0.5, 0.5), σ = 0.10)[1], v = x -> -bumpder(x; x0 = (0.5, 0.5), σ = 0.10)[2], g = x -> [0.0, 0.0],
              φ = x -> -bumpder(x; x0 = (0.5, 0.5), σ = 0.10)[3]))
let mesh = VoronoiMesh("meshes/mesh3.nc"), S = cached_stars("mesh3"; mesh = VoronoiMesh("meshes/mesh3.nc"))
    tris, D1, _ = triangles_ccw(mesh); At = [T.area for T in tris]; polys, own = periodic_copies(tripolys(tris)); cc = ccpoints(tris)
    shift(q) = q .- sum(At .* q) / sum(At)
    names = ("D1 potential flow", "D2 with body force", "D3 localised")
    f = Figure(size = (1700, 820), fontsize = 14)
    for i in 1:3
        local E = RT[("mesh3", :Dex, i)]; pe = shift(E.p); vm = maximum(norm.(eachrow(E.V)))
        ex = darcy_data(mesh, tris; v = DT3[i].v, p = DT3[i].p, g = DT3[i].g, φ = DT3[i].φ)
        Vref = triangle_velocity(mesh, tris, ex.f)                       # the same reconstruction applied to the exact fluxes
        map_panel(f[i, 1], polys, own, pe, "exact, " * names[i]; range = extrema(pe), pts = cc, dirs = vecs(E.V), ls = 0.045 / vm, titlesize = 13)
        ep = Dict(s => shift(RT[("mesh3", :Dsol, i, s)].p) .- pe for s in S3); mp = maximum(maximum(abs, ep[s]) for s in S3)
        sol = Dict(s => solve_darcy(S[s], D1, ex.G, ex.Φ, At) for s in S3)
        ev = Dict(s => norm.(eachrow(triangle_velocity(mesh, tris, sol[s][1]) .- Vref)) for s in S3); mv = maximum(maximum(ev[s]) for s in S3)
        for (j, s) in enumerate(S3)
            map_panel(f[i, 1+j], polys, own, ep[s], @sprintf("p error %s, %.1e", s, maximum(abs, ep[s])); cmap = DIVM, range = (-mp, mp), titlesize = 13)
            map_panel(f[i, 5+j], polys, own, ev[s], @sprintf("u error %s, %.1e", s, maximum(ev[s])); cmap = VMAG, range = (0, mv), titlesize = 13)
        end
        Colorbar(f[i, 5]; colormap = DIVM, limits = (-mp, mp), width = 10, ticklabelsize = 11, tickformat = v -> [@sprintf("%.0e", x) for x in v])
        Colorbar(f[i, 9]; colormap = VMAG, limits = (0, mv), width = 10, ticklabelsize = 11, tickformat = v -> [@sprintf("%.0e", x) for x in v])
    end
    Label(f[4, 1:9], "Darcy on mesh 3 (h = 1/16): pressure errors (columns 2-4) and velocity errors per triangle, against the same reconstruction of the exact fluxes (columns 6-8); each group of three shares the colour bar on its right";
          fontsize = 14, color = INK2)
    CairoMakie.save(thesis_out("fig_st_darcy.png"), f; px_per_unit = 2)
end
println("elliptic figures written")
