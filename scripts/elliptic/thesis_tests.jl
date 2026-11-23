# ============================================================================
#  thesis_tests.jl -- more Poisson and Darcy solutions for hodge_thesis.tex
#
#      julia -t 4 scripts/elliptic/thesis_tests.jl      ->  ts_data/thesis_tests.{txt,jls}
#        fig_th_elliptic_conv.png   convergence: Poisson (3 solutions) and Darcy (3 solutions), stars H_DEC, H_P1, H_P2
#        fig_th_poisson_tests.png   Poisson on mesh 4: exact, H_P2 solution, errors of the three stars, for the three solutions
#        fig_th_darcy_tests.png     Darcy on mesh 3: pressure and velocity (one arrow per triangle, at its circumcentre) for the
#                                   three solutions, exact / H_DEC / H_P2, and the pressure errors
#
#  Poisson  -Δp = f:  P1 smooth  p = sin2πx sin2πy + ½cos4πx;  P2 localised  p = bump(x0=(0.4,0.6), σ=0.06);
#                     P3 several scales  p = sin2πx cos4πy + 0.3 sin6π(x+y)
#  Darcy  v + ∇p = g, div v = φ (Hirani's mixed form, flux on Delaunay edges, pressure on triangles):
#           D1 p = cos2πx cos2πy, g = 0;   D2 the same p with a divergence-free body force g (rotational velocity);
#           D3 localised pressure p = bump(x0=(0.5,0.5), σ=0.10), g = 0
# ============================================================================
isdefined(Main, :cached_stars) || include(joinpath(@__DIR__, "..", "stars", "stars_cache.jl"))
isdefined(Main, :bump) || (bump(x; x0 = (0.5, 0.75), σ = 0.1) = exp((cos(TP * (x[1] - x0[1])) + cos(TP * (x[2] - x0[2])) - 2) / (TP^2 * σ^2)))
include(joinpath(@__DIR__, "..", "shared", "thesis_plot_helpers.jl"))
using Serialization, LinearAlgebra, SparseArrays
const OUTL = String[]
sayt(s) = (println(s); flush(stdout); push!(OUTL, s))
const STARS3 = ("H_DEC", "H_P1", "H_P2")
const TMESH = ("mesh2", "mesh3", "mesh4", "mesh5", "mesh6")

# ---------------------------------------------------------------------------- exact solutions
"bump and its gradient and Laplacian"
function bumpder(x; x0, σ)
    a = 1 / (TP^2 * σ^2); X = TP * (x[1] - x0[1]); Y = TP * (x[2] - x0[2]); b = exp(a * (cos(X) + cos(Y) - 2))
    g = b .* [-a * TP * sin(X), -a * TP * sin(Y)]
    lap = b * (a^2 * TP^2 * (sin(X)^2 + sin(Y)^2) - a * TP^2 * (cos(X) + cos(Y)))
    return b, g, lap
end
const PTESTS = (
    (name = "P1 smooth", p = x -> sin(TP * x[1]) * sin(TP * x[2]) + 0.5cos(2TP * x[1]),
     lap = x -> -2TP^2 * sin(TP * x[1]) * sin(TP * x[2]) - 2TP^2 * cos(2TP * x[1])),
    (name = "P2 localised", p = x -> bumpder(x; x0 = (0.4, 0.6), σ = 0.06)[1], lap = x -> bumpder(x; x0 = (0.4, 0.6), σ = 0.06)[3]),
    (name = "P3 several scales", p = x -> sin(TP * x[1]) * cos(2TP * x[2]) + 0.3sin(3TP * (x[1] + x[2])),
     lap = x -> -5TP^2 * sin(TP * x[1]) * cos(2TP * x[2]) - 0.3 * 18TP^2 * sin(3TP * (x[1] + x[2]))))
gsw(x) = TP .* [cos(TP * x[1]) * sin(TP * x[2]), -sin(TP * x[1]) * cos(TP * x[2])]     # divergence free, as large as ∇p
const DTESTS = (
    (name = "D1 potential flow", p = PD, v = VD, g = x -> [0.0, 0.0], φ = PHID),
    (name = "D2 with body force", p = PD, v = x -> VD(x) .+ gsw(x), g = gsw, φ = PHID),
    (name = "D3 localised", p = x -> bumpder(x; x0 = (0.5, 0.5), σ = 0.10)[1], v = x -> -bumpder(x; x0 = (0.5, 0.5), σ = 0.10)[2],
     g = x -> [0.0, 0.0], φ = x -> -bumpder(x; x0 = (0.5, 0.5), σ = 0.10)[3]))
supnorm(f) = maximum(norm(f([i / 200, j / 200])) for i in 0:199 for j in 0:199)

# ---------------------------------------------------------------------------- solve
RT = Dict{Any,Any}()
for mn in TMESH
    mesh = VoronoiMesh("meshes/$mn.nc"); S = cached_stars(mn; mesh); A = mesh.cells.area; h = sqrt(1 / mesh.cells.n); RT[(mn, :h)] = h
    sayt(@sprintf("%s (h = 1/%d)", mn, round(Int, 1 / h)))
    pts = [_pt(x) for x in mesh.cells.position]
    for (i, T) in enumerate(PTESTS)
        pe = [T.p(x) for x in pts]; pe .-= sum(A .* pe) / sum(A); line = "   " * rpad(T.name, 20)
        for s in STARS3
            ph = solve_poisson_primal(mesh, S[s], x -> -T.lap(x)); ph .-= sum(A .* ph) / sum(A); e = ph .- pe
            RT[(mn, :P, i, s)] = (inf = maximum(abs.(e)) / maximum(abs.(pe)), l2 = sqrt(sum(A .* e .^ 2) / sum(A .* pe .^ 2)))
            mn == "mesh4" && (RT[(mn, :Psol, i, s)] = ph; RT[(mn, :Pex, i)] = pe)
            line *= @sprintf("  %s max %.2e", s, RT[(mn, :P, i, s)].inf)
        end
        sayt(line)
    end
    tris, D1, _ = triangles_ccw(mesh); At = [T.area for T in tris]; lp = edge_lengths(mesh, true)
    for (i, T) in enumerate(DTESTS)
        ex = darcy_data(mesh, tris; v = T.v, p = T.p, g = T.g, φ = T.φ); vn = supnorm(T.v); line = "   " * rpad(T.name, 20)
        for s in STARS3
            fh, ph = solve_darcy(S[s], D1, ex.G, ex.Φ, At)
            d = darcy_errors(mesh, tris, fh, ph, ex; vnorm = vn)
            RT[(mn, :D, i, s)] = d
            mn == "mesh3" && (RT[(mn, :Dsol, i, s)] = (p = ph, V = triangle_velocity(mesh, tris, fh)))
            line *= @sprintf("  %s flux %.2e p %.2e", s, d.flux, d.p_inf)
        end
        mn == "mesh3" && (RT[(mn, :Dex, i)] = (p = ex.p, V = reduce(vcat, [T.v(Tr.cc)' for Tr in tris])))
        sayt(line)
    end
    serialize(joinpath(TSDIR, "thesis_tests.jls"), RT)
end
write(joinpath(TSDIR, "thesis_tests.txt"), join(OUTL, "\n") * "\n")

# ---------------------------------------------------------------------------- convergence
let f = Figure(size = (1500, 860), fontsize = 16), hs = [RT[(m, :h)] for m in TMESH]
    local ax
    for (i, T) in enumerate(PTESTS)
        y0 = RT[(TMESH[2], :P, i, "H_DEC")].inf
        ax = conv_axis(f[1, i], hs, "Poisson, " * T.name; ylabel = i == 1 ? "max error of p (relative)" : "", anchor = 0.5y0)
        for s in STARS3
            scatterlines!(ax, hs, [RT[(m, :P, i, s)].inf for m in TMESH]; color = STARCOL[s], marker = STARMK[s], markersize = 11, linewidth = 2.2, label = s)
        end
    end
    for (i, T) in enumerate(DTESTS)
        y0 = RT[(TMESH[2], :D, i, "H_DEC")].flux
        ax = conv_axis(f[2, i], hs, "Darcy, " * T.name; ylabel = i == 1 ? "flux error (relative, H_DEC norm)" : "", anchor = 0.5y0)
        for s in STARS3
            scatterlines!(ax, hs, [RT[(m, :D, i, s)].flux for m in TMESH]; color = STARCOL[s], marker = STARMK[s], markersize = 11, linewidth = 2.2, label = s)
        end
    end
    Legend(f[3, 1:3], ax; orientation = :horizontal, framevisible = false)
    CairoMakie.save(thesis_out("fig_th_elliptic_conv.png"), f; px_per_unit = 2)
end

# ---------------------------------------------------------------------------- Poisson solutions, mesh 4
let mesh = VoronoiMesh("meshes/mesh4.nc"), (polys, own) = periodic_copies(cellpolys(mesh))
    f = Figure(size = (1500, 960), fontsize = 15)
    for (i, T) in enumerate(PTESTS)
        pe = RT[("mesh4", :Pex, i)]; r = (minimum(pe), maximum(pe))
        map_panel(f[i, 1], polys, own, pe, "exact, " * T.name; range = r)
        map_panel(f[i, 2], polys, own, RT[("mesh4", :Psol, i, "H_P2")], "H_P2"; range = r)
        for (j, s) in enumerate(STARS3)
            err_panel(f[i, j+2], polys, own, RT[("mesh4", :Psol, i, s)] .- pe, "error " * s)
        end
    end
    Label(f[4, 3:5], "errors: each panel has its own symmetric colour range (blue < 0 < red)"; fontsize = 14, color = INK2)
    CairoMakie.save(thesis_out("fig_th_poisson_tests.png"), f; px_per_unit = 2)
end

# ---------------------------------------------------------------------------- Darcy solutions, mesh 3
let mesh = VoronoiMesh("meshes/mesh3.nc")
    tris, _, _ = triangles_ccw(mesh); At = [T.area for T in tris]; polys, own = periodic_copies(tripolys(tris)); cc = ccpoints(tris)
    shift(q) = q .- sum(At .* q) / sum(At)
    f = Figure(size = (1500, 960), fontsize = 15)
    for (i, T) in enumerate(DTESTS)
        E = RT[("mesh3", :Dex, i)]; pe = shift(E.p); r = (minimum(pe), maximum(pe)); vm = maximum(norm.(eachrow(E.V))); ls = 0.045 / vm
        map_panel(f[i, 1], polys, own, pe, "exact, " * T.name; range = r, pts = cc, dirs = vecs(E.V), ls = ls)
        for (j, s) in enumerate(("H_DEC", "H_P2"))
            D = RT[("mesh3", :Dsol, i, s)]
            map_panel(f[i, j+1], polys, own, shift(D.p), s; range = r, pts = cc, dirs = vecs(D.V), ls = ls)
            err_panel(f[i, j+3], polys, own, shift(D.p) .- pe, "pressure error " * s)
        end
    end
    CairoMakie.save(thesis_out("fig_th_darcy_tests.png"), f; px_per_unit = 2)
end
println("thesis tests written")
