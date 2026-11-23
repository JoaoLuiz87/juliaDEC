# ============================================================================
#  thesis_transport.jl -- transport solutions and convergence for hodge_thesis.tex
#
#      julia -t 4 scripts/transport/thesis_transport.jl
#        fig_th_flows.png            the five flows on mesh 3 at t = 0: one arrow per triangle at its circumcentre (vector recovered
#                                    from the circulations v on the three edges), colour = speed
#        fig_th_density_tests.png    densities, mesh 5, cubic extrusion with H_P2: t = 0, T/2, T, exact at T, error at T (5 tests)
#        fig_th_oneform_tests.png    1-forms, mesh 4: arrows (field recovered from the circulations) over the vorticity, at
#                                    t = 0, T/2, T, exact at T, and the vorticity error at T (3 tests)
#        fig_th_transport_all.png    convergence of all five tests: standard DEC/TRiSK, quadratic and cubic extrusion, 1-form
# ============================================================================
include(joinpath(@__DIR__, "extrusion_oneform.jl"))
include(joinpath(@__DIR__, "..", "shared", "thesis_plot_helpers.jl"))
using Serialization
const TNAMES = ("translation", "cellular", "deformational", "divergent", "cosine_bells")
const TTITLE = Dict("translation" => "translation", "cellular" => "cellular (exact by characteristics)", "deformational" => "deformational",
                    "divergent" => "divergent", "cosine_bells" => "cosine bells")

# ---------------------------------------------------------------------------- the flows
let mesh = VoronoiMesh("meshes/mesh3.nc")
    tris, _, _ = triangles_ccw(mesh); polys, own = periodic_copies(tripolys(tris)); cc = ccpoints(tris)
    f = Figure(size = (1500, 360), fontsize = 15)
    for (j, name) in enumerate(TNAMES)
        test = TESTS[Symbol(name)]
        v = sum(c .* reduce_primal_edges(mesh, fld).values for (c, fld) in zip(test.coefs(0.0, test.T), test.fields))
        V = triangle_vector(mesh, tris, v); sp = norm.(eachrow(V)); vm = maximum(sp)
        map_panel(f[1, j], polys, own, sp, TTITLE[name] * ", t = 0"; cmap = cgrad([colorant"#f0efec", colorant"#86b6ef", colorant"#104281"]),
                  range = (0, vm), pts = cc, dirs = vecs(V), ls = 0.05 / vm, acol = INK, titlesize = 14)
    end
    CairoMakie.save(thesis_out("fig_th_flows.png"), f; px_per_unit = 2)
end

# ---------------------------------------------------------------------------- densities, mesh 5, cubic extrusion
let mn = "mesh5"
    G = lie_setup(mn; cfv = (), upw = ()); extrusion_setup!(G; p_rho = 3, rings = 2, constrained = true)
    mesh = G["mesh"]; A = G["A"]; h = G["h"]; polys, own = periodic_copies(cellpolys(mesh))
    f = Figure(size = (1500, 1560), fontsize = 15)
    for (i, name) in enumerate(TNAMES)
        test = TESTS[Symbol(name)]
        ρ0 = cell_averages(mesh, test.ρ0) .* A; ex = exact_at_T(mesh, test); rhs = extrusion_rhs(G, "H_P2", test)
        ρh = rk4(rhs, ρ0, test.T / 2, h / 8) ./ A; ρT = rk4(rhs, ρ0, test.T, h / 8) ./ A
        r = extrema(ρ0 ./ A)
        map_panel(f[i, 1], polys, own, ρ0 ./ A, TTITLE[name] * ": t = 0"; range = r, titlesize = 14)
        map_panel(f[i, 2], polys, own, ρh, "t = T/2"; range = r, titlesize = 14)
        map_panel(f[i, 3], polys, own, ρT, "t = T, computed"; range = r, titlesize = 14)
        map_panel(f[i, 4], polys, own, ex, "t = T, exact"; range = r, titlesize = 14)
        err_panel(f[i, 5], polys, own, ρT .- ex, "error"; titlesize = 14)
        @printf("density %s: max error %.2e, min value %.2e\n", name, maximum(abs.(ρT .- ex)), minimum(ρT))
    end
    CairoMakie.save(thesis_out("fig_th_density_tests.png"), f; px_per_unit = 1.6)
end

# ---------------------------------------------------------------------------- 1-forms, mesh 4
let mn = "mesh4"
    G = lie_setup(mn; cfv = (), upw = ()); extrusion_setup!(G); oneform_setup!(G)
    mesh = G["mesh"]; h = G["h"]; tris, _, _ = triangles_ccw(mesh); polys, own = periodic_copies(tripolys(tris)); cc = ccpoints(tris)
    ζ(b) = G["σT"] .* (G["D1p"] * b) ./ G["AT"]
    f = Figure(size = (1500, 960), fontsize = 15)
    for (i, name) in enumerate(("translation", "cellular", "deformational"))
        test = TESTS[Symbol(name)]; β0 = x -> test.ρ0(x) .* [1.0, 0.5]
        b0 = edge_integrals(mesh, β0); rhs = oneform_rhs(G, test)
        bh = rk4(rhs, b0, test.T / 2, h / 8); bT = rk4(rhs, b0, test.T, h / 8)
        bex = test.exact == :initial ? b0 : exact_oneform(mesh, test, β0, test.T)
        zr = maximum(abs, ζ(b0)); Vm = maximum(norm.(eachrow(triangle_vector(mesh, tris, b0)))); ls = 0.035 / Vm
        for (j, (b, t)) in enumerate(((b0, TTITLE[name] * ": t = 0"), (bh, "t = T/2"), (bT, "t = T, computed"), (bex, "t = T, exact")))
            map_panel(f[i, j], polys, own, ζ(b), t; cmap = DIVM, range = (-zr, zr), pts = cc, dirs = vecs(triangle_vector(mesh, tris, b)),
                      ls = ls, acol = INK, titlesize = 14)
        end
        err_panel(f[i, 5], polys, own, ζ(bT) .- ζ(bex), "vorticity error"; titlesize = 14)
        @printf("1-form %s: max error of b/|e| %.2e\n", name, maximum(abs.((bT .- bex) ./ G["le"])) / maximum(abs.(bex ./ G["le"])))
    end
    Label(f[4, 1:4], "colour: vorticity D₁b/|T| (blue < 0 < red);  arrows: the vector recovered on each triangle from its three edge circulations, at the circumcentre";
          fontsize = 14, color = INK2)
    CairoMakie.save(thesis_out("fig_th_oneform_tests.png"), f; px_per_unit = 2)
end

# ---------------------------------------------------------------------------- convergence, all tests
let C = deserialize(joinpath(TSDIR, "lie_testcases.jls")), X = deserialize(joinpath(TSDIR, "extrusion.jls")),
    K = deserialize(joinpath(TSDIR, "extrusion_cubic.jls")), O = deserialize(joinpath(TSDIR, "extrusion_oneform.jls"))
    ms = ("mesh3", "mesh4", "mesh5", "mesh6", "mesh7"); hs = [1 / 16, 1 / 32, 1 / 64, 1 / 128, 1 / 256]
    get(D, k) = haskey(D, k) ? D[k] : nothing
    series = (("standard DEC (H_DEC, two-cell average)", C_DEC, :circle, (m, t) -> get(C, (m, t, "H_DEC", "TRiSK"))),
              ("density, quadratic extrusion, H_P2", C_P1, :diamond, (m, t) -> get(X, (m, t, "H_P2", :ext))),
              ("density, cubic extrusion, H_DEC", VIOLET, :xcross, (m, t) -> get(K, (m, t, "H_DEC", "EXT3"))),
              ("density, cubic extrusion, H_P2", C_P2, :rect, (m, t) -> get(K, (m, t, "H_P2", "EXT3"))),
              ("1-form b, extrusion", AQUA, :utriangle, (m, t) -> get(O, (m, t, "rec", :oneform))))
    f = Figure(size = (1500, 900), fontsize = 16); local ax
    for (j, t) in enumerate(TNAMES)
        r, c = (j - 1) ÷ 3 + 1, (j - 1) % 3 + 1
        ax = conv_axis(f[r, c], hs, TTITLE[t]; ylabel = c == 1 ? "relative max error at T" : "", anchor = 0.25, slopes = (2, 4))
        for (lab, col, mk, g) in series
            y = [(q = g(m, t); q === nothing ? NaN : q.linf) for m in ms]; ok = .!isnan.(y)
            any(ok) && scatterlines!(ax, hs[ok], y[ok]; color = col, marker = mk, markersize = 11, linewidth = 2.2, label = lab)
        end
    end
    Legend(f[2, 3], ax; framevisible = false, labelsize = 15)
    CairoMakie.save(thesis_out("fig_th_transport_all.png"), f; px_per_unit = 2)
end
println("transport figures written")
