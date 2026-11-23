# ============================================================================
#  thesis_both_stars.jl -- solution figures with H_DEC and H_P2 side by side, for hodge_thesis.tex
#
#      julia -t 4 scripts/elliptic/thesis_both_stars.jl
#        fig_th_poisson_tests.png   Poisson, mesh 4: exact, H_DEC and H_P2 solutions, their errors   (from ts_data/thesis_tests.jls)
#        fig_th_density_tests.png   densities, mesh 5, cubic extrusion: exact at T, H_DEC and H_P2 at T, their errors (common range per row)
#        fig_th_sw.png              linear shallow water, mesh 4, t = 1, corrected 0-form star for both: exact, H_DEC, H_P2, errors
#      ts_data/thesis_both_stars.txt
# ============================================================================
include(joinpath(@__DIR__, "..", "transport", "extrusion_oneform.jl"))
include(joinpath(@__DIR__, "..", "shared", "thesis_plot_helpers.jl"))
using Serialization
OUTB = String[]; sayb(s) = (println(s); flush(stdout); push!(OUTB, s))

# ---------------------------------------------------------------------------- Poisson
let RT = deserialize(joinpath(TSDIR, "thesis_tests.jls")), mesh = VoronoiMesh("meshes/mesh4.nc"), (polys, own) = periodic_copies(cellpolys(mesh))
    names = ("P1 smooth", "P2 localised", "P3 several scales")
    f = Figure(size = (1500, 960), fontsize = 15)
    for i in 1:3
        pe = RT[("mesh4", :Pex, i)]; r = (minimum(pe), maximum(pe))
        map_panel(f[i, 1], polys, own, pe, "exact, " * names[i]; range = r)
        for (j, s) in enumerate(("H_DEC", "H_P2"))
            map_panel(f[i, 1+j], polys, own, RT[("mesh4", :Psol, i, s)], s; range = r)
            err_panel(f[i, 3+j], polys, own, RT[("mesh4", :Psol, i, s)] .- pe, "error " * s)
            sayb(@sprintf("Poisson %s, mesh 4, %s: max error %.2e", names[i], s, maximum(abs, RT[("mesh4", :Psol, i, s)] .- pe)))
        end
    end
    Label(f[4, 4:5], "errors: each panel has its own symmetric colour range (blue < 0 < red)"; fontsize = 14, color = INK2)
    CairoMakie.save(thesis_out("fig_th_poisson_tests.png"), f; px_per_unit = 2)
end

# ---------------------------------------------------------------------------- densities, mesh 5, cubic extrusion, both stars
const TN = ("translation", "cellular", "deformational", "divergent", "cosine_bells")
const TT = Dict("translation" => "translation", "cellular" => "cellular", "deformational" => "deformational", "divergent" => "divergent",
                "cosine_bells" => "cosine bells")
let mn = "mesh5"
    G = lie_setup(mn; cfv = (), upw = ()); extrusion_setup!(G; p_rho = 3, rings = 2, constrained = true)
    mesh = G["mesh"]; A = G["A"]; h = G["h"]; polys, own = periodic_copies(cellpolys(mesh))
    f = Figure(size = (1500, 1560), fontsize = 15)
    for (i, name) in enumerate(TN)
        test = TESTS[Symbol(name)]
        ρ0 = cell_averages(mesh, test.ρ0) .* A; ex = exact_at_T(mesh, test); r = extrema(ρ0 ./ A)
        sol = Dict(s => rk4(extrusion_rhs(G, s, test), ρ0, test.T, h / 8) ./ A for s in ("H_DEC", "H_P2"))
        m = maximum(maximum(abs, sol[s] .- ex) for s in ("H_DEC", "H_P2"))
        map_panel(f[i, 1], polys, own, ex, TT[name] * ": exact at T"; range = r, titlesize = 14)
        for (j, s) in enumerate(("H_DEC", "H_P2"))
            map_panel(f[i, 1+j], polys, own, sol[s], s * " at T"; range = r, titlesize = 14)
            e = sol[s] .- ex
            map_panel(f[i, 3+j], polys, own, e, @sprintf("error %s, max %.1e", s, maximum(abs, e)); cmap = DIVM, range = (-m, m), titlesize = 14)
            sayb(@sprintf("density %-13s %s: max error %.3e (relative %.3e), min value %.3e", name, s, maximum(abs, e), maximum(abs, e) / maximum(abs, ex), minimum(sol[s])))
        end
    end
    Label(f[6, 1:5], "cubic extrusion (mean-preserving, two rings), RK4 with Δt = h/8, mesh 5 (h = 1/64); the two error panels of a row share one colour range";
          fontsize = 14, color = INK2)
    CairoMakie.save(thesis_out("fig_th_density_tests.png"), f; px_per_unit = 1.6)
end

# ---------------------------------------------------------------------------- linear shallow water, both stars with the corrected 0-form star
ENV["SW_MAIN"] = "0"; include(joinpath(@__DIR__, "..", "shallow_water", "sw_linear_check.jl"))
let mn = "mesh4"
    mesh = VoronoiMesh("meshes/$mn.nc"); nc = mesh.cells.n; S = cached_stars(mn; mesh)
    hf, uf = wave(mesh); pts = [_pt(p) for p in mesh.cells.position]
    x0 = Vector{Float64}(vcat(hf.(pts, 0.0), reduce_primal_edges(mesh, x -> uf(x, 0.0)).values))
    hT = hf.(pts, 1.0); vT = Vector{Float64}(reduce_primal_edges(mesh, x -> uf(x, 1.0)).values)
    d0 = 1.0 * d0_matrix_primal(mesh); F0c = lu(mass0_corrected(mesh))
    sol = Dict{String,Any}()
    for s in ("H_DEC", "H_P2")
        Hs = S[s]; fc(x) = vcat(DEPTH .* (F0c \ (d0' * (Hs * x[nc+1:end]))), -GRAV .* (d0 * x[1:nc]))
        sol[s] = rk4(fc, x0, 1e-3, 1000)
        sayb(@sprintf("linear SW mesh 4, t = 1, corrected M0, %s: max error of h %.3e, of v %.3e", s, maximum(abs, sol[s][1:nc] .- hT),
                      maximum(abs, sol[s][nc+1:end] .- vT)))
    end
    tris, _, _ = triangles_ccw(mesh); cc = ccpoints(tris); polys, own = periodic_copies(cellpolys(mesh))
    r = (minimum(hT), maximum(hT)); VT = triangle_vector(mesh, tris, vT); ls = 0.03 / maximum(norm.(eachrow(VT)))
    f = Figure(size = (1500, 380), fontsize = 15)
    map_panel(f[1, 1], polys, own, hT, "exact h and u at t = 1"; range = r, pts = cc, dirs = vecs(VT), ls = ls)
    for (j, s) in enumerate(("H_DEC", "H_P2"))
        map_panel(f[1, 1+j], polys, own, sol[s][1:nc], s * ", corrected M0"; range = r, pts = cc,
                  dirs = vecs(triangle_vector(mesh, tris, sol[s][nc+1:end])), ls = ls)
        err_panel(f[1, 3+j], polys, own, sol[s][1:nc] .- hT, "error of h, " * s)
    end
    CairoMakie.save(thesis_out("fig_th_sw.png"), f; px_per_unit = 2)
end
write(joinpath(TSDIR, "thesis_both_stars.txt"), join(OUTB, "\n") * "\n")
println("both-star figures written")
