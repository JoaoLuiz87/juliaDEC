# ============================================================================
#  extrusion_cubic_check.jl -- the density extrusion EXT with a MEAN-PRESERVING upwind reconstruction
#
#      julia -t 2 scripts/transport/extrusion_cubic_check.jl        ->  output/ts_data/extrusion_cubic.{txt,jls}
#
#  The plain least-squares upwind fit is stable only on the 1-ring quadratic (UPW2); wider stencils were violently
#  unstable.  Forcing the fit to reproduce the upwind cell's own mean makes 2-ring quadratics and cubics stable.
#  EXT3 = cubic on 2 rings, mean-preserving.
#  A. truncation (as extrusion_check.jl), meshes 4-7: EXT2 (plain, 1 ring), EXT3, with H_P2 and the exact area
#  B. spectra of EXT3 (H_P2) for steady translation / swirl / cellular, meshes 4-5; plain 2-ring fits for comparison
#  C. the five tests with EXT3: H_DEC, H_P2, exact (meshes 4-6), H_P2 (mesh 7)
# ============================================================================
include(joinpath(@__DIR__, "extrusion_lie.jl"))
using Printf, Serialization
empty!(lines)
const CMESHES = haskey(ENV, "CUB_MESHES") ? Tuple(String.(split(ENV["CUB_MESHES"], ","))) : ("mesh4", "mesh5", "mesh6", "mesh7")
const CFILE = joinpath(TSDIR, "extrusion_cubic.jls")
R = isfile(CFILE) && get(ENV, "CUB_KEEP", "0") == "1" ? deserialize(CFILE) : Dict{Any,Any}()
function ext_matrix(G, U, Qv)
    ld = G["ld"]; w = (_GL3[1][2], _GL3[2][2], _GL3[3][2]); m = 0.5 .* (w[1] .* Qv[1] .+ w[2] .* Qv[2] .+ w[3] .* Qv[3])
    K = spzeros(length(U), G["mesh"].cells.n)
    for g in 1:3
        ŵ = Qv[g] .- m .+ U ./ ld
        K = K .+ Diagonal((ld ./ 2) .* w[g] .* max.(ŵ, 0)) * G["R1g"][g] .+ Diagonal((ld ./ 2) .* w[g] .* min.(ŵ, 0)) * G["R2g"][g]
    end
    return G["Div"] * K * Diagonal(1 ./ G["A"])
end
setrec!(G, p, rings, con) = (G["R1g"] = lsq_edge_points(G["mesh"], p; side = 1, rings, constrained = con);
                             G["R2g"] = lsq_edge_points(G["mesh"], p; side = 2, rings, constrained = con); G)

for mn in CMESHES
    t0 = time()
    G = lie_setup(mn; cfv = (), upw = ()); extrusion_setup!(G; p_rho = 3, rings = 2, constrained = true)
    mesh = G["mesh"]; A = G["A"]; h = G["h"]; R[(mn, :h)] = h
    logp(@sprintf("\n%s: %d cells, h = 1/%d (setup %.0f s)", mn, mesh.cells.n, round(Int, 1 / h), time() - t0))
    # ---------------- A. truncation
    ρf(x) = 1 + bump(x); ρ̄ = discretize_scalar_field_cells(mesh, ρf; qdeg = 5).values ./ A
    divex = G["Div"] * reduce_dual_edges(mesh, star_field(x -> ρf(x) .* swirl_div(x))).values; sc = maximum(abs.(divex ./ A))
    vd = reduce_primal_edges(mesh, swirl_div).values; Ud = reduce_dual_edges(mesh, star_field(swirl_div)).values
    Qd = ntuple(g -> G["Qg"][g] * vd, 3)
    Qdex = ntuple(g -> [dot(swirl_div(dual_gauss(mesh, e)[1][g]), _unit_t(mesh, e)) for e in 1:mesh.edges.n], 3)
    out = String[]
    for s in ("H_P2", "exact")
        τ = (G["Div"] * extrusion_flux(G, s == "exact" ? Ud : G[s] * vd, s == "exact" ? Qdex : Qd, ρ̄) .- divex) ./ A
        R[(mn, s, "EXT3", :trunc)] = (inf = maximum(abs.(τ)) / sc, l2 = sqrt(sum(A .* τ .^ 2)) / sc)
        push!(out, @sprintf("EXT3 %s %.2e/%.2e", s, R[(mn, s, "EXT3", :trunc)].inf, R[(mn, s, "EXT3", :trunc)].l2))
    end
    logp("   A. trunc density (max/L2): " * join(out, "  "))
    # ---------------- B. spectra
    if mn in ("mesh4", "mesh5")
        for (fname, f) in (("translation", x -> [1.0, 1.0]), ("swirl", swirl), ("cellular", cellular))
            v = reduce_primal_edges(mesh, f).values; Qv = ntuple(g -> G["Qg"][g] * v, 3)
            R[(mn, fname, "EXT3", :spec)] = maximum(real.(eigvals(Matrix(-ext_matrix(G, G["H_P2"] * v, Qv)))))
            logp(@sprintf("   B. max Re λ, %s: EXT3 (H_P2) %+.1e", fname, R[(mn, fname, "EXT3", :spec)]))
        end
        if mn == "mesh4"
            Rs = (G["R1g"], G["R2g"])
            for (p, rings, con, key) in ((2, 2, false, "q2plain"), (2, 2, true, "q2mp"), (3, 2, false, "c2plain"))
                setrec!(G, p, rings, con); v = reduce_primal_edges(mesh, swirl).values; Qv = ntuple(g -> G["Qg"][g] * v, 3)
                R[(mn, "swirl", key, :spec)] = maximum(real.(eigvals(Matrix(-ext_matrix(G, G["H_P2"] * v, Qv)))))
                logp(@sprintf("   B. max Re λ, swirl, %s: %+.1e", key, R[(mn, "swirl", key, :spec)]))
            end
            G["R1g"], G["R2g"] = Rs
        end
    end
    # ---------------- C. the five tests
    for (name, test) in pairs(TESTS)
        ρ0 = cell_averages(mesh, test.ρ0) .* A; ex = exact_at_T(mesh, test); lo, hi = extrema(ex); out = String[]
        for s in (mn == "mesh7" ? ("H_P2",) : ("H_DEC", "H_P2", "exact"))
            t = @elapsed (ρT = rk4(extrusion_rhs(G, s, test), ρ0, test.T, h / 8)); e = ρT ./ A .- ex
            R[(mn, String(name), s, "EXT3")] = (l2 = sqrt(sum(A .* e .^ 2) / sum(A .* ex .^ 2)), linf = maximum(abs.(e)) / maximum(abs.(ex)),
                                                under = (minimum(ρT ./ A) - lo) / (hi - lo), mass = abs(sum(ρT) - sum(ρ0)) / sum(ρ0), time = t)
            push!(out, @sprintf("%s %.2e/%.2e", s, R[(mn, String(name), s, "EXT3")].l2, R[(mn, String(name), s, "EXT3")].linf))
        end
        logp("   C. $name EXT3 (ℓ2/ℓ∞): " * join(out, "  "))
    end
    serialize(CFILE, R)
end
write(joinpath(TSDIR, "extrusion_cubic.txt"), join(lines, "\n") * "\n")
