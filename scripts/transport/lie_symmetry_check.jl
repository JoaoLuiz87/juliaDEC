# ============================================================================
#  lie_symmetry_check.jl -- does the asymmetry of H_P2 hurt the discrete Lie derivative?
#
#      julia -t 4 scripts/transport/lie_symmetry_check.jl     ->  output/ts_data/lie_symmetry.{txt,jls}
#
#  Controlled comparison: H_P2 against its symmetric part H_s = (H_P2 + H_P2')/2 (same stencil, same entries, only the
#  antisymmetric part removed) and against the symmetric H_P1.  For each:
#    free stream    max |D̃ H v / A|  (swirl, meshes 4-7)
#    spectra        max Re λ(-L) on meshes 4 and 5 (steady translation, swirl, cellular) in the TRiSK, DEC, UPW2 orderings
#    transport      ℓ2 / ℓ∞ errors of the deformational test (TESTS) on meshes 5-7 with UPW2 and with the DEC ordering
# ============================================================================
include(joinpath(@__DIR__, "lie_derivative.jl"))
using Printf, Serialization
empty!(lines); R = Dict{Any,Any}()
symstar!(G) = (G["H_sym"] = 0.5 .* (G["H_P2"] .+ sparse(G["H_P2"]')); G)
const SSTARS = ("H_P1", "H_P2", "H_sym")

logp("free stream max|D̃Hv/A| (swirl):")
for mn in ("mesh4", "mesh5", "mesh6", "mesh7")
    G = symstar!(lie_setup(mn; cfv = (), upw = mn in ("mesh5", "mesh6", "mesh7") ? (2,) : ()))
    mesh = G["mesh"]; A = G["A"]; R[(mn, :h)] = G["h"]
    v0 = reduce_primal_edges(mesh, swirl).values
    for s in SSTARS
        R[(mn, s, :free)] = maximum(abs.(G["Div"] * (G[s] * v0) ./ A))
    end
    logp(@sprintf("   %s:  H_P1 %.2e   H_P2 %.2e   H_sym %.2e   (‖H_a‖/‖H_P2‖ = %.2f)", mn, R[(mn, "H_P1", :free)], R[(mn, "H_P2", :free)],
                  R[(mn, "H_sym", :free)], norm(G["H_P2"] .- G["H_sym"]) / norm(G["H_P2"])))
    if mn in ("mesh4", "mesh5")                                  # spectra
        for (fname, f) in (("translation", x -> [1.0, 1.0]), ("swirl", swirl), ("cellular", cellular))
            v = reduce_primal_edges(mesh, f).values
            out = String[]
            for s in SSTARS, var in ("TRiSK", "DEC", "UPW2")
                var == "UPW2" && mn == "mesh4" && continue
                L = var == "UPW2" ? lie_density_upwind(G, var, G[s] * v) : lie_density(G, flux_operator(G, var, s, v))
                R[(mn, fname, s, var)] = maximum(real.(eigvals(Matrix(-L))))
                push!(out, @sprintf("%s %s %+.1e", s, var, R[(mn, fname, s, var)]))
            end
            logp("   $mn $fname: " * join(out, "  "))
        end
    end
    if mn in ("mesh5", "mesh6", "mesh7")                         # transport, deformational test
        test = TESTS.deformational
        ρ0 = cell_averages(mesh, test.ρ0) .* A; ex = cell_averages(mesh, test.ρ0)
        for s in SSTARS, var in ("UPW2", "DEC")
            ρT = rk4(rhs_function(G, var, s, test), ρ0, test.T, G["h"] / 8) ./ A
            e = ρT .- ex
            R[(mn, :deform, s, var)] = (l2 = sqrt(sum(A .* e .^ 2) / sum(A .* ex .^ 2)), linf = maximum(abs.(e)) / maximum(abs.(ex)))
            logp(@sprintf("   %s deformational %-6s %-5s ℓ2 %.3e  ℓ∞ %.3e", mn, s, var, R[(mn, :deform, s, var)].l2, R[(mn, :deform, s, var)].linf))
        end
    end
    serialize(joinpath(TSDIR, "lie_symmetry.jls"), R)
end
write(joinpath(TSDIR, "lie_symmetry.txt"), join(lines, "\n") * "\n")
