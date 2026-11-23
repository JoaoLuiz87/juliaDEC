# ============================================================================
#  lie_spectra_check.jl -- spectra of the semi-discrete Lie derivative for steady flows (stability)
#
#      julia -t 4 scripts/transport/lie_spectra_check.jl      ->  output/ts_data/lie_spectra.{txt,jls}
#
#  For dρ̃/dt = -L ρ̃ with a steady velocity: max Re λ(-L) (> 0: a growing mode; the continuous operator is skew for a
#  divergence-free u, Re λ = 0).  For the TRiSK ordering the energy identity of the note gives
#        max Re λ(-L) ≤ ½ max_c ( -(D̃U)_c / A_c ) =: bound,     U = H v (the discrete volume flux),
#  so the growth is controlled by the discrete divergence of U only -- not by the symmetry of H.
# ============================================================================
include(joinpath(@__DIR__, "lie_derivative.jl"))
using Printf, Serialization
empty!(lines)
const SPEC_SCHEMES = (("exact", "TRiSK"), ("H_DEC", "TRiSK"), ("H_P1", "TRiSK"), ("H_P2", "TRiSK"), ("H_P1", "DEC"), ("H_P2", "DEC"),
                      ("H_P2", "Ham"), ("H_P2", "Peixoto"), ("H_P2", "CFV2"), ("H_DEC", "UPW2"), ("H_P2", "UPW2"))
const FLOWS = (("translation", x -> [1.0, 1.0]), ("swirl", swirl), ("cellular", cellular))
R = Dict{Any,Any}()
for mn in ("mesh4", "mesh5")
    G = lie_setup(mn; cfv = (2,), upw = (2,)); mesh = G["mesh"]; A = G["A"]
    logp("\n$mn ($(mesh.cells.n) cells): max Re λ(-L)   [bound ½max(-D̃U/A) for the TRiSK ordering]")
    for (fname, f) in FLOWS
        v = reduce_primal_edges(mesh, f).values; Uex = reduce_dual_edges(mesh, star_field(f)).values
        out = String[]
        for (s, var) in SPEC_SCHEMES
            U = s == "exact" ? Uex : G[s] * v
            L = var in UPWINDS ? lie_density_upwind(G, var, U) : lie_density(G, flux_operator(G, var, s, v; Uex))
            λ = eigvals(Matrix(-L)); m = maximum(real.(λ))
            bound = var == "TRiSK" ? 0.5 * maximum(-(G["Div"] * U) ./ A) : NaN
            R[(mn, fname, s, var)] = (maxre = m, bound = bound, λ = mn == "mesh4" ? λ : nothing)
            push!(out, var == "TRiSK" ? @sprintf("%s %s %+.1e [%.1e]", s, var, m, bound) : @sprintf("%s %s %+.1e", s, var, m))
        end
        logp("   $fname:"); foreach(o -> logp("      " * o), out)
    end
    serialize(joinpath(TSDIR, "lie_spectra.jls"), R)
end
write(joinpath(TSDIR, "lie_spectra.txt"), join(lines, "\n") * "\n")
