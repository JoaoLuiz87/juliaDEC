# ============================================================================
#  lie_testcases_check.jl -- convergence of the discrete Lie derivative (lie_derivative.jl) on five advection tests
#
#      julia -t 4 scripts/transport/lie_testcases_check.jl      ->  output/ts_data/lie_testcases.{txt,jls}
#
#  Tests (TESTS in lie_derivative.jl), unit torus, T = 1, RK4 with dt = h/8, meshes 2-7:
#    translation    u = (1,1), Gaussian hill, one period (exact = initial)
#    cellular       steady Taylor–Green cells, sheared hill (exact by characteristics, cell-averaged)
#    deformational  translation + reversing swirl, two Gaussian hills (exact = initial)
#    divergent      translation + reversing divergent flow (the density is compressed and expanded; exact = initial)
#    cosine_bells   deformational flow, two C¹ cosine bells on a 0.1 background (exact = initial)
#  Errors of the cell values ρ̄ at T, normalised as in Williamson et al. (1992):
#    ℓ1 = Σ A|e| / Σ A|ρex|,  ℓ2 = (Σ A e² / Σ A ρex²)^½,  ℓ∞ = max|e| / max|ρex|;
#  and the over/undershoots of Lauritzen et al. (2012): (min ρ̄ - min ρex) and (max ρ̄ - max ρex) over (max - min ρex).
# ============================================================================
include(joinpath(@__DIR__, "lie_derivative.jl"))
using Printf, Serialization
empty!(lines)

const TMESHES = haskey(ENV, "LIE_MESHES") ? Tuple(String.(split(ENV["LIE_MESHES"], ","))) : ("mesh2", "mesh3", "mesh4", "mesh5", "mesh6", "mesh7")
const SCHEMES = (("H_DEC", "TRiSK"), ("H_P1", "TRiSK"), ("H_P2", "TRiSK"), ("H_P2", "DEC"), ("H_P2", "Ham"), ("H_P2", "Peixoto"),
                 ("H_DEC", "CFV2"), ("H_P1", "CFV2"), ("H_P2", "CFV2"), ("H_P2", "CFV3"), ("exact", "CFV2"),
                 ("H_DEC", "UPW2"), ("H_P1", "UPW2"), ("H_P2", "UPW2"), ("exact", "UPW2"))
slabel(s, v) = s == "H_DEC" && v == "TRiSK" ? "H_DEC TRiSK=DEC=Ham" : "$s $v"
R = Dict{Any,Any}()

for mn in TMESHES
    t0 = time()
    G = lie_setup(mn); mesh = G["mesh"]; A = G["A"]; h = G["h"]; R[(mn, :h)] = h
    logp(@sprintf("\n%s: %d cells, h = 1/%d (setup %.0f s)", mn, mesh.cells.n, round(Int, 1 / h), time() - t0))
    for (name, test) in pairs(TESTS)
        ρ0 = cell_averages(mesh, test.ρ0) .* A
        te = @elapsed (ex = exact_at_T(mesh, test))
        lo, hi = extrema(ex)
        logp(@sprintf("   %s: %s   (exact %.0f s)", name, test.what, te))
        logp("      scheme                 |      ℓ1        ℓ2        ℓ∞   | undershoot overshoot |  mass drift   (s)")
        for (s, var) in SCHEMES
            t = @elapsed (ρT = rk4(rhs_function(G, var, s, test), ρ0, test.T, h / 8))
            ρ̄ = ρT ./ A; e = ρ̄ .- ex
            r = (l1 = sum(A .* abs.(e)) / sum(A .* abs.(ex)), l2 = sqrt(sum(A .* e .^ 2) / sum(A .* ex .^ 2)), linf = maximum(abs.(e)) / maximum(abs.(ex)),
                 under = (minimum(ρ̄) - lo) / (hi - lo), over = (maximum(ρ̄) - hi) / (hi - lo), mass = abs(sum(ρT) - sum(ρ0)) / sum(ρ0), time = t)
            R[(mn, String(name), s, var)] = r
            mn == "mesh5" && (R[(mn, String(name), s, var, :final)] = ρ̄)
            logp(@sprintf("      %-22s | %9.2e %9.2e %9.2e | %+9.1e %+9.1e |  %8.1e  %5.1f", slabel(s, var), r.l1, r.l2, r.linf, r.under, r.over, r.mass, t))
        end
        mn == "mesh5" && (R[(mn, String(name), :exact)] = ex)
    end
    serialize(joinpath(TSDIR, "lie_testcases.jls"), R)
end

# ---------------- convergence orders
hs = [R[(m, :h)] for m in TMESHES]
ords(v) = [log(v[i] / v[i+1]) / log(hs[i] / hs[i+1]) for i in 1:length(v)-1]
fit(v, k) = (k = min(k, length(v)); x = log.(hs[end-k+1:end]); y = log.(v[end-k+1:end]); sum((x .- mean(x)) .* (y .- mean(y))) / sum((x .- mean(x)) .^ 2))
mean(x) = sum(x) / length(x)
logp("\nObserved orders between consecutive meshes (" * join(TMESHES, " -> ") * "); 'fit' = least-squares slope over the last three meshes")
for (name, _) in pairs(TESTS)
    logp("   $name")
    for (s, var) in SCHEMES
        v2 = [R[(m, String(name), s, var)].l2 for m in TMESHES]; vi = [R[(m, String(name), s, var)].linf for m in TMESHES]
        logp(@sprintf("      %-22s ℓ2 %s | fit %5.2f    ℓ∞ %s | fit %5.2f", slabel(s, var), join([@sprintf("%5.2f", o) for o in ords(v2)], " "),
                      fit(v2, 3), join([@sprintf("%5.2f", o) for o in ords(vi)], " "), fit(vi, 3)))
    end
end
serialize(joinpath(TSDIR, "lie_testcases.jls"), R)
write(joinpath(TSDIR, "lie_testcases.txt"), join(lines, "\n") * "\n")
