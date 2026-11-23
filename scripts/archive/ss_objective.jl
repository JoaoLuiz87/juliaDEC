# ============================================================================
#  ss_objective.jl -- which admissible shares?  minimum norm (the guide) versus closest to uniform
#
#      julia scripts/archive/ss_objective.jl        (writes ss_data/objective.txt)
#
#  Both are exact (partition of unity and admissibility to rounding); they differ only in the objective.
#  For each: max|θ|, max/mean row constant C_e, flux consistency error of the guide's field u, the Poisson
#  errors in the norms of the guide's Theorem 4.2 (nodal max and ‖H d0 p_h - β(∇p)‖_{H⁻¹}, relative), and the
#  spectral bound μ = λmin(Hd^-1/2 H Hd^-1/2).  DEC for comparison.
# ============================================================================
include(joinpath(@__DIR__, "..", "..", "SymStar.jl")); using .SymStar
include(joinpath(@__DIR__, "ss_mesh.jl"))
using LinearAlgebra, SparseArrays, Printf
const OUT = joinpath(@__DIR__, "..", "..", "output", "ss_data"); mkpath(OUT)
lines = String[]; logp(s) = (println(s); flush(stdout); push!(lines, s))
logp(@sprintf("%-11s %6s %-8s %8s %8s %7s %9s %9s %9s %9s %9s %7s", "mesh", "E", "shares", "max|θ|", "max C_e", "mean C",
              "partition", "flux max", "flux rms", "p max", "flux H⁻¹", "μ"))

function poisson_errors(g, S, period)
    kx, ky = 2pi / period[1], 2pi / period[2]
    p(x) = sin(kx * x[1] + 0.3) * cos(ky * x[2]) + 0.5cos(kx * x[1] + 2ky * x[2])
    gp(x) = [kx * cos(kx * x[1] + 0.3) * cos(ky * x[2]) - 0.5kx * sin(kx * x[1] + 2ky * x[2]),
             -ky * sin(kx * x[1] + 0.3) * sin(ky * x[2]) - ky * sin(kx * x[1] + 2ky * x[2])]
    _, βg = reductions(g, gp)
    nV = size(g.d0, 2); L = g.d0' * S * g.d0
    ph = (lu([L ones(nV); ones(1, nV) 0.0]) \ vcat(g.d0' * βg, 0.0))[1:nV]
    X = [g.me[1, :]]                                   # nodal values: vertices from the incidence (endpoint of the first edge)
    pI = zeros(nV); seen = falses(nV)
    for e in 1:size(g.d0, 1)
        for (k, s) in zip(findnz(g.d0[e, :])...)
            seen[k] && continue
            pI[k] = p(g.me[e, :] + (s > 0 ? 0.5 : -0.5) * g.le[e] * g.t[e, :]); seen[k] = true
        end
    end
    ph .+= sum(pI - ph) / nV
    F = cholesky(Symmetric(S)); Hinv(x) = sqrt(dot(x, F \ x))
    maximum(abs.(ph - pI)) / maximum(abs.(pI)), Hinv(S * g.d0 * ph - βg) / Hinv(βg)
end

function compare(name, X, T, period, h)
    g = edge_geometry(X, T, period)
    kx, ky = 2pi / period[1], 2pi / period[2]
    u(x) = [sin(ky * x[2]) + 0.4cos(kx * x[1]), cos(kx * x[1] + ky * x[2])]
    α, β = reductions(g, u); Hd = spdiagm(g.lde ./ g.le)
    for obj in (:minnorm, :uniform)
        H, info = symmetric_star(g.me, g.t, g.le, g.lde, g.delta, period, h; rho = 2.0, objective = obj, verbose = false)
        C, _ = row_constants(H, g.me, g.t, g.le, g.lde, period, h); ε = abs.(H * α - β) ./ g.lde
        μ = try smallest_eigenvalue(H; D = Hd) catch; NaN end
        pe, fe = isnan(μ) || μ <= 0 ? (NaN, NaN) : poisson_errors(g, H, period)
        logp(@sprintf("%-11s %6d %-8s %8.3f %8.2f %7.2f %9.1e %9.2e %9.2e %9.2e %9.2e %7.3f", name, length(g.le), obj,
                      max(abs(info.theta_min), abs(info.theta_max)), maximum(C), sum(C) / length(C), info.partition_err,
                      maximum(ε), sqrt(sum(abs2, ε) / length(ε)), pe, fe, μ))
    end
    ε0 = abs.(Hd * α - β) ./ g.lde; pe, fe = poisson_errors(g, Hd, period)
    logp(@sprintf("%-11s %6d %-8s %8s %8s %7s %9s %9.2e %9.2e %9.2e %9.2e %7s", name, length(g.le), "DEC", "", "1", "1", "",
                  maximum(ε0), sqrt(sum(abs2, ε0) / length(ε0)), pe, fe, "1"))
end

for n in (10, 20, 40, 80)
    compare("lattice $n", porting_mesh(n)...)
end
for mn in ("mesh3", "mesh4", "mesh5")
    compare("scvt $mn", scvt_mesh(mn)...)
end
write(joinpath(OUT, "objective.txt"), join(lines, "\n") * "\n")
