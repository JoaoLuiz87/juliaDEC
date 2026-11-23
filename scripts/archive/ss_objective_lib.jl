# ss_objective_lib.jl -- Poisson errors in the norms of the guide's Theorem 4.2 (shared by ss_objective.jl, ss_abstol.jl)
function poisson_errors(g, S, period)
    kx, ky = 2pi / period[1], 2pi / period[2]
    p(x) = sin(kx * x[1] + 0.3) * cos(ky * x[2]) + 0.5cos(kx * x[1] + 2ky * x[2])
    gp(x) = [kx * cos(kx * x[1] + 0.3) * cos(ky * x[2]) - 0.5kx * sin(kx * x[1] + 2ky * x[2]),
             -ky * sin(kx * x[1] + 0.3) * sin(ky * x[2]) - ky * sin(kx * x[1] + 2ky * x[2])]
    _, βg = reductions(g, gp)
    nV = size(g.d0, 2); L = g.d0' * S * g.d0
    ph = (lu([L ones(nV); ones(1, nV) 0.0]) \ vcat(g.d0' * βg, 0.0))[1:nV]
    pI = zeros(nV); seen = falses(nV)
    for e in 1:size(g.d0, 1), (k, s) in zip(findnz(g.d0[e, :])...)
        seen[k] && continue
        pI[k] = p(g.me[e, :] + (s > 0 ? 0.5 : -0.5) * g.le[e] * g.t[e, :]); seen[k] = true
    end
    ph .+= sum(pI - ph) / nV
    F = cholesky(Symmetric(S)); Hinv(x) = sqrt(dot(x, F \ x))
    maximum(abs.(ph - pI)) / maximum(abs.(pI)), Hinv(S * g.d0 * ph - βg) / Hinv(βg)
end
