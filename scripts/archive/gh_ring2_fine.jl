# ============================================================================
#  gh_ring2_fine.jl -- finer levels for the relaxed star ([P1]² rows + [P2]² patch test, 2-ring stencils, ε = 1e-8)
#
#      julia scripts/archive/gh_ring2_fine.jl        (writes gh_data/ring2_fine.txt)
# ============================================================================
include(joinpath(@__DIR__, "gh_lab.jl"))
const OUTF = joinpath(@__DIR__, "..", "..", "output", "gh_data", "ring2_fine.txt")
lines = String[]
logp(s) = (println(s); flush(stdout); push!(lines, s))
logp(@sprintf("%-10s %7s %-12s %9s %9s %9s %9s %9s %6s %5s", "mesh", "h", "star", "row P1", "weak P2", "flux(u)",
              "Poisson", "patch", "Λ", "dev"))
for (mname, M) in (("scvt-mesh3", readmesh(joinpath(@__DIR__, "..", "..", "output", "gh_data", "mesh3.txt"))), ("random32", latmesh(32, "random")),
                   ("scvt-mesh4", readmesh(joinpath(@__DIR__, "..", "..", "output", "gh_data", "mesh4.txt"))))
    geos = [egeo(M, e, M.X[M.E[e][1]]) for e in eachindex(M.E)]
    u(x) = [sin(2pi * x[2] / M.Py) + 0.5cos(2pi * (x[1] / M.Px + x[2] / M.Py)), cos(2pi * x[1] / M.Px)]
    p(x) = sin(2pi * x[1] / M.Px) * cos(2pi * x[2] / M.Py)
    gp(x) = [2pi / M.Px * cos(2pi * x[1] / M.Px) * cos(2pi * x[2] / M.Py), -2pi / M.Py * sin(2pi * x[1] / M.Px) * sin(2pi * x[2] / M.Py)]
    hmax = maximum(g.L for g in geos)
    cells = tri_cells(M; ring = 2); D2 = celldata(M, cells, 2)
    _, info = constrained_R(M, cells, D2, geos, 2)
    S = select_op(M, geos, 2, 1); Gs = S * info.G * S'; rs = S * info.rhs
    ν = cholesky(Symmetric(Gs + 1e-8 * maximum(abs, diag(Gs)) * I)) \ rs
    R2 = R_from_mu(cells, D2, S' * ν, nbas(2))
    dev = maximum(sqrt(tr((R - d.R0)' * d.Mm * (R - d.R0)) / tr(d.R0' * d.Mm * d.R0)) for (R, d) in zip(R2, D2))
    H = assemble_H(M, cells, D2, R2); pe, pd = poisson(M, geos, H, p, gp)
    logp(@sprintf("%-10s %7.4f %-12s %9.1e %9.1e %9.2e %9.2e %9.1e %6.2f %5.2f", mname, hmax, "P1+wP2 ring2",
                  row_test(M, H, geos, 1), weak_test(M, H, geos, 2), flux_err(M, geos, H, u), pe, pd, Λnorm(H), dev))
end
write(OUTF, join(lines, "\n") * "\n")
