# ============================================================================
#  gh_ring2.jl -- refinement with 2-ring triangle stencils: strong [P1]² versus [P1]² rows + [P2]² patch test
#
#      julia scripts/archive/gh_ring2.jl        (writes gh_data/ring2.txt)
# ============================================================================
include(joinpath(@__DIR__, "gh_lab.jl"))
const OUTF = joinpath(@__DIR__, "..", "..", "output", "gh_data", "ring2.txt")
lines = String[]
logp(s) = (println(s); flush(stdout); push!(lines, s))
logp(@sprintf("%-10s %7s %-12s %9s %9s %9s %9s %9s %6s %6s %5s", "mesh", "h", "star", "row P1", "weak P2", "flux(u)",
              "Poisson", "patch", "Λ", "λmin", "dev"))
devof(Rs, D) = maximum(sqrt(tr((R - d.R0)' * d.Mm * (R - d.R0)) / tr(d.R0' * d.Mm * d.R0)) for (R, d) in zip(Rs, D))
for N in (8, 12, 16, 24)
    M = latmesh(N, "random"); mname = "random$N"
    geos = [egeo(M, e, M.X[M.E[e][1]]) for e in eachindex(M.E)]
    u(x) = [sin(2pi * x[2] / M.Py) + 0.5cos(2pi * (x[1] / M.Px + x[2] / M.Py)), cos(2pi * x[1] / M.Px)]
    p(x) = sin(2pi * x[1] / M.Px) * cos(2pi * x[2] / M.Py)
    gp(x) = [2pi / M.Px * cos(2pi * x[1] / M.Px) * cos(2pi * x[2] / M.Py), -2pi / M.Py * sin(2pi * x[1] / M.Px) * sin(2pi * x[2] / M.Py)]
    w = [g.Ld / g.L for g in geos]; hmax = maximum(g.L for g in geos)
    stars = Any[("DEC", spdiagm(w), NaN)]
    for ring in (1, 2)
        cells = tri_cells(M; ring = ring); D = celldata(M, cells, 1)
        Rs, _ = constrained_R(M, cells, D, geos, 1)
        push!(stars, ("P1 ring$ring", assemble_H(M, cells, D, Rs), devof(Rs, D)))
    end
    cells = tri_cells(M; ring = 2); D2 = celldata(M, cells, 2)
    _, info = constrained_R(M, cells, D2, geos, 2)
    S = select_op(M, geos, 2, 1); Gs = S * info.G * S'; rs = S * info.rhs
    ν = cholesky(Symmetric(Gs + 1e-8 * maximum(abs, diag(Gs)) * I)) \ rs
    R2 = R_from_mu(cells, D2, S' * ν, nbas(2))
    push!(stars, ("P1+wP2 ring2", assemble_H(M, cells, D2, R2), devof(R2, D2)))
    for (lbl, H, dv) in stars
        lmin, _ = wspec(H, w); pe, pd = poisson(M, geos, H, p, gp)
        logp(@sprintf("%-10s %7.4f %-12s %9.1e %9.1e %9.2e %9.2e %9.1e %6.2f %6.3f %5.2f", mname, hmax, lbl,
                      row_test(M, H, geos, 1), weak_test(M, H, geos, 2), flux_err(M, geos, H, u), pe, pd, Λnorm(H), lmin, dv))
    end
end
write(OUTF, join(lines, "\n") * "\n")
