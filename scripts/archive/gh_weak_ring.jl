# ============================================================================
#  gh_weak_ring.jl -- the relaxed target ([P1]² rows + [P2]² patch test): stencil width and regularisation
#
#      julia scripts/archive/gh_weak_ring.jl        (writes gh_data/weak_ring.txt)
# ============================================================================
include(joinpath(@__DIR__, "gh_lab.jl"))
const OUTF = joinpath(@__DIR__, "..", "..", "output", "gh_data", "weak_ring.txt")
lines = String[]
logp(s) = (println(s); flush(stdout); push!(lines, s))
logp(@sprintf("%-10s %7s %4s %7s %9s %9s %9s %9s %9s %8s %6s %6s", "mesh", "h", "ring", "ε", "row P1", "weak P2",
              "flux(u)", "Poisson", "patch", "Λ", "λmin", "dev"))
for (mname, M) in (("random16", latmesh(16, "random")), ("scvt-mesh2", readmesh(joinpath(@__DIR__, "..", "..", "output", "gh_data", "mesh2.txt"))),
                   ("scvt-mesh3", readmesh(joinpath(@__DIR__, "..", "..", "output", "gh_data", "mesh3.txt"))))
    geos = [egeo(M, e, M.X[M.E[e][1]]) for e in eachindex(M.E)]
    Px, Py = M.Px, M.Py
    u(x) = [sin(2pi * x[2] / Py) + 0.5cos(2pi * (x[1] / Px + x[2] / Py)), cos(2pi * x[1] / Px)]
    p(x) = sin(2pi * x[1] / Px) * cos(2pi * x[2] / Py)
    gp(x) = [2pi / Px * cos(2pi * x[1] / Px) * cos(2pi * x[2] / Py), -2pi / Py * sin(2pi * x[1] / Px) * sin(2pi * x[2] / Py)]
    w = [g.Ld / g.L for g in geos]; hmax = maximum(g.L for g in geos)
    S = select_op(M, geos, 2, 1)
    for ring in (1, 2)
        cells = tri_cells(M; ring = ring); D2 = celldata(M, cells, 2)
        _, info = constrained_R(M, cells, D2, geos, 2)
        Gs = S * info.G * S'; rs = S * info.rhs
        for reg in (1e-13, 1e-8, 1e-5)
            ν = cholesky(Symmetric(Gs + reg * maximum(abs, diag(Gs)) * I)) \ rs
            R2 = R_from_mu(cells, D2, S' * ν, nbas(2))
            dev = maximum(sqrt(tr((R - d.R0)' * d.Mm * (R - d.R0)) / tr(d.R0' * d.Mm * d.R0)) for (R, d) in zip(R2, D2))
            H = assemble_H(M, cells, D2, R2); lmin, _ = wspec(H, w); pe, pd = poisson(M, geos, H, p, gp)
            logp(@sprintf("%-10s %7.4f %4d %7.0e %9.1e %9.1e %9.2e %9.2e %9.1e %8.2f %6.3f %6.2f", mname, hmax, ring, reg,
                          row_test(M, H, geos, 1), weak_test(M, H, geos, 2), flux_err(M, geos, H, u), pe, pd, Λnorm(H), lmin, dev))
        end
    end
end
write(OUTF, join(lines, "\n") * "\n")
