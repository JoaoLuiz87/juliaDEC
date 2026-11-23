# ============================================================================
#  gh_weak.jl -- the relaxed target: strong [P1]² flux consistency + the [P2]² patch test (vertex closure)
#
#      julia scripts/archive/gh_weak.jl        (writes gh_data/weak_results.txt)
#
#  A. Small meshes: dimension of the kernel of the constraint Gram matrix and the obstruction (component of
#     the right-hand side along it) for three targets with [P2]²-exact reconstructions on triangle cells:
#        strong P2      : all row moment conditions on [P2]²  (impossible, Theorem)
#        P1 + weak P2   : rows on [P1]², vertex patch test on [P2]²
#  B. Refinement: the P1 + weak P2 star versus the strong P1 star (gh_main 'tri-cons') and DEC.
# ============================================================================
include(joinpath(@__DIR__, "gh_lab.jl"))
const OUTF = joinpath(@__DIR__, "..", "..", "output", "gh_data", "weak_results.txt")
lines = String[]
logp(s) = (println(s); flush(stdout); push!(lines, s))

logp("A. kernels and obstructions (triangle cells, 1-ring, [P2]²-exact reconstructions)")
for (mname, M) in (("random N=6", latmesh(6, "random")), ("pattern N=6", latmesh(6, "pattern")),
                   ("SCVT mesh2", readmesh(joinpath(@__DIR__, "..", "..", "output", "gh_data", "mesh2.txt"))))
    geos = [egeo(M, e, M.X[M.E[e][1]]) for e in eachindex(M.E)]
    cells = tri_cells(M); D = celldata(M, cells, 2)
    _, info = constrained_R(M, cells, D, geos, 2)
    for (tname, S) in (("strong P2", sparse(1.0I, size(info.G, 1), size(info.G, 1))), ("P1 + weak P2", select_op(M, geos, 2, 1)))
        Gs = Matrix(S * info.G * S'); rs = S * info.rhs
        F = eigen(Symmetric(Gs)); nul = F.values .< 1e-10 * F.values[end]
        ob = norm(F.vectors[:, nul]' * rs) / norm(rs)
        logp(@sprintf("  %-12s %-13s: %5d conditions, dim kernel %3d, obstruction %.1e", mname, tname, length(rs), count(nul), ob))
    end
end

logp("\nB. refinement (triangle cells, 1-ring)")
logp(@sprintf("%-10s %7s %-9s %9s %9s %9s %9s %9s %9s %6s %6s %5s", "mesh", "h", "star", "row P0", "row P1", "weak P2",
              "flux(u)", "Poisson", "patch", "Λ", "λmin", "dev"))
function refine(M, mname)
    geos = [egeo(M, e, M.X[M.E[e][1]]) for e in eachindex(M.E)]
    Px, Py = M.Px, M.Py
    u(x) = [sin(2pi * x[2] / Py) + 0.5cos(2pi * (x[1] / Px + x[2] / Py)), cos(2pi * x[1] / Px)]
    p(x) = sin(2pi * x[1] / Px) * cos(2pi * x[2] / Py)
    gp(x) = [2pi / Px * cos(2pi * x[1] / Px) * cos(2pi * x[2] / Py), -2pi / Py * sin(2pi * x[1] / Px) * sin(2pi * x[2] / Py)]
    w = [g.Ld / g.L for g in geos]; hmax = maximum(g.L for g in geos)
    cells = tri_cells(M)
    D1 = celldata(M, cells, 1); R1, _ = constrained_R(M, cells, D1, geos, 1)
    D2 = celldata(M, cells, 2); _, info = constrained_R(M, cells, D2, geos, 2)
    S = select_op(M, geos, 2, 1); Gs = S * info.G * S'; rs = S * info.rhs
    ν = cholesky(Symmetric(Gs + 1e-13 * maximum(abs, diag(Gs)) * I)) \ rs
    R2 = R_from_mu(cells, D2, S' * ν, nbas(2))
    dev = maximum(sqrt(tr((R - d.R0)' * d.Mm * (R - d.R0)) / tr(d.R0' * d.Mm * d.R0)) for (R, d) in zip(R2, D2))
    for (lbl, H, dv) in (("DEC", spdiagm(w), NaN), ("P1", assemble_H(M, cells, D1, R1), NaN), ("P1+wP2", assemble_H(M, cells, D2, R2), dev))
        lmin, _ = length(w) <= 3100 ? wspec(H, w) : (NaN, NaN)
        pe, pd = poisson(M, geos, H, p, gp)
        logp(@sprintf("%-10s %7.4f %-9s %9.1e %9.1e %9.1e %9.2e %9.2e %9.1e %6.2f %6.3f %5.2f", mname, hmax, lbl,
                      row_test(M, H, geos, 0), row_test(M, H, geos, 1), weak_test(M, H, geos, 2),
                      flux_err(M, geos, H, u), pe, pd, Λnorm(H), lmin, dv))
    end
end
for N in (8, 16, 32)
    refine(latmesh(N, "random"), "random$N")
end
for mn in ("mesh2", "mesh3", "mesh4")
    refine(readmesh(joinpath(@__DIR__, "..", "..", "output", "gh_data", "$mn.txt")), "scvt-$mn")
end
write(OUTF, join(lines, "\n") * "\n")
