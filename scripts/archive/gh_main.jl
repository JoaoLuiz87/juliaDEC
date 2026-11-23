# ============================================================================
#  gh_main.jl -- symmetric Hodge stars from a bilinear form: least-squares versus moment-constrained
#               reconstructions, triangle cells versus diamond cells, k = 1 (and k = 2 in least squares)
#
#      julia scripts/archive/gh_main.jl            (writes gh_data/main_results.txt)
#
#  Stars: DEC (diagonal), and H = Σ_C S_Cᵀ[R_CᵀM_CR_C + stab]S_C with
#     tri-pinv / dia-pinv : R_C = P_C⁺  (the sandwich; dia-pinv is the juliaDEC construction on our meshes)
#     tri-cons / dia-cons : R_C = P_C⁺ + Z_C W_Cᵀ solving the Voronoi moment conditions (minimal change)
#     tri-cons2           : k = 2, least-squares moment conditions (exact ones are impossible)
#  Columns: row errors on constants / affine fields, Hodge error of a smooth field, Poisson gradient
#  error and patch defect, Λ = max row sum, spectrum of W^{-1/2} H W^{-1/2}, deviation from P⁺.
# ============================================================================
include(joinpath(@__DIR__, "gh_lab.jl"))
const OUTF = joinpath(@__DIR__, "..", "..", "output", "gh_data", "main_results.txt")

function study(M, mname; k2 = false)
    geos = [egeo(M, e, M.X[M.E[e][1]]) for e in eachindex(M.E)]
    Px, Py = M.Px, M.Py
    u(x) = [sin(2pi * x[2] / Py) + 0.5cos(2pi * (x[1] / Px + x[2] / Py)), cos(2pi * x[1] / Px)]
    p(x) = sin(2pi * x[1] / Px) * cos(2pi * x[2] / Py)
    gp(x) = [2pi / Px * cos(2pi * x[1] / Px) * cos(2pi * x[2] / Py), -2pi / Py * sin(2pi * x[1] / Px) * sin(2pi * x[2] / Py)]
    w = [g.Ld / g.L for g in geos]
    stars = Any[("DEC", spdiagm(w), NaN, NaN)]
    for (cname, cells) in (("tri", tri_cells(M)), ("dia", dia_cells(M)))
        D = celldata(M, cells, 1)
        push!(stars, ("$cname-pinv", assemble_H(M, cells, D, [d.R0 for d in D]), NaN, NaN))
        Rs, info = constrained_R(M, cells, D, geos, 1)
        push!(stars, ("$cname-cons", assemble_H(M, cells, D, Rs), info.resid, info.dev))
    end
    if k2
        cells = tri_cells(M); D = celldata(M, cells, 2)
        Rs, info = constrained_R(M, cells, D, geos, 2)
        push!(stars, ("tri-cons2", assemble_H(M, cells, D, Rs), info.resid, info.dev))
    end
    hmax = maximum(g.L for g in geos)
    out = String[]
    for (lbl, H, res, dev) in stars
        r0 = row_test(M, H, geos, 0); r1 = row_test(M, H, geos, 1); r2 = row_test(M, H, geos, 2)
        fe = flux_err(M, geos, H, u); pe, pd = poisson(M, geos, H, p, gp)
        lmin, lmax = wspec(H, w)
        s = @sprintf("%-10s %7.4f %-9s %9.1e %9.1e %9.1e %9.2e %9.2e %9.1e %6.2f %6.3f %6.2f %8.1e %5.2f",
                     mname, hmax, lbl, r0, r1, r2, fe, pe, pd, Λnorm(H), lmin, lmax, res, dev)
        println(s); flush(stdout); push!(out, s)
    end
    out
end

hdr = @sprintf("%-10s %7s %-9s %9s %9s %9s %9s %9s %9s %6s %6s %6s %8s %5s", "mesh", "h", "star", "row P0", "row P1", "row P2",
               "flux(u)", "Poisson", "patch", "Λ", "λmin", "λmax", "moment", "dev")
println(hdr)
lines = [hdr]
for N in (8, 16, 32), kind in ("random", "mapped")
    append!(lines, study(latmesh(N, kind), "$kind$N"; k2 = N <= 16))
end
for mn in ("mesh2", "mesh3", "mesh4")
    append!(lines, study(readmesh(joinpath(@__DIR__, "..", "..", "output", "gh_data", "$mn.txt")), "scvt-$mn"; k2 = mn != "mesh4"))
end
write(OUTF, join(lines, "\n") * "\n")
