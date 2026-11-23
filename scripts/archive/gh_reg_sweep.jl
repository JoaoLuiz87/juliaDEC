# ============================================================================
#  gh_reg_sweep.jl -- exact versus approximate moment conditions: does the correction Z stay bounded?
#
#      julia scripts/archive/gh_reg_sweep.jl        (writes gh_data/reg_sweep.txt)
#
#  The minimal-change correction solves (G + ε I) μ = r.  ε → 0 gives exact [P1]² consistency; a larger ε
#  trades a small moment residual for a smaller correction.  For several ε and refinement levels we report the
#  moment residual, row errors on constants / affine fields, the deviation from P⁺, Λ, the Hodge error of a
#  smooth field and the Poisson gradient error.
# ============================================================================
include(joinpath(@__DIR__, "gh_lab.jl"))
const OUTF = joinpath(@__DIR__, "..", "..", "output", "gh_data", "reg_sweep.txt")

function sweep(M, mname, cname; regs = (1e-13, 1e-10, 1e-8, 1e-6))
    geos = [egeo(M, e, M.X[M.E[e][1]]) for e in eachindex(M.E)]
    Px, Py = M.Px, M.Py
    u(x) = [sin(2pi * x[2] / Py) + 0.5cos(2pi * (x[1] / Px + x[2] / Py)), cos(2pi * x[1] / Px)]
    p(x) = sin(2pi * x[1] / Px) * cos(2pi * x[2] / Py)
    gp(x) = [2pi / Px * cos(2pi * x[1] / Px) * cos(2pi * x[2] / Py), -2pi / Py * sin(2pi * x[1] / Px) * sin(2pi * x[2] / Py)]
    cells = cname == "tri" ? tri_cells(M) : dia_cells(M)
    D = celldata(M, cells, 1)
    hmax = maximum(g.L for g in geos); out = String[]
    for reg in regs
        Rs, info = constrained_R(M, cells, D, geos, 1; reg = reg)
        H = assemble_H(M, cells, D, Rs)
        r0 = row_test(M, H, geos, 0); r1 = row_test(M, H, geos, 1)
        fe = flux_err(M, geos, H, u); pe, pd = poisson(M, geos, H, p, gp)
        s = @sprintf("%-10s %7.4f %-4s %7.0e %9.1e %9.1e %9.1e %6.2f %6.2f %9.2e %9.2e %9.1e",
                     mname, hmax, cname, reg, info.resid, r0, r1, info.dev, Λnorm(H), fe, pe, pd)
        println(s); flush(stdout); push!(out, s)
    end
    out
end

hdr = @sprintf("%-10s %7s %-4s %7s %9s %9s %9s %6s %6s %9s %9s %9s", "mesh", "h", "cell", "ε", "moment", "row P0",
               "row P1", "dev", "Λ", "flux(u)", "Poisson", "patch")
println(hdr); lines = [hdr]
for N in (16, 32, 64), cname in ("tri", "dia")
    append!(lines, sweep(latmesh(N, "mapped"), "mapped$N", cname))
end
for mn in ("mesh3", "mesh4", "mesh5"), cname in ("tri", "dia")
    append!(lines, sweep(readmesh(joinpath(@__DIR__, "..", "..", "output", "gh_data", "$mn.txt")), "scvt-$mn", cname))
end
write(OUTF, join(lines, "\n") * "\n")
