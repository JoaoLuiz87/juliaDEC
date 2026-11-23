# ============================================================================
#  ss_abstol.jl -- a variant: impose only the skew directions that are not already negligible
#
#      julia scripts/archive/ss_abstol.jl        (writes ss_data/abstol.txt)
#
#  rowspace keeps singular values of S_P above max(tol σ₁, abstol ‖N_P‖‖R_P‖).  abstol = 0 is the guide's star
#  (exact on affine fields).  abstol > 0: fewer constraints, the patch matrix uses sym(A_P), H stays exactly
#  symmetric, and the affine exactness error equals the neglected skew.  Same metrics as ss_objective.jl.
# ============================================================================
include(joinpath(@__DIR__, "..", "..", "SymStar.jl")); using .SymStar
include(joinpath(@__DIR__, "ss_mesh.jl"))
include(joinpath(@__DIR__, "ss_objective_lib.jl"))
using LinearAlgebra, SparseArrays, Printf
const OUT = joinpath(@__DIR__, "..", "..", "output", "ss_data")
lines = String[]; logp(s) = (println(s); flush(stdout); push!(lines, s))
logp(@sprintf("%-11s %6s %7s %6s %8s %8s %7s %9s %9s %9s %9s %9s %9s %7s", "mesh", "E", "abstol", "rank", "max|θ|", "max C_e",
              "mean C", "skew", "affine", "flux max", "flux rms", "p max", "flux H⁻¹", "μ"))
function sweep(name, X, T, period, h)
    g = edge_geometry(X, T, period)
    kx, ky = 2pi / period[1], 2pi / period[2]
    u(x) = [sin(ky * x[2]) + 0.4cos(kx * x[1]), cos(kx * x[1] + ky * x[2])]
    α, β = reductions(g, u); Hd = spdiagm(g.lde ./ g.le)
    for at in (0.0, 1e-4, 1e-3, 1e-2)
        H, info = symmetric_star(g.me, g.t, g.le, g.lde, g.delta, period, h; rho = 2.0, abstol = at, verbose = false)
        C, _ = row_constants(H, g.me, g.t, g.le, g.lde, period, h); ε = abs.(H * α - β) ./ g.lde
        μ = try smallest_eigenvalue(H; D = Hd) catch; NaN end
        pe, fe = isnan(μ) || μ <= 0 ? (NaN, NaN) : poisson_errors(g, H, period)
        logp(@sprintf("%-11s %6d %7.0e %6.2f %8.3f %8.2f %7.2f %9.1e %9.1e %9.2e %9.2e %9.2e %9.2e %7.3f", name, length(g.le), at,
                      15 - info.null_mean + info.patch_mean - 15, max(abs(info.theta_min), abs(info.theta_max)), maximum(C), sum(C) / length(C),
                      info.patch_skew_max, check_affine(H, g.me, g.t, g.le, g.lde, g.delta, period, h),
                      maximum(ε), sqrt(sum(abs2, ε) / length(ε)), pe, fe, μ))
    end
end
for n in (20, 40, 80)
    sweep("lattice $n", porting_mesh(n)...)
end
for mn in ("mesh3", "mesh4", "mesh5")
    sweep("scvt $mn", scvt_mesh(mn)...)
end
write(joinpath(OUT, "abstol.txt"), join(lines, "\n") * "\n")
