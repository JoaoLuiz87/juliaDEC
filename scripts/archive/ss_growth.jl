# ============================================================================
#  ss_growth.jl -- growth of the shares under refinement, and Tikhonov regularisation of the share solve
#
#      julia scripts/archive/ss_growth.jl        (writes ss_data/growth.txt)
#
#  (G + ε I) λ = 1, θ_P = Π0_P λ|_P.  ε → 0 is the guide's minimum-norm solution (exact partition of unity);
#  ε > 0 bounds the near-kernel components of λ at the price of a partition error ≈ ε‖λ‖, which is the
#  relative row error on affine fields.  Meshes: the porting lattice refined (n = 10..80; its perturbation is a
#  smooth pattern in index space) and the juliaDEC SCVT meshes 2-5.  ρ = 2.
# ============================================================================
include(joinpath(@__DIR__, "..", "..", "SymStar.jl")); using .SymStar
include(joinpath(@__DIR__, "ss_mesh.jl"))
using LinearAlgebra, SparseArrays, Printf
const OUT = joinpath(@__DIR__, "..", "..", "output", "ss_data"); mkpath(OUT)
lines = String[]; logp(s) = (println(s); flush(stdout); push!(lines, s))
logp(@sprintf("%-11s %6s %7s %9s %8s %8s %7s %9s %9s %9s %7s", "mesh", "E", "ε", "partition", "max|θ|", "max C_e", "mean C", "affine", "flux max", "flux rms", "μ"))

function run(name, X, T, period, h; regs = (1e-13, 1e-6, 1e-4, 1e-3, 1e-2))
    g = edge_geometry(X, T, period)
    kx, ky = 2pi / period[1], 2pi / period[2]
    u(x) = [sin(ky * x[2]) + 0.4cos(kx * x[1]), cos(kx * x[1] + ky * x[2])]
    α, β = reductions(g, u)
    Hd = spdiagm(g.lde ./ g.le)
    for reg in regs
        H, info = symmetric_star(g.me, g.t, g.le, g.lde, g.delta, period, h; rho = 2.0, reg = reg, verbose = false)
        C, _ = row_constants(H, g.me, g.t, g.le, g.lde, period, h)
        ε = abs.(H * α - β) ./ g.lde
        μ = try smallest_eigenvalue(H; D = Hd) catch; NaN end
        logp(@sprintf("%-11s %6d %7.0e %9.1e %8.3f %8.2f %7.2f %9.1e %9.2e %9.2e %7.3f", name, length(g.le), reg, info.partition_err,
                      max(abs(info.theta_min), abs(info.theta_max)), maximum(C), sum(C) / length(C),
                      check_affine(H, g.me, g.t, g.le, g.lde, g.delta, period, h), maximum(ε), sqrt(sum(abs2, ε) / length(ε)), μ))
    end
    ε0 = abs.(Hd * α - β) ./ g.lde
    logp(@sprintf("%-11s %6d %7s %9s %8s %8s %7s %9s %9.2e %9.2e %7s", name, length(g.le), "DEC", "", "", "1", "1", "", maximum(ε0), sqrt(sum(abs2, ε0) / length(ε0)), "1"))
end

for n in (10, 20, 40, 80)
    run("lattice $n", porting_mesh(n)...)
end
for mn in ("mesh3", "mesh4", "mesh5")
    run("scvt $mn", scvt_mesh(mn)...)
end
write(joinpath(OUT, "growth.txt"), join(lines, "\n") * "\n")
