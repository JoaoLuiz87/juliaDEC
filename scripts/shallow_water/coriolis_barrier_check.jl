# ============================================================================
#  coriolis_barrier_check.jl -- is there a Coriolis operator P that is energy-neutral (H_s P antisymmetric) and exact on smooth fields?
#  With a symmetric star H exact on u, w (H R u = R̃⋆u) and P R u = R(Ju), antisymmetry of H P gives
#      k(u,w) = R̃⋆u · R(Jw) + R̃⋆w · R(Ju) = 0.
#  k is computed from exact cochains (Gauss rules on e and ẽ), divided by |Ω| ‖u‖∞ ‖w‖∞.
#      julia scripts/shallow_water/coriolis_barrier_check.jl        -> output/ts_data/coriolis_barrier.txt
# ============================================================================
include(joinpath(@__DIR__, "sw_linear_three.jl"))
OUTK = String[]; sk(s) = (println(s); flush(stdout); push!(OUTK, s))
const κ = 2π / LDOM
u1(x, t) = [cos(κ * x[1]) * cos(κ * x[2]), -sin(κ * x[1]) * sin(κ * x[2])]                       # gradient-like
u2(x, t) = [-2 * sin(2κ * x[1]) - 0.5 * cos(κ * (x[1] + x[2])), 0.5 * cos(κ * (x[1] + x[2]))]
u3(x, t) = [sin(κ * (2x[1] - x[2])), 0.7 * cos(κ * x[1])]
F = (u1, u2, u3)
sup(u) = maximum(norm(u((a, b) .* LDOM, 0.0)) for a in 0:0.02:1, b in 0:0.02:1)
sk("k(u,w)/(|Ω|‖u‖‖w‖) for (1,1), (1,2), (2,2), (2,3), (3,3);  [divided by h²]")
for mn in ("mesh3", "mesh4", "mesh5", "mesh6", "mesh7", "mesh3_regular", "mesh5_regular")
    mesh = VoronoiMesh("meshes/$mn.nc"); N = mesh.cells.n; h = 1 / sqrt(N)
    O = sw_ops(mesh, spdiagm(0 => [mesh.edges.lengthDual[e] for e in 1:mesh.edges.n]) * 0 + spdiagm(0 => [1.0 for e in 1:mesh.edges.n]))
    C = [cochains(O, (x, t) -> 0.0, f, 0.0) for f in F]; s = [sup(f) for f in F]
    k(i, j) = (dot(C[i][3], C[j][4]) + dot(C[j][3], C[i][4])) / (O.L^2 * s[i] * s[j])
    sk(@sprintf("%-14s N %6d  ", mn, N) * join([@sprintf("%+.2e [%+.1e]", k(i, j), k(i, j) / h^2) for (i, j) in ((1, 1), (1, 2), (2, 2), (2, 3), (3, 3))], "  "))
end
write(joinpath(TSDIR, "coriolis_barrier.txt"), join(OUTK, "\n") * "\n")
