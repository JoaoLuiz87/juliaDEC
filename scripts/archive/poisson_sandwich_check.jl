# ============================================================================
#  poisson_sandwich_check.jl -- Strang identity for a Galerkin (Hiptmair) Hodge
#
#      julia scripts/archive/poisson_sandwich_check.jl
#
#  DEC Poisson  d0ᵀ H d0 p_h = d0ᵀ β(∇p)   (the right-hand side is the exact Voronoi-cell flux
#  balance ∫_V Δp, no quadrature in the scheme).  With the Hodge error ε = H α(∇p) - β(∇p),
#       d0ᵀ H d0 (p_h - p_I) = - d0ᵀ ε ,   hence   ‖d0 p_h - α(∇p)‖_H = sup_q (d0 q)ᵀ ε / ‖d0 q‖_H .
#  The scheme sees only the part of the Hodge error tested against gradients.
#  Stars: the [P1]² and [P2]² sandwiches, and DEC's diagonal star W = diag(|ẽ|/|e|).
#  patch: the same dual norm for a CONSTANT field a, i.e. the linear patch test d0ᵀ H α(a) = 0
#         (= closure of the induced dual cells; automatic for DEC, by Voronoi closure).
# ============================================================================
using VoronoiMeshes, TensorsLite, TensorsLiteGeometry, SmallCollections, LinearAlgebra, NCDatasets
using StaticArrays, SparseArrays, Printf
Base.similar(bc::Base.Broadcast.Broadcasted, ::Type{Union{}}) = Array{Union{}}(undef, length.(axes(bc)))
const ROOT = joinpath(@__DIR__, "..", "..")
cd(ROOT)
include(joinpath(ROOT, "src", "juliaDEC.jl")); include(joinpath(ROOT, "src", "hodge_stars.jl"))

α(m, w) = reduce_primal_edges(m, w).values
β(m, w) = reduce_dual_edges(m, star_field(w)).values
d0m(m) = sparse(Float64.(d0_matrix_primal(m)))

"‖x‖_H and the H-dual norm of ε on range(d0):  sqrt(εᵀ d0 (d0ᵀ H d0)⁺ d0ᵀ ε)"
Hn(H, x) = sqrt(abs(dot(x, H * x)))
function solveL(L, b)
    n = size(L, 1); o = ones(n) / sqrt(n)
    return (L + o * o') \ b                       # L is singular on constants only (torus)
end

@printf("%-14s %8s %-5s %14s %14s %12s %14s\n", "mesh", "h", "star", "‖grad err‖_H", "dual norm", "identity", "patch (a)")
for mn in ("mesh2_regular", "mesh3_regular", "mesh4_regular", "mesh5_regular", "mesh2", "mesh3", "mesh4", "mesh5")
    m = VoronoiMesh("meshes/$mn.nc"); Lx, Ly = m.x_period, m.y_period
    gp(x) = [2pi / Lx * cos(2pi * x[1] / Lx) * cos(2pi * x[2] / Ly), -2pi / Ly * sin(2pi * x[1] / Lx) * sin(2pi * x[2] / Ly)]
    d0 = d0m(m); ag = α(m, gp); bg = β(m, gp); rhs = d0' * bg
    h = maximum(edge_lengths(m, true))
    stars = (("P1", hodge_star_sandwich_matrix(m; only_linear = true)),
             ("P2", hodge_star_sandwich_matrix(m; only_linear = false)),
             ("DEC", spdiagm(edge_lengths(m, false) ./ edge_lengths(m, true))))
    for (lbl, H) in stars
        L = d0' * H * d0
        ph = solveL(L, rhs)
        err = Hn(H, d0 * ph - ag) / Hn(H, ag)
        ε = H * ag - bg; r = d0' * ε
        dn = sqrt(abs(dot(r, solveL(L, r)))) / Hn(H, ag)
        aa = α(m, x -> [1.0, 0.7]); ra = d0' * (H * aa - β(m, x -> [1.0, 0.7]))
        pa = sqrt(abs(dot(ra, solveL(L, ra)))) / Hn(H, aa)
        @printf("%-14s %8.4f %-5s %14.3e %14.3e %12.1e %14.3e\n", mn, h, lbl, err, dn, abs(err - dn) / dn, pa)
        flush(stdout)
    end
    @printf("   (periods %.4f x %.4f)\n", Lx, Ly)
end
