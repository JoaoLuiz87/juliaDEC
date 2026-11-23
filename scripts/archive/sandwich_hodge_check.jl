# ============================================================================
#  sandwich_hodge_check.jl -- the edge-based sandwich read as a (Galerkin/Hiptmair) Hodge
#
#      julia scripts/archive/sandwich_hodge_check.jl
#
#  H = Σ_e S_eᵀ (Π_eᵀ M_e Π_e + stab_e) S_e,  Π_e = A_e⁺ (reduction on the stencil),
#  M_e = mass on the diamond D_e  (hodge_star_sandwich_matrix in hodge_stars.jl).
#  For the [P1]² and [P2]² sandwiches on the lattice and SCVT meshes 2-4:
#    energy  : z'Hz - ∫|v|² for a constant v                 (energy consistency)
#    flux    : H α(v) - β(v) for a constant v and a smooth v  (Hiptmair's Hodge error, W-norm)
#    weak    : the parts of the flux error seen by the Poisson problem (gradients) and by
#              Darcy (discretely divergence-free circulations), dual norms as in the lift paper.
# ============================================================================
using VoronoiMeshes, TensorsLite, TensorsLiteGeometry, SmallCollections, LinearAlgebra, NCDatasets
using StaticArrays, SparseArrays, Printf
Base.similar(bc::Base.Broadcast.Broadcasted, ::Type{Union{}}) = Array{Union{}}(undef, length.(axes(bc)))
const ROOT = joinpath(@__DIR__, "..", "..")
cd(ROOT)
include(joinpath(ROOT, "src", "juliaDEC.jl")); include(joinpath(ROOT, "src", "hodge_stars.jl"))

α(m, w) = reduce_primal_edges(m, w).values
β(m, w) = reduce_dual_edges(m, star_field(w)).values
ω(m) = edge_lengths(m, false) ./ edge_lengths(m, true)          # |ẽ|/|e|
d0m(m) = sparse(Float64.(d0_matrix_primal(m))); D1m(m) = sparse(Float64.(d1_matrix_primal(m)))
Vs(x) = [sin(2pi * x[2]), cos(2pi * x[1])]
"dual norm sup_{z ∈ span(Z)} zᵀε / ‖z‖_W with ‖z‖_W² = Σ ω z²"
function dualnorm(Z, ε, w)
    Zw = Diagonal(sqrt.(w)) * Z                     # coordinates in the W-metric
    F = qr(Matrix(Zw)); Q = Matrix(F.Q)[:, 1:rank(Zw)]
    return norm(Q' * (ε ./ sqrt.(w)))
end

@printf("%-14s %-6s %12s %14s %14s %16s %16s\n", "mesh", "star", "energy(a)", "flux(a) rel", "flux(V) rel", "grad part of ε", "div-free part")
for mn in ("mesh2_regular", "mesh3_regular", "mesh4_regular", "mesh2", "mesh3", "mesh4")
    m = VoronoiMesh("meshes/$mn.nc"); w = ω(m)
    for (lbl, ol) in (("P1", true), ("P2", false))
        H = hodge_star_sandwich_matrix(m; only_linear = ol)
        a = [1.0, 0.7]; za = α(m, x -> a)
        en = abs(dot(za, H * za) - sum(abs2, a) * sum(edge_lengths(m, true) .* edge_lengths(m, false)) / 2) /
             (sum(abs2, a) * sum(edge_lengths(m, true) .* edge_lengths(m, false)) / 2)
        εa = H * za - β(m, x -> a); εV = H * α(m, Vs) - β(m, Vs)
        fa = sqrt(sum(εa .^ 2 ./ w)) / sqrt(sum(β(m, x -> a) .^ 2 ./ w))
        fV = sqrt(sum(εV .^ 2 ./ w)) / sqrt(sum(β(m, Vs) .^ 2 ./ w))
        nV = sqrt(sum(β(m, Vs) .^ 2 ./ w))
        g = dualnorm(d0m(m), εV, w) / nV                       # part of ε seen by gradients (Poisson)
        dfree = dualnorm(nullspace(Matrix(D1m(m))), εV, w) / nV  # part seen by discretely div-free fields (Darcy)
        @printf("%-14s %-6s %12.1e %14.2e %14.2e %16.2e %16.2e\n", mn, lbl, en, fa, fV, g, dfree)
        flush(stdout)
    end
end
