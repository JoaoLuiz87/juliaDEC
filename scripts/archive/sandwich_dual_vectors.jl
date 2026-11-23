# ============================================================================
#  sandwich_dual_vectors.jl -- the dual vectors induced by the edge-based sandwich
#
#      julia scripts/archive/sandwich_dual_vectors.jl
#
#  For a constant field a, row e of H α(a) is g_e · a with g_e = ∫ Φ_e (the constant moment of
#  the dual object Φ_e of the sandwich).  DEC's Voronoi flux is β_e(a) = b_e · a.  We compare
#  the two vectors edge by edge: length ratio |g_e|/|b_e| and the angle between them.
# ============================================================================
using VoronoiMeshes, TensorsLite, TensorsLiteGeometry, SmallCollections, LinearAlgebra, NCDatasets
using StaticArrays, SparseArrays, Printf, Statistics
Base.similar(bc::Base.Broadcast.Broadcasted, ::Type{Union{}}) = Array{Union{}}(undef, length.(axes(bc)))
const ROOT = joinpath(@__DIR__, "..", "..")
cd(ROOT)
include(joinpath(ROOT, "src", "juliaDEC.jl")); include(joinpath(ROOT, "src", "hodge_stars.jl"))

α(m, w) = reduce_primal_edges(m, w).values
β(m, w) = reduce_dual_edges(m, star_field(w)).values

@printf("%-14s %-4s %8s %8s | %10s %10s %10s | %9s %9s\n", "mesh", "star", "|e| min", "|e| max",
        "ratio min", "ratio mean", "ratio max", "angle avg", "angle max")
for mn in ("mesh2_regular", "mesh3_regular", "mesh2", "mesh3")
    m = VoronoiMesh("meshes/$mn.nc")
    le = edge_lengths(m, true)
    for (lbl, ol) in (("P1", true), ("P2", false))
        H = hodge_star_sandwich_matrix(m; only_linear = ol)
        gx = H * α(m, x -> [1.0, 0.0]); gy = H * α(m, x -> [0.0, 1.0])
        bx = β(m, x -> [1.0, 0.0]);     by = β(m, x -> [0.0, 1.0])
        ratio = [hypot(gx[e], gy[e]) / hypot(bx[e], by[e]) for e in eachindex(gx)]
        ang = [acosd(clamp((gx[e] * bx[e] + gy[e] * by[e]) / (hypot(gx[e], gy[e]) * hypot(bx[e], by[e])), -1, 1))
               for e in eachindex(gx)]
        @printf("%-14s %-4s %8.4f %8.4f | %10.3f %10.3f %10.3f | %8.2f° %8.2f°\n", mn, lbl, minimum(le), maximum(le),
                minimum(ratio), mean(ratio), maximum(ratio), mean(ang), maximum(ang))
        flush(stdout)
    end
end
