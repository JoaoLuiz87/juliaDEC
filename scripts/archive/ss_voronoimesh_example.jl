# ============================================================================
#  ss_voronoimesh_example.jl -- using the symmetric star inside juliaDEC (worked example of the note)
#
#      julia scripts/archive/ss_voronoimesh_example.jl
#
#  1. build H on a VoronoiMesh in the mesh's own numbering;  2. check it against juliaDEC's reductions
#  (reduce_primal_edges, reduce_dual_edges ∘ star_field);  3. solve Poisson with d0_matrix_primal;
#  4. Darcy-type use: fluxes H d0 p and the divergence d0ᵀ H d0 p.
# ============================================================================
using VoronoiMeshes, TensorsLite, TensorsLiteGeometry, SmallCollections, LinearAlgebra, NCDatasets
using StaticArrays, SparseArrays, Printf
Base.similar(bc::Base.Broadcast.Broadcasted, ::Type{Union{}}) = Array{Union{}}(undef, length.(axes(bc)))
const ROOT = joinpath(@__DIR__, "..", ".."); cd(ROOT)
include(joinpath(ROOT, "src", "juliaDEC.jl")); include(joinpath(ROOT, "src", "hodge_stars.jl"))
include(joinpath(ROOT, "SymStar.jl")); include(joinpath(ROOT, "symstar_voronoi.jl"))

mesh = VoronoiMesh("meshes/mesh4.nc")                       # relative path (NetCDF and non-ASCII paths)
println("1. build")
H, info, g = symmetric_star_mesh(mesh; rho = 2.0)
@printf("   affine check %.1e, smallest eigenvalue of Hd^-1/2 H Hd^-1/2 = %.3f\n",
        check_affine(H, g.me, g.t, g.le, g.lde, g.delta, g.period, g.h),
        smallest_eigenvalue(H; D = spdiagm(g.lde ./ g.le)))

println("2. against juliaDEC's reductions")
u(x) = [sin(2pi * x[2]) + 0.4cos(2pi * x[1]), cos(2pi * (x[1] + x[2]))]
αj = reduce_primal_edges(mesh, u).values                     # juliaDEC: circulations on Delaunay edges
βj = reduce_dual_edges(mesh, star_field(u)).values           # juliaDEC: exact fluxes (what a star should return)
α, β = reductions(g, u)
@printf("   |α_juliaDEC - α_SymStar| = %.1e,  |β_juliaDEC - β_SymStar| = %.1e   (same numbering, same signs)\n",
        maximum(abs.(αj - α)), maximum(abs.(βj - β)))
Hd = spdiagm(g.lde ./ g.le)
for (lbl, S) in (("diagonal", Hd), ("symmetric ρ=2", H))
    ε = abs.(S * αj - βj) ./ g.lde
    @printf("   %-14s flux error max %.2e  rms %.2e\n", lbl, maximum(ε), sqrt(sum(abs2, ε) / length(ε)))
end

println("3. Poisson  d0ᵀ H d0 p = d0ᵀ β(∇p)")
d0 = sparse(Float64.(d0_matrix_primal(mesh)))
@printf("   d0_matrix_primal == geometry incidence: %s\n", d0 == g.d0)
p(x) = sin(2pi * x[1]) * cos(2pi * x[2])
gp(x) = [2pi * cos(2pi * x[1]) * cos(2pi * x[2]), -2pi * sin(2pi * x[1]) * sin(2pi * x[2])]
βg = reduce_dual_edges(mesh, star_field(gp)).values
pI = [p(_xy(x)) for x in mesh.cells.position]
for (lbl, S) in (("diagonal", Hd), ("symmetric ρ=2", H))
    L = d0' * S * d0; n = size(L, 1)
    ph = (lu([L ones(n); ones(1, n) 0.0]) \ vcat(d0' * βg, 0.0))[1:n]
    ph .+= sum(pI - ph) / n
    flux = S * d0 * ph                                         # discrete Darcy/Poisson fluxes on Voronoi edges
    @printf("   %-14s nodal error %.2e  flux error max %.2e  (div of the flux - load: %.1e)\n", lbl,
            maximum(abs.(ph - pI)) / maximum(abs.(pI)), maximum(abs.(flux - βg) ./ g.lde) / maximum(abs.(βg) ./ g.lde),
            maximum(abs.(d0' * flux - d0' * βg)))
end
