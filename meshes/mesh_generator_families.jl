# ============================================================================
#  mesh_generator_families.jl
#
#  Two extra periodic Delaunay/Voronoi families on [0,1]^2, to probe how the
#  symmetry/consistency obstruction depends on the mesh:
#
#    scvt_var<k>.nc  variable-resolution SCVT: Lloyd iterations with a smooth
#                    periodic density bump (h varies by ~2x across the domain).
#                    Refinement as in mesh_generator.jl: new generators =
#                    old cells + old edge midpoints, then Lloyd again.
#
#    pert<k>.nc      perturbed lattice WITHOUT Lloyd: the regular family
#                    mesh<k>_regular with every generator displaced by a random
#                    vector of length eps*h (eps fixed).  The irregularity does
#                    NOT decay under refinement.
#
#      cd juliaDEC/meshes ; julia mesh_generator_families.jl
# ============================================================================
using NCDatasets, VoronoiMeshes, DelaunayTriangulation, TensorsLite, TensorsLiteGeometry
using Random, LinearAlgebra
Random.seed!(2024)
cd(@__DIR__)

# ---- variable-density SCVT
bump(x) = 1 + 15 * exp(-(sin(pi * (x.x - 0.5))^2 + sin(pi * (x.y - 0.5))^2) / 0.05)

if !isfile("scvt_var5.nc")
m = VoronoiMesh(16, 1.0, 1.0; density = bump, max_iter = 20000)
VoronoiMeshes.save("scvt_var1.nc", m); println("scvt_var1: ", m.cells.n, " cells")
for k in 2:5
    global m = VoronoiMesh(vcat(m.cells.position, m.edges.position), 1.0, 1.0;
                           density = bump, max_iter = 20000)
    VoronoiMeshes.save("scvt_var$k.nc", m); println("scvt_var$k: ", m.cells.n, " cells")
end
end

# ---- perturbed regular lattice, no Lloyd iterations
const EPS = 0.12
for k in 1:6
    r = VoronoiMesh("mesh$(k)_regular.nc")
    h = sqrt(2 / (sqrt(3) * r.cells.n))                # triangular-lattice spacing
    pts = similar(r.cells.position)
    for i in eachindex(pts)
        th = 2pi * rand(); a = EPS * h * sqrt(rand())
        p = r.cells.position[i]
        pts[i] = typeof(p)(mod(p.x + a * cos(th), 1.0), mod(p.y + a * sin(th), 1.0))
    end
    pm = VoronoiMesh(pts, 1.0, 1.0; max_iter = 0)
    VoronoiMeshes.save("pert$k.nc", pm); println("pert$k: ", pm.cells.n, " cells")
end
println("done")
