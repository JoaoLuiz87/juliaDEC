# The meshes of the experiments: SCVT family (answer y) or regular lattices (answer n).  Each level adds the edge midpoints as generators and runs Lloyd.
using NCDatasets, VoronoiMeshes, DelaunayTriangulation, TensorsLite, TensorsLiteGeometry
using Random
Random.seed!(1111)

println("Distorted mesh? (y/n)")
dist = readline() == "y"

if dist
    mesh1 = VoronoiMesh(16, 1.0, 1.0, density = x-> 1.0, max_iter=10000)
    VoronoiMeshes.save("mesh1.nc", mesh1)

    mesh2 = VoronoiMesh(vcat(mesh1.cells.position, mesh1.edges.position), 1.0, 1.0, max_iter=20000)
    VoronoiMeshes.save("mesh2.nc", mesh2)

    mesh3 = VoronoiMesh(vcat(mesh2.cells.position, mesh2.edges.position), 1.0, 1.0, max_iter=20000)
    VoronoiMeshes.save("mesh3.nc", mesh3)

    mesh4 = VoronoiMesh(vcat(mesh3.cells.position, mesh3.edges.position), 1.0, 1.0, max_iter=20000)
    VoronoiMeshes.save("mesh4.nc", mesh4)

    mesh5 = VoronoiMesh(vcat(mesh4.cells.position, mesh4.edges.position), 1.0, 1.0, max_iter=20000)
    VoronoiMeshes.save("mesh5.nc", mesh5)

    mesh6 = VoronoiMesh(vcat(mesh5.cells.position, mesh5.edges.position), 1.0, 1.0, max_iter=10000)
    VoronoiMeshes.save("mesh6.nc", mesh6)

    mesh7 = VoronoiMesh(vcat(mesh6.cells.position, mesh6.edges.position), 1.0, 1.0, max_iter=3000)
    VoronoiMeshes.save("mesh7.nc", mesh7)

else
    mesh1 = VoronoiMesh(12, 1.0, 1.0)
    VoronoiMeshes.save("mesh1_regular.nc", mesh1)

    mesh2 = VoronoiMesh(vcat(mesh1.cells.position, mesh1.edges.position), 1.0, 1.0)
    VoronoiMeshes.save("mesh2_regular.nc", mesh2)

    mesh3 = VoronoiMesh(vcat(mesh2.cells.position, mesh2.edges.position), 1.0, 1.0)
    VoronoiMeshes.save("mesh3_regular.nc", mesh3)

    mesh4 = VoronoiMesh(vcat(mesh3.cells.position, mesh3.edges.position), 1.0, 1.0)
    VoronoiMeshes.save("mesh4_regular.nc", mesh4)

    mesh5 = VoronoiMesh(vcat(mesh4.cells.position, mesh4.edges.position), 1.0, 1.0)
    VoronoiMeshes.save("mesh5_regular.nc", mesh5)

    mesh6 = VoronoiMesh(vcat(mesh5.cells.position, mesh5.edges.position), 1.0, 1.0)
    VoronoiMeshes.save("mesh6_regular.nc", mesh6)

    mesh7 = VoronoiMesh(vcat(mesh6.cells.position, mesh6.edges.position), 1.0, 1.0)
    VoronoiMeshes.save("mesh7_regular.nc", mesh7)

    mesh8 = VoronoiMesh(vcat(mesh7.cells.position, mesh7.edges.position), 1.0, 1.0)
    VoronoiMeshes.save("mesh8_regular.nc", mesh8)

    mesh9 = VoronoiMesh(vcat(mesh8.cells.position, mesh8.edges.position), 1.0, 1.0)
    VoronoiMeshes.save("mesh9_regular.nc", mesh9)

    mesh10 = VoronoiMesh(vcat(mesh9.cells.position, mesh9.edges.position), 1.0, 1.0)
    VoronoiMeshes.save("mesh10_regular.nc", mesh10)

    println("done")
end

println("done")
