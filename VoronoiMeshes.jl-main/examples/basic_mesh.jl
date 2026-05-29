#########################################################################
# VoronoiMeshes — basic example
#
# Purpose
#   Create a simple centroidal Voronoi mesh on a periodic 2D domain and
#   test simple functionalities of VoronoiMeshes
#
# Requirements
#   - Julia with the project environment activated (run with `julia --project=.`)
#   - Optional plotting backends: `GLMakie` or `CairoMakie` for rendering and
#     saving figures. If not available, the plotting lines can be commented out.
#   - Optional `NCDatasets` for saving/loading meshes in NetCDF format (the
#     package provides `VoronoiMeshes.save` which delegates to the extension).
#   - Optional `DelaunayTriangulation` required by constructors that generate
#     meshes from random points (this example uses the package-provided
#     constructor which will require that package to be available).
#
# Usage
#   Run from the repository root so the package environment is used:
#     julia --project=. examples/basic_mesh.jl
#
# Output
#   - `mesh_primal.png` : PNG image of the primal Voronoi mesh (if Makie present)
#   - `mesh_dual.png`   : PNG image of the dual (Delaunay) mesh (if Makie present)
#   - `mesh_primal_and_dual.png` : PNG image of both primal and dual meshes (if Makie present)
#   - `mesh.nc`         : NetCDF file containing the mesh (if NCDatasets present)
#
#########################################################################
using VoronoiMeshes
using DelaunayTriangulation
using TensorsLite
using LinearAlgebra

using GLMakie  # Optional, for plotting if Makie is installed. Could also be any other Makie backend
using NCDatasets # Optional, for saving/loading meshes in NetCDF format

# Create a centroidal Voronoi mesh with 20 cells on a 1×1 periodic domain
mesh = VoronoiMesh(20, 1.0, 1.0)

# Print a summary of the mesh
println(mesh)


# Basic inspection of the mesh object:
println(" Cells: ", mesh.cells.n)
for (i, p) in enumerate(mesh.cells.position)
    println("Cell $i : ")
    println("  Coordinates: ($(p[1]), $(p[2]))")
    print("  Neighbor cells: ")
    for j in mesh.cells.cells[i]
        print(" $j ")
    end
    println()
    print("  Cell edges: ")
    for j in mesh.cells.edges[i]
        print(" $j")
    end
    println()
end


# Plot the mesh (requires some Makie backend)
println("Plotting the mesh ...")
plt = plotmesh(mesh)
GLMakie.save("mesh_primal.png", plt)

# Plot the dual mesh
plt1 = plotdualmesh(mesh)
GLMakie.save("mesh_dual.png", plt1)

# Plot the Voronoi mesh with red lines of with=2
fig_both = Figure()
ax = Axis(fig_both[1,1])

# Plot the Voronoi mesh with red lines of with=2
plotmesh!(ax, mesh, color=:red, linewidth=2)

# Plot the dual mesh with blue dashed lines of width=0.75, and use dotted lines for the ghost edges.
plotdualmesh!(ax, mesh, color=:blue, linewith=0.75, edgelinestyle=:dash, ghostedgelinestyle=:dot)
GLMakie.save("mesh_primal_and_dual.png", fig_both)

println("Mesh plots saved to mesh_primal.png, mesh_dual.png, and mesh_primal_and_dual.png")

# Save the mesh to a NetCDF file (requires NCDatasets)
println("Saving the mesh to mesh.nc ...")
VoronoiMeshes.save("mesh.nc", mesh)
println("Mesh saved to mesh.nc")

# Uncomment below for other examples of mesh creation

# Create a mesh from a given set of generator points (VecArray of x,y coordinates):
#generators = VecArray(x = rand(10), y = rand(10))
#mesh2 = VoronoiMesh(generators, 1.0, 1.0)
#println(mesh2)

# Create a regular hexagonal planar mesh (utility provided in the Delaunay extension):
# dx ~ target cell spacing
#hexmesh = create_planar_hex_mesh(1.0, 1.0, 0.05)
#println(hexmesh)
