
#########################################################################
# VoronoiMeshes — VTU save/read example
#
# This example demonstrates how to:
#   - Construct a centroidal Voronoi mesh on a periodic domain
#   - Export the Voronoi (dual) and Delaunay (triangulation) grids to VTU files
#   - Use the high-level `save` function for automatic format selection
#   - Read a VoronoiMesh from a vtu file
#
# Requirements:
#   - VoronoiMeshes.jl (with extensions)
#   - WriteVTK, ReadVTK (for VTU I/O)
#   - DelaunayTriangulation, TensorsLite, GLMakie, NCDatasets (optional)
#########################################################################

# Basic modules for mesh construction and manipulation
using VoronoiMeshes
using DelaunayTriangulation
using TensorsLite
using LinearAlgebra


# Required for VTU export/import (loads extensions automatically)
using WriteVTK     # For saving meshes in VTU format
using ReadVTK     # For reading meshes in VTU format

# Create a centroidal Voronoi mesh with 40 cells on a 1×1 periodic domain
mesh = VoronoiMesh(20, 1.0, 1.0)

# --- Export Voronoi (dual) grid to VTU ---
# This will handle periodic ghost vertices automatically
save_voronoi_to_vtu("mesh_voronoi.vtu", mesh)

# --- Export Delaunay triangulation (primal) grid to VTU ---
save_triangulation_to_vtu("mesh_triangulation.vtu", mesh)

# --- Use high-level save function (auto-selects format by extension) ---
# This will call the appropriate extension based on the file extension
# This function will actually create two vtu files named mesh_vor.vtu and mesh_tri.vtu
# for the Voronoi grid and the Dual triangular grid
VoronoiMeshes.save("mesh.vtu", mesh)

# This function will actually read two files named "mesh_vor.vtu" and "mesh_tri.vtu"
# In order to construct the VoronoiMesh
mesh_read = VoronoiMesh("mesh.vtu")

