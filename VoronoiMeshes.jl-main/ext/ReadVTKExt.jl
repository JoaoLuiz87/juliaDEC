"""
    ReadVTKExt

Import a VoronoiMesh meshes from VTU (VTK Unstructured Grid) format.
- Provides functions to import Voronoi and Delaunay meshes with periodic ghost vertices.
- Uses ReadVTK for file input
"""
module ReadVTKExt

using VoronoiMeshes, TensorsLite, TensorsLiteGeometry, Zeros, LinearAlgebra
using SmallCollections: FixedVector
using VoronoiMeshes: SmallVectorArray
import VoronoiMeshes: save_triangulation_to_vtu, save_voronoi_to_vtu

using PrecompileTools

using ReadVTK # For saving meshes in VTU format
using ReadVTK.VTKBase


#Wrapper to load a VoronoiMesh from VTU files with a single filename
function VoronoiMeshes.read_from_vtu(filename::String)

    # If a single filename was given, search for the 2 associated grids (vor and tri)
    println("Attempt to read VTU file with base name: ", filename)

    name, ext = Base.Filesystem.splitext(filename)

    if ext == ".vtu" #Save to VTU using VTKExt
        name_vor = name * "_vor" * ext
        name_tri = name * "_tri" * ext
    else
        error("Unsupported file extension: $filename")
    end

    return VoronoiMeshes.read_from_vtu(name_vor, name_tri)
end

# Function to load a VoronoiMesh from VTU files (2 files must be given)
function VoronoiMeshes.read_from_vtu(filename_vor::String, filename_tri::String)

    if isfile(filename_vor) && isfile(filename_tri)
        println("Loading files: ", filename_vor, " and ", filename_tri)
    else
        error("Couldn't find files: ", filename_vor, " and ", filename_tri)
    end

    vtk_tri = VTKFile(filename_tri)
    vtk_vor = VTKFile(filename_vor)

    return read_mesh_from_vtu_data(vtk_vor, vtk_tri)

end

function read_mesh_from_vtu_data(vtk_vor, vtk_tri)

    offsets = get_cells(vtk_vor).offsets
    maxEdges = maximum((offsets[i] - (i == 1 ? 0 : offsets[i-1])) for i in eachindex(offsets))

    return read_mesh_from_vtu_data(Val(maxEdges), vtk_vor, vtk_tri)
end

function read_mesh_from_vtu_data(::Val{maxEdges}, vtk_vor, vtk_tri) where {maxEdges}

    # Extract metadata from field data
    #--------------------
    fd_vor = get_field_data(vtk_vor)

    num_cells = only(get_data(fd_vor["NumCells"]))
    num_vertices = only(get_data(fd_vor["NumVertices"]))

    x_period = only(get_data(fd_vor["XPeriod"]))
    y_period = only(get_data(fd_vor["YPeriod"]))

    fd_tri = get_field_data(vtk_tri)

    num_vertices_from_tri = only(get_data(fd_tri["NumCells"]))

    @assert only(get_data(fd_tri["XPeriod"])) == x_period
    @assert only(get_data(fd_tri["YPeriod"])) == y_period

    # Get Voronoi vertices (these are triangle circumcenters)
    vert_coords_vtk = ReadVTK.get_points(vtk_vor)
    vertices = VecArray(x=vert_coords_vtk[1, 1:num_vertices], y=vert_coords_vtk[2, 1:num_vertices])

    # Get generators (Voronoi centers are triangle vertices)
    gen_coords_vtk = ReadVTK.get_points(vtk_tri)
    generators = VecArray(x=gen_coords_vtk[1, 1:num_vertices_from_tri], y=gen_coords_vtk[2, 1:num_vertices_from_tri])

    # Vertices on cells (convert to SmVecArray)
    #--------------------
    vtk_cells_vor = get_cells(vtk_vor)
    connectivity = vtk_cells_vor.connectivity
    offsets = vtk_cells_vor.offsets
    ranges = [((i == 1 ? 1 : offsets[i-1] + 1):offsets[i]) for i in 1:num_cells]
    verticesOnCell_raw = [Int32.(connectivity[r]) for r in ranges]

    # Ghost indexes are in the end (num_vertices+1:) and have negative sign pointing to original vertex index
    vertex_indx_with_ghosts = get_data(ReadVTK.get_point_data(vtk_vor)["Index"])

    # Convert to SmVecArray and fix ghost indices
    nEdges = Int16[length(v) for v in verticesOnCell_raw]
    verticesOnCell_data = Vector{FixedVector{maxEdges,Int32}}(undef, num_cells)

    for i in 1:num_cells
        cell_verts = verticesOnCell_raw[i]
        n_verts = length(cell_verts)
        tmp = Vector{Int32}(undef, maxEdges)
        for j in 1:n_verts
            idx = cell_verts[j]
            # Check if index points to a ghost cell
            if idx > num_vertices # Ghost vertex - convert index from 0-based to 1-based 
                tmp[j] = vertex_indx_with_ghosts[idx] + 1  # Get correct index from ghost indices
            else # no need to convert, only vertex_indx_with_ghosts has ghost indices with 0-based
                tmp[j] = idx  # Keep positive (regular vertex)
            end
        end
        # Remaining slots are already uninitialized, set to zero
        for j in (n_verts+1):maxEdges
            tmp[j] = Int32(0)
        end
        verticesOnCell_data[i] = FixedVector{maxEdges,Int32}(tmp)
    end
    verticesOnCell = VoronoiMeshes.SmallVectorArray(verticesOnCell_data, nEdges)

    # cellsOnVertex (convert to Vector{FixedVector{3, Int32}})
    #--------------------
    vtk_cells_tri = get_cells(vtk_tri)
    connectivity = vtk_cells_tri.connectivity
    offsets = vtk_cells_tri.offsets
    ranges = [((i == 1 ? 1 : offsets[i-1] + 1):offsets[i]) for i in 1:num_vertices]
    cellsOnVertex = [FixedVector{3,Int32}(Int32.(connectivity[r])) for r in ranges]

    # Ghost indexes are in the end (num_cells+1:) and have negative sign pointing to original cell index
    cell_indx_with_ghosts = get_data(ReadVTK.get_point_data(vtk_tri)["Index"])

    # loop over cells on vertex and fix ghost indices
    for i in 1:num_vertices
        cell_inds = cellsOnVertex[i]
        tmp = Vector{Int32}(undef, 3)
        for j in 1:3
            idx = cell_inds[j]
            if idx > num_cells 
                # Adjust indexing from 0-base to 1-base 
                # and get correct index from ghost indices
                tmp[j] = cell_indx_with_ghosts[idx] + 1  
            else
                tmp[j] = idx  # Keep original index
            end
        end
        cellsOnVertex[i] = FixedVector{3,Int32}(tmp)
    end

    # Get mesh density (constant for now)
    meshDensity = ones(Float64, num_cells)

    # Create the VoronoiMesh
    diagram = VoronoiDiagram(PlanarVoronoiDiagram(
        generators, vertices, verticesOnCell, cellsOnVertex,
        meshDensity, x_period, y_period
    ))

    mesh = VoronoiMeshes.VoronoiMesh(diagram)
    println("VoronoiMesh loaded from VTU files with ", mesh.cells.n, " cells, ", mesh.vertices.n, " vertices")
    return mesh
end

end # module ReadVTKExt
