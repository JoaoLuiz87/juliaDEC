"""
    WriteVTKExt

Export VoronoiMeshes.jl meshes to VTU (VTK Unstructured Grid) format.
- Provides functions to export Voronoi and Delaunay meshes with periodic ghost vertices.
- Uses WriteVTK for file output; supports mesh metadata and ghost marking.
- Includes both specific and generic mesh export helpers for flexible usage.
"""
module WriteVTKExt

using VoronoiMeshes, TensorsLite, TensorsLiteGeometry, Zeros, SmallCollections, LinearAlgebra
import VoronoiMeshes: save_triangulation_to_vtu, save_voronoi_to_vtu

using PrecompileTools

using WriteVTK # For saving meshes in VTU format
using WriteVTK.VTKBase


"""
    create_ghost_periodic_voronoi_vertices(mesh::AbstractVoronoiMesh{false})

Create Voronoi cell vertex and topology arrays, adding ghost vertices for periodic boundaries.
Returns (vertices_with_ghosts, verticesOnPolygon_with_ghosts, n_ghosts).
"""
function create_ghost_periodic_voronoi_vertices(mesh::AbstractVoronoiMesh{false})

    # Mesh info
    vert_pos = mesh.vertices.position # Voronoi cell vertices
    polygon_pos = mesh.cells.position # Voronoi cell centers
    verticesOnPolygon = mesh.cells.vertices # Indexes of vertices per Voronoi cell

    return create_ghost_periodic_points(vert_pos, polygon_pos, verticesOnPolygon, mesh.x_period, mesh.y_period)
end



"""
    create_ghost_periodic_triangulation_vertices(mesh::AbstractVoronoiMesh{false})

Create Delaunay triangulation vertex and topology arrays, adding ghost vertices for periodic boundaries.
Returns (vertices_with_ghosts, verticesOnPolygon_with_ghosts, n_ghosts).
"""
function create_ghost_periodic_triangulation_vertices(mesh::AbstractVoronoiMesh{false})

    # Mesh info
    vert_pos = mesh.cells.position # Triangle circumcenters
    polygon_pos = mesh.vertices.position # Triangle vertices
    verticesOnPolygon = mesh.vertices.cells # Indexes of vertices per triangle

    return create_ghost_periodic_points(vert_pos, polygon_pos, verticesOnPolygon, mesh.x_period, mesh.y_period)
end


"""
    create_ghost_periodic_points(vert_pos, polygon_pos, verticesOnPolygon, x_period, y_period)

Helper to add ghost vertices for periodic boundaries to a set of polygons.
Returns (vertices_with_ghosts, verticesOnPolygon_with_ghosts, n_ghosts).
"""
function create_ghost_periodic_points(vert_pos, polygon_pos, verticesOnPolygon, x_period, y_period)

    # Ghost info
    vertices_with_ghosts = copy(vert_pos)
    ivertices_with_ghosts = length(vert_pos) + 1

    verticesOnPolygon_with_ghosts = [Vector(v) for v in verticesOnPolygon]

    ghost_dict = Dict{Int,Vector{Int}}() # Save ghost index for each original vertex

    for i in eachindex(polygon_pos)

        ppos = polygon_pos[i]
        vert_on_pol = verticesOnPolygon[i]
        vert_on_pol_w_ghost = verticesOnPolygon_with_ghosts[i]

        for j in eachindex(vert_on_pol)

            original_index = vert_on_pol[j]
            vpos = closest(ppos, vert_pos[original_index], x_period, y_period)

            if norm(vpos - vert_pos[original_index]) > 0.4 * min(x_period, y_period)
                if !haskey(ghost_dict, original_index) #first time we see this ghost

                    ghost_dict[original_index] = [ivertices_with_ghosts]
                    vert_on_pol_w_ghost[j] = ivertices_with_ghosts
                    push!(vertices_with_ghosts, vpos)
                    ivertices_with_ghosts += 1
                    #println(i, " ", j, " Ghost found:", vpos, " for original vertex ", original_index, " at ", vert_pos[original_index], " Dict:", d[original_index])

                else #in this case a key already exists, check if this vertex is close or not to existing ghosts

                    ghost_exists = false
                    for k in ghost_dict[original_index]
                        if norm(vpos - vertices_with_ghosts[k]) < 1e-8 * min(x_period, y_period)

                            # use the existing ghost vertex
                            vert_on_pol_w_ghost[j] = k
                            ghost_exists = true

                        end
                    end

                    #New ghost vertex needed
                    if !ghost_exists

                        # add new ghost vertex
                        push!(ghost_dict[original_index], ivertices_with_ghosts)
                        vert_on_pol_w_ghost[j] = ivertices_with_ghosts
                        push!(vertices_with_ghosts, vpos)
                        ivertices_with_ghosts += 1

                    end
                end
            end
        end
    end

    n_ghosts = length(vertices_with_ghosts) - length(vert_pos)

    println("Added $n_ghosts ghost vertices to VTU to handle periodicity in mesh with ", length(vert_pos), " original vertices.")

    return vertices_with_ghosts, verticesOnPolygon_with_ghosts, n_ghosts, ghost_dict

end

"""
    save_voronoi_to_vtu(file_name::String, mesh::AbstractVoronoiMesh{false})

Export the Voronoi mesh to a VTU file, handling periodic ghost vertices.
"""
function save_voronoi_to_vtu(file_name::String, mesh::AbstractVoronoiMesh{false})

    # Here the vertices are Voronoi cell vertices and the polygons are the Voronoi cells
    vertices_with_ghosts, verticesOnPolygon_with_ghosts, n_ghosts, ghost_dict =
        create_ghost_periodic_voronoi_vertices(mesh)

    n_polys = mesh.cells.n #number of Voronoi cells
    n_vertices = mesh.vertices.n #number of Voronoi cell vertices (matches number of triangles)

    points_mat = hcat((v[1:2] for v in vertices_with_ghosts)...)

    cells = Vector{MeshCell}(undef, n_polys)

    for i in 1:n_polys
        links = collect(verticesOnPolygon_with_ghosts[i])
        cells[i] = MeshCell(VTKCellTypes.VTK_POLYGON, links)
    end
    # Save ghost periodicity information

    # Vertices from 1:n_polys will save their index 1:n_polys
    # Ghost vertices will save the index of the original vertex they correspond to
    n_vertices_with_ghosts = length(vertices_with_ghosts)

    ghost_idx = collect(1:n_vertices_with_ghosts)

    for i in 1:n_vertices
        if haskey(ghost_dict, i)
            for j in ghost_dict[i]
                ghost_idx[j] = i
            end
        end
    end

    # Shift indices by -1 for storage convention (0-based in VTK)
    @inbounds ghost_idx .-= 1

    # add mesh metadata
    saved_file = vtk_grid(file_name, points_mat, cells) do vtk
        vtk["Index", VTKPointData()] = ghost_idx
        vtk["NumCells"] = mesh.cells.n
        vtk["NumVertices"] = mesh.vertices.n
        vtk["NumEdges"] = mesh.edges.n
        vtk["XPeriod"] = mesh.x_period
        vtk["YPeriod"] = mesh.y_period
        vtk["NumPeriodicGhosts"] = n_ghosts
    end

    return saved_file
end


"""
    save_triangulation_to_vtu(file_name::String, mesh::AbstractVoronoiMesh{false})

Export the Delaunay triangulation (dual mesh) to a VTU file, handling periodic ghost vertices.
"""
function save_triangulation_to_vtu(file_name::String, mesh::AbstractVoronoiMesh{false})

    # Here the vertices are cell centers and the polygons are the triangles
    vertices_with_ghosts, verticesOnPolygon_with_ghosts, n_ghosts, ghost_dict =
        create_ghost_periodic_triangulation_vertices(mesh)

    n_polys = mesh.vertices.n #number of mesh triangles
    n_vertices = mesh.cells.n #number of triangle vertices

    points_mat = hcat((v[1:2] for v in vertices_with_ghosts)...)

    cells = Vector{MeshCell}(undef, n_polys)

    for i in 1:n_polys
        links = collect(verticesOnPolygon_with_ghosts[i])
        cells[i] = MeshCell(VTKCellTypes.VTK_TRIANGLE, links)
    end

    # Save ghost periodicity information

    # Vertices from 1:n_polys will save their index 1:n_polys
    # Ghost vertices will save the index of the original vertex they correspond to
    n_vertices_with_ghosts = length(vertices_with_ghosts)

    ghost_idx = collect(1:n_vertices_with_ghosts)

    for i in 1:n_vertices
        if haskey(ghost_dict, i)
            for j in ghost_dict[i]
                ghost_idx[j] = i
            end
        end
    end

    # Shift indices by -1 for storage convention (VTK uses 0-based indexing)
    ghost_idx .-= 1

    # add mesh metadata
    saved_file = vtk_grid(file_name, points_mat, cells) do vtk
        vtk["Index", VTKPointData()] = ghost_idx
        vtk["NumCells"] = mesh.cells.n
        vtk["NumVertices"] = mesh.vertices.n
        vtk["NumEdges"] = mesh.edges.n
        vtk["XPeriod"] = mesh.x_period
        vtk["YPeriod"] = mesh.y_period
        vtk["NumPeriodicGhosts"] = n_ghosts
    end

    return saved_file
end

end # module VTKExt
