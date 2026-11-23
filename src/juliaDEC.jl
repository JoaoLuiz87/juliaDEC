using VoronoiMeshes,TensorsLite, TensorsLiteGeometry,  SmallCollections, LinearAlgebra
using StaticArrays, SparseArrays, QuadGK
using CairoMakie

struct Metric{N, T<:SMatrix{N, N, Float64}}
    mat::Symmetric{Float64, T}
end
Metric(mat::T) where {N, T<:SMatrix{N, N, Float64}} = Metric{N, T}(Symmetric(mat))

function Metric(mat::AbstractMatrix{<:Real})
    N, M = size(mat)
    @assert N == M
    return Metric(SMatrix{N, N, Float64}(mat))
end
Metric(N::Int) = Metric(SMatrix{N,N}(1.0I))

struct Cochain{T}
    values::Vector{T}
    k::Int
    is_primal::Bool
end

Cochain(values::Vector{T}, k::Int, is_primal::Bool) where {T} = Cochain{T}(values, k, is_primal)

"""
    get_cell_edge_orientations(mesh::VoronoiMesh, cell_idx::Int)

For a given cell (hexagon), returns a vector of orientations (+1 or -1) for each edge,
where +1 indicates the edge orientation matches the cell's counter-clockwise orientation,
and -1 indicates it opposes it.

Returns:
- Vector{Int}: orientations for each edge in the cell
"""

function get_cells_points(mesh::VoronoiMesh, cell_idx::Int)
    cell_center = mesh.cells.position[cell_idx]
    cell_points = []
    for v_idx in mesh.cells.vertices[cell_idx]
        position = closest(cell_center, mesh.vertices.position[v_idx], mesh.x_period, mesh.y_period)
        push!(cell_points, position)
    end
    return cell_points
end

function get_cells(mesh::VoronoiMesh, cell_idx::Int)
    # Get edges for this cell in CCW order
    edges = mesh.cells.edges[cell_idx]
    orientations = Vector{Int}(undef, length(edges))

    for (i, edge_idx) in enumerate(edges)
        # Get cells on either side of this edge
        c1, _ = mesh.edges.cells[edge_idx]

        # If this cell is c1, edge follows CCW: -1
        # If this cell is c2, edge opposes CCW: +1
        orientations[i] = (cell_idx == c1) ? 1 : -1
    end

    return edges, orientations
end

function get_triangle(triangle_idx::Union{Int32, Int64},
    cellsOnVertex::Vector{SmallCollections.FixedVector{3, Int32}},
    edgesOnVertex::Vector{SmallCollections.FixedVector{3, Int32}},
    cellsOnEdge::Vector{SmallCollections.FixedVector{2, Int32}},
    xVertex::Vector{Float64},
    yVertex::Vector{Float64},
    xCell::Vector{Float64},
    yCell::Vector{Float64},
    xp::Float64,
    yp::Float64)

    edge_indices = Vector{Vector{Int}}(undef, 3)
    edge_orientations = Vector{Int}(undef, 3)

    v1, v2, v3 = cellsOnVertex[triangle_idx]
    circumcenter = TensorsLite.Vec(x = xVertex[triangle_idx], y = yVertex[triangle_idx])
    p1_raw = TensorsLite.Vec(x = xCell[v1], y = yCell[v1])
    p2_raw = TensorsLite.Vec(x = xCell[v2], y = yCell[v2])
    p3_raw = TensorsLite.Vec(x = xCell[v3], y = yCell[v3])

    p1_closest = closest(circumcenter, p1_raw, xp, yp)
    p2_closest = closest(circumcenter, p2_raw, xp, yp)
    p3_closest = closest(circumcenter, p3_raw, xp, yp)

    # Convert back to 2D vectors for output
    points = [
        [p1_closest[1], p1_closest[2]],
        [p2_closest[1], p2_closest[2]],
        [p3_closest[1], p3_closest[2]]
    ]
    # Get the edges for this vertex
    e1, e2, e3 = edgesOnVertex[triangle_idx]

    # Determine the correct order and orientation of edges
    edge_order = zeros(Int, 3)
    orientations = zeros(Int, 3)

    for edge in (e1, e2, e3)
        v_start, v_end = cellsOnEdge[edge]
        if (v_start == v1 && v_end == v2) || (v_end == v1 && v_start == v2)
            edge_order[1] = edge
            orientations[1] = v_start == v1 ? 1 : -1
        elseif (v_start == v2 && v_end == v3) || (v_end == v2 && v_start == v3)
            edge_order[2] = edge
            orientations[2] = v_start == v2 ? 1 : -1
        elseif (v_start == v3 && v_end == v1) || (v_end == v3 && v_start == v1)
            edge_order[3] = edge
            orientations[3] = v_start == v3 ? 1 : -1
        else
            error("Edge $edge does not connect any of the vertices of the triangle")
        end
    end

    if any(iszero, edge_order)
        error("Not all edges were assigned to the triangle")
    end

    edge_indices = edge_order
    edge_orientations = orientations
    return points, edge_indices, edge_orientations
end
get_triangle(mesh::VoronoiMesh, triangle_idx::Union{Int32, Int64}) = get_triangle(triangle_idx,
                                                                mesh.vertices.cells, mesh.vertices.edges,
                                                                mesh.edges.cells, mesh.vertices.position.x,
                                                                mesh.vertices.position.y, mesh.cells.position.x,
                                                                mesh.cells.position.y, mesh.x_period, mesh.y_period)

"""
    d0_matrix_primal(mesh::VoronoiMesh)

Compute the primal d0 operator (triangles vertices -> triangles edges).
Returns a sparse matrix where:
- Rows represent triangles edges
- Columns represent triangles vertices
- Values are {-1, 1} based on edge orientation
"""
function d0_matrix_primal(mesh::VoronoiMesh)
    I, J, V = Int[], Int[], Int[]

    for edge_idx in 1:mesh.edges.n
        c1, c2 = mesh.edges.cells[edge_idx]

        # Edge orientation: c1 -> c2
        push!(I, edge_idx, edge_idx)
        push!(J, c1, c2)
        push!(V, -1, 1)
    end

    return sparse(I, J, V, mesh.edges.n, mesh.cells.n)
end

"""
    d1_matrix_primal(mesh::VoronoiMesh)

Compute the primal d1 operator (triangle edges -> triangles cells).
Returns a sparse matrix where:
- Rows represent triangles cells
- Columns represent edges
- Values are {-1, 1} based on CCW orientation of triangle
"""
function d1_matrix_primal(mesh::VoronoiMesh)
    I, J, V = Int[], Int[], Int[]

    # For each triangle
    for tri_idx in 1:mesh.vertices.n
        _, edge_indices, edge_orientations = get_triangle(mesh, tri_idx)
        push!(I, tri_idx, tri_idx, tri_idx)
        push!(J, edge_indices[1], edge_indices[2], edge_indices[3])
        push!(V, edge_orientations[1], edge_orientations[2], edge_orientations[3])
    end

    return sparse(I, J, V, mesh.vertices.n, mesh.edges.n)
end

"""
    d0_matrix_dual(mesh::VoronoiMesh)

Compute the dual d0 operator (hexagon centers -> hexagon edges).
Returns a sparse matrix where:
- Rows represent hexagon edges
- Columns represent hexagon centers
- Values are {-1, 1} based on edge orientation
"""
function d0_matrix_dual(mesh::VoronoiMesh)
    I, J, V = Int[], Int[], Int[]

    for edge_idx in 1:mesh.edges.n
        v1, v2 = mesh.edges.vertices[edge_idx]

        # Edge orientation: v1 -> v2
        push!(I, edge_idx, edge_idx)
        push!(J, v1, v2)
        push!(V, -1, 1)
    end

    return sparse(I, J, V, mesh.edges.n, mesh.vertices.n)
end

"""
    d1_matrix_dual(mesh::VoronoiMesh)

Compute the dual d1 operator (hexagon edges -> hexagon cells).
Returns a sparse matrix where:
- Rows represent hexagon vertices
- Columns represent hexagon edges
- Values are {-1, 1} based on CCW orientation of hexagon
"""
function d1_matrix_dual(mesh::VoronoiMesh)
    I, J, V = Int[], Int[], Int[]

    # For each hexagon vertex
    for cell_idx in 1:mesh.cells.n
        edges, orientations = get_cells(mesh, cell_idx)
        for (edge_idx, orientation) in zip(edges, orientations)
            push!(I, cell_idx)
            push!(J, edge_idx)
            push!(V, orientation)
        end
    end

    return sparse(I, J, V, mesh.cells.n, mesh.edges.n)
end

# Generic exterior derivative operators remain the same
function d(mesh::VoronoiMesh, k::Int, is_primal::Bool)
    if is_primal
        if k == 0
            return d0_matrix_primal(mesh)
        elseif k == 1
            return d1_matrix_primal(mesh)
        else
            error("Exterior derivative not implemented for k > 1")
        end
    else
        if k == 0
            return d0_matrix_dual(mesh)
        elseif k == 1
            return d1_matrix_dual(mesh)
        else
            error("Exterior derivative not implemented for k > 1")
        end
    end
end

d(mesh::VoronoiMesh, w::Cochain{T}) where T =
    Cochain(d(mesh, w.k, w.is_primal) * w.values, w.k + 1, w.is_primal)

#---------------------------------------------------------------------------------------------------
#Diagonal Hodge \star

"""
    hodge_star(mesh::VoronoiMesh, k::Int, is_primal::Bool)

Compute the Hodge star operator for a VoronoiMesh.
"""
function hodge_star_0_primal(areaCell::Vector{T}) where {T}
    return Diagonal(areaCell)
end

function hodge_star_1_primal(PrimalEdgeLength::Vector{T}, DualEdgeLength::Vector{T}) where {T}
    return Diagonal([DualEdgeLength[i] / PrimalEdgeLength[i] for i in 1:length(PrimalEdgeLength)])
end

function hodge_star_2_primal(areaTriangle::Vector{T}) where {T}
    return Diagonal([1.0 / areaTriangle[i] for i in 1:length(areaTriangle)])
end

function hodge_star_0_dual(areaTriangle::Vector{T}) where {T}
    return Diagonal([areaTriangle[i] for i in 1:length(areaTriangle)])
end

function hodge_star_1_dual(PrimalEdgeLength::Vector{T}, DualEdgeLength::Vector{T}) where {T}
    return Diagonal([DualEdgeLength[i] / PrimalEdgeLength[i] for i in 1:length(PrimalEdgeLength)])
end

function hodge_star_2_dual(areaCell::Vector{T}) where {T}
    return Diagonal([1.0 / areaCell[i] for i in 1:length(areaCell)])
end

function star(mesh::VoronoiMesh, k::Int, is_primal::Bool)
    if is_primal
        if k == 0
            return hodge_star_0_primal(mesh.cells.area)
        elseif k == 1
            return hodge_star_1_primal(mesh.edges.lengthDual, mesh.edges.length)
        elseif k == 2
            return hodge_star_2_primal(mesh.vertices.area)
        else
            error("Hodge star not implemented for k > 2")
        end
    else
        if k == 0
            return hodge_star_0_dual(mesh.vertices.area)
        elseif k == 1
            return hodge_star_1_dual(mesh.edges.length, mesh.edges.lengthDual)
        elseif k == 2
            return hodge_star_2_dual(mesh.cells.area)
        else
            error("Hodge star not implemented for k > 2")
        end
    end
end

star(mesh::VoronoiMesh, w::Cochain{T}) where {T} = Cochain(star(mesh, w.k, w.is_primal) * w.values, 2-w.k, !w.is_primal)

#---------------------------------------------------------------------------------------------------
# Quadrature on edges

function gaussian_quadrature_points_and_weights(n::Int)
    if n == 1
        return ([0.0], [2.0])
    elseif n == 2
        return ([-1/sqrt(3), 1/sqrt(3)], [1.0, 1.0])
    elseif n == 3
        return ([-sqrt(3/5), 0.0, sqrt(3/5)], [5/9, 8/9, 5/9])
    elseif n == 4
        points = [-sqrt((3+2*sqrt(6/5))/7), -sqrt((3-2*sqrt(6/5))/7),
                  sqrt((3-2*sqrt(6/5))/7),  sqrt((3+2*sqrt(6/5))/7)]
        weights = [(18-sqrt(30))/36, (18+sqrt(30))/36,
                  (18+sqrt(30))/36, (18-sqrt(30))/36]
        return (points, weights)

    elseif n == 5
        # 5-point Gaussian quadrature on [-1,1]
        points = [
            -sqrt(5 + 2*sqrt(10/7))/3,
            -sqrt(5 - 2*sqrt(10/7))/3,
            0.0,
            sqrt(5 - 2*sqrt(10/7))/3,
            sqrt(5 + 2*sqrt(10/7))/3
        ]
        weights = [
            (322 - 13*sqrt(70))/900,
            (322 + 13*sqrt(70))/900,
            128/225,
            (322 + 13*sqrt(70))/900,
            (322 - 13*sqrt(70))/900
        ]
        return (points, weights)
    else
        error("Gaussian quadrature not implemented for n > 4")
    end
end

function edge_integral(v::Function, p1::Vector{T}, p2::Vector{T}, xp::Float64, yp::Float64, n::Int=4) where T
    # Compute the integral of v along the edge from p1 to p2 using n-point Gaussian quadrature
    #p1_vec = TensorsLite.Vec(p1..., 0.0)
    #p2_vec = TensorsLite.Vec(p2..., 0.0)

    #p2_closest = closest(p1_vec, p2_vec, xp, yp)[1:2]
    edge_vector = p2 - p1
    points, weights = gaussian_quadrature_points_and_weights(n)

    integral = 0.0
    for (xi, wi) in zip(points, weights)
        # Map xi from [-1, 1] to [0, 1]
        t = (xi + 1) / 2
        # Compute the point on the edge
        p = p1 + t * edge_vector
        # Evaluate the vector field and compute the contribution to the integral
        integral += wi * dot(v(p), edge_vector)
    end

    return integral / 2  # Scale by edge length and adjust for the change of interval
end

@inline area(a::Vector{T}, b::Vector{T}, c::Vector{T}) where T = 0.5 * abs((b[1] - a[1]) * (c[2] - a[2]) - (c[1] - a[1]) * (b[2] - a[2]))

#---------------------------------------------------------------------------------------------------
# Stencils of neighbouring edges, and the integrals of the polynomial basis on edges

# Get the edges connected an edge vertex.
# VoronoiMeshes stores these as Vector{FixedVector{2}} / Vector{FixedVector{3}}.
function get_neighbors_1level_primal(verticesOnEdge::AbstractVector, edgesOnVertex::AbstractVector, edge_index::Integer)
    v1, v2 = verticesOnEdge[edge_index]
    neighbors = Set{Int}()

    # Add the edge itself
    push!(neighbors, Int(edge_index))

    # Add edges connected to v1
    for e in edgesOnVertex[v1]
        push!(neighbors, Int(e))
    end

    # Add edges connected to v2
    for e in edgesOnVertex[v2]
        push!(neighbors, Int(e))
    end
    return collect(neighbors)
end

get_neighbors_1level_primal(mesh::VoronoiMesh, edge_index::Int) = get_neighbors_1level_primal(mesh.edges.vertices, mesh.vertices.edges, edge_index)

function get_neighbors_2level_primal(mesh::VoronoiMesh, edge_index::Int)
    neighbors_1level = get_neighbors_1level_primal(mesh, edge_index)
    neighbors_2level = Set{Int}()

    for e in neighbors_1level
        union!(neighbors_2level, get_neighbors_1level_primal(mesh, e))
    end
    return collect(neighbors_2level)
end

function get_neighbors_3level_primal(mesh::VoronoiMesh, edge_index::Int)
    v1, v2 = mesh.edges.vertices[edge_index]
    neighbors = Set{Int}()
    cells = Set{Int}()
    # Add the edge itself
    push!(neighbors, edge_index)

    # Add edges connected to v1
    for v in [v1, v2]
        for cell in mesh.vertices.cells[v]
            if cell ∉ cells
                push!(cells, cell)
            end
        end
    end
    for cell in cells
        for edge in mesh.cells.edges[cell]
            push!(neighbors, edge)
        end
    end
    return collect(neighbors)
end

function integrate_basic_fields_primal_edge(mesh::VoronoiMesh, edge_idx::Int, x0::Vec2Dxy{Float64}, only_linear::Bool=true, hodge_applied::Bool=false)
    if hodge_applied == false
        f1 = x -> [1.0, 0.0]
        f2 = x -> [0.0, 1.0]
        f3 = x -> [x[1]-x0[1], 0.0]
        f4 = x -> [x[2]-x0[2], 0.0]
        f5 = x -> [0.0, x[1]-x0[1]]
        f6 = x -> [0.0, x[2]-x0[2]]
        f7 = x -> [(x[1]-x0[1])^2, 0.0]
        f8 = x -> [(x[1]-x0[1])*(x[2]-x0[2]), 0.0]
        f9 = x -> [(x[2]-x0[2])^2, 0.0]
        f10 = x -> [0.0, (x[1]-x0[1])^2]
        f11 = x -> [0.0, (x[1]-x0[1])*(x[2]-x0[2])]
        f12 = x -> [0.0, (x[2]-x0[2])^2]
    else
        f1 = x -> [0.0, 1.0]
        f2 = x -> [-1.0, 0.0]
        f3 = x -> [0.0, x[1]-x0[1]]
        f4 = x -> [0.0, x[2]-x0[2]]
        f5 = x -> [-x[1]+x0[1], 0.0]
        f6 = x -> [-x[2]+x0[2], 0.0]
        f7 = x -> [0.0, (x[1]-x0[1])^2]
        f8 = x -> [0.0, (x[1]-x0[1])*(x[2]-x0[2])]
        f9 = x -> [0.0, (x[2]-x0[2])^2]
        f10 = x -> [-(x[1]-x0[1])^2, 0.0]
        f11 = x -> [-(x[1]-x0[1])*(x[2]-x0[2]), 0.0]
        f12 = x -> [-(x[2]-x0[2])^2, 0.0]
    end
    if only_linear
        f = [f1, f2, f3, f4, f5, f6]
    else
        f = [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12]
    end
    edge_pos = closest(x0, mesh.edges.position[edge_idx], mesh.x_period, mesh.y_period)

    # Compute the hodge from dual to primal, by integrating over the primal edge
    v1, v2 = mesh.edges.cells[edge_idx]
    p1_primal = closest(edge_pos, mesh.cells.position[v1], mesh.x_period, mesh.y_period)
    p2_primal = closest(edge_pos, mesh.cells.position[v2], mesh.x_period, mesh.y_period)
    p1_primal = [p1_primal[1], p1_primal[2]]
    p2_primal = [p2_primal[1], p2_primal[2]]

    integration_values_on_primal = [edge_integral(f[j], p1_primal, p2_primal, mesh.x_period, mesh.y_period) for j in eachindex(f)]

    return integration_values_on_primal
end

function integrate_basic_fields_dual_edge(mesh::VoronoiMesh, edge_idx::Int, x0::Vec2Dxy{Float64}, only_linear::Bool=true, hodge_applied::Bool=false)
    if hodge_applied == false
        star_f1 = x -> [1.0, 0.0]
        star_f2 = x -> [0.0, 1.0]
        star_f3 = x -> [x[1]-x0[1], 0.0]
        star_f4 = x -> [x[2]-x0[2], 0.0]
        star_f5 = x -> [0.0, x[1]-x0[1]]
        star_f6 = x -> [0.0, x[2]-x0[2]]
        star_f7 = x -> [(x[1]-x0[1])^2, 0.0]
        star_f8 = x -> [(x[1]-x0[1])*(x[2]-x0[2]), 0.0]
        star_f9 = x -> [(x[2]-x0[2])^2, 0.0]
        star_f10 = x -> [0.0, (x[1]-x0[1])^2]
        star_f11 = x -> [0.0, (x[1]-x0[1])*(x[2]-x0[2])]
        star_f12 = x -> [0.0, (x[2]-x0[2])^2]
    else
        star_f1 = x -> [0.0, 1.0]
        star_f2 = x -> [-1.0, 0.0]
        star_f3 = x -> [0.0, x[1]-x0[1]]
        star_f4 = x -> [0.0, x[2]-x0[2]]
        star_f5 = x -> [-x[1]+x0[1], 0.0]
        star_f6 = x -> [-x[2]+x0[2], 0.0]
        star_f7 = x -> [0.0, (x[1]-x0[1])^2]
        star_f8 = x -> [0.0, (x[1]-x0[1])*(x[2]-x0[2])]
        star_f9 = x -> [0.0, (x[2]-x0[2])^2]
        star_f10 = x -> [-(x[1]-x0[1])^2, 0.0]
        star_f11 = x -> [-(x[1]-x0[1])*(x[2]-x0[2]), 0.0]
        star_f12 = x -> [-(x[2]-x0[2])^2, 0.0]
    end

    if only_linear
        f = [star_f1, star_f2, star_f3, star_f4, star_f5, star_f6]
    else
        f = [star_f1, star_f2, star_f3, star_f4, star_f5, star_f6, star_f7, star_f8, star_f9, star_f10, star_f11, star_f12]
    end
    edge_pos = closest(x0, mesh.edges.position[edge_idx], mesh.x_period, mesh.y_period)

    # Compute the hodge from dual to primal, by integrating over the primal edge
    v1, v2 = mesh.edges.vertices[edge_idx]
    p1_primal = closest(edge_pos, mesh.vertices.position[v1], mesh.x_period, mesh.y_period)
    p2_primal = closest(edge_pos, mesh.vertices.position[v2], mesh.x_period, mesh.y_period)
    p1_primal = [p1_primal[1], p1_primal[2]]
    p2_primal = [p2_primal[1], p2_primal[2]]

    integration_values_on_primal = [edge_integral(f[j], p1_primal, p2_primal, mesh.x_period, mesh.y_period) for j in eachindex(f)]

    return integration_values_on_primal
end
