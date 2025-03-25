using VoronoiMeshes,TensorsLite, TensorsLiteGeometry, ImmutableVectors, LinearAlgebra, NCDatasets, MPASTools
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

primal_cochain(vec::Vector{T}, k::Int) where {T} = Cochain(vec, k, true)
dual_cochain(vec::Vector{T}, k::Int) where {T} = Cochain(vec, k, false)

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
    cellsOnVertex::Vector{Tuple{Int32, Int32, Int32}}, 
    edgesOnVertex::Vector{Tuple{Int32, Int32, Int32}}, 
    cellsOnEdge::Vector{Tuple{Int32, Int32}}, 
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
get_triangle(mesh::VoronoiMesh, triangle_idx::Int) = get_triangle(triangle_idx, 
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
#Function that takes out the periodicity to compute


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
#**(mesh::VoronoiMesh, w::Cochain{T}) where {T} = Cochain( (-1)^((2-w.k)*w.k) * w.values, w.k, w.is_primal)
#---------------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------------
# Laplace-de Rham operator for 0-forms

"""
    laplace_de_rham(mesh::VoronoiMesh, is_primal::Bool)

Compute the Laplace-de Rham operator for a VoronoiMesh of a 0-form.
"""
function laplace_de_rham(mesh::VoronoiMesh, is_primal::Bool)
    if is_primal
        return star(mesh, 2, false) * d(mesh, 1, false) * star(mesh, 1, true) * d(mesh, 0, true)
    else
        return star(mesh, 2, true) * d(mesh, 1, true) * star(mesh, 1, false) * d(mesh, 0, false)
    end
end

laplace_de_rham(mesh::VoronoiMesh, w::Cochain{T}) where T = Cochain(laplace_de_rham(mesh, w.is_primal) * w.values, 0, w.is_primal)
#---------------------------------------------------------------------------------------------------
#Interpolation operators
# Vector field discretization and Whitney interpolation

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

function triangle_quadrature_rule(order::Int)
    if order == 1
        # 1-point rule
        points = [[1/3, 1/3]]
        weights = [1.0]
    elseif order == 2
        # 3-point rule
        points = [[1/6, 1/6], [2/3, 1/6], [1/6, 2/3]]
        weights = [1/3, 1/3, 1/3]
    elseif order == 4
        # Quadrature points and weights for Order 3
    else
        error("Order $order not implemented")
    end
    return points, weights
end

"""
Integrate a 2-form over a triangle using triangle quadrature
"""
function triangle_integral(f::Function, p1::Vector{T}, p2::Vector{T}, p3::Vector{T}, 
                            order::Int=2) where T
    # Get quadrature points and weights
    points, weights = triangle_quadrature_rule(order)
    
    # Initialize result
    integral = 0.0
    
    # Compute area of triangle
    area = 0.5 * abs((p2[1] - p1[1])*(p3[2] - p1[2]) - (p3[1] - p1[1])*(p2[2] - p1[2]))
    
    # For each quadrature point
    for (point, weight) in zip(points, weights)
        # Map reference point to physical triangle
        x = p1[1] + (p2[1] - p1[1])*point[1] + (p3[1] - p1[1])*point[2]
        y = p1[2] + (p2[2] - p1[2])*point[1] + (p3[2] - p1[2])*point[2]
        
        # Add contribution
        integral += weight * f([x, y])
    end
    
    # Scale by area
    integral *= area
    
    return integral
end

function discretize_vector_field_primal(mesh::VoronoiMesh, v::Function)
    edge_values = zeros(length(mesh.edges.vertices))
    xp = mesh.x_period
    yp = mesh.y_period  
    for i in 1:length(mesh.edges.vertices)
        edge_pos = mesh.edges.position[i]
        v1, v2 = mesh.edges.vertices[i]
        
        # Get closest periodic points relative to edge_pos
        p1_vec = closest(edge_pos, mesh.vertices.position[v1], xp, yp)
        p2_vec = closest(edge_pos, mesh.vertices.position[v2], xp, yp)
        
        p1 = [p1_vec[1], p1_vec[2]]
        p2 = [p2_vec[1], p2_vec[2]]

        edge_values[i] = edge_integral(v, p1, p2, xp, yp)
    end
    return Cochain(edge_values, 1, false)
end

function discretize_vector_field_dual(mesh::VoronoiMesh, v::Function)
    edge_values = zeros(length(mesh.edges.cells))   
    xp = mesh.x_period
    yp = mesh.y_period
    for i in 1:length(mesh.edges.cells)
        edge_pos = mesh.edges.position[i]
        v1, v2 = mesh.edges.cells[i]
        
        # Get closest periodic points relative to edge_pos
        p1_vec = closest(edge_pos, mesh.cells.position[v1], xp, yp)
        p2_vec = closest(edge_pos, mesh.cells.position[v2], xp, yp)
        
        p1 = [p1_vec[1], p1_vec[2]]
        p2 = [p2_vec[1], p2_vec[2]]

        edge_values[i] = edge_integral(v, p1, p2, xp, yp)
    end
    return Cochain(edge_values, 1, true)
end

function discretize_scalar_field_triangles(mesh::VoronoiMesh, f::Function)
    tam = mesh.vertices.n
    cell_values = zeros(tam)
    for i = 1:tam
        points, _, _= get_triangle(i, mesh.vertices.cells, mesh.vertices.edges, mesh.edges.cells, mesh.vertices.position.x, mesh.vertices.position.y, mesh.cells.position.x, mesh.cells.position.y, mesh.x_period, mesh.y_period)
        cell_values[i] = triangle_integral(f, points[1], points[2], points[3])
    end
    return Cochain(cell_values, 2, true)
end
#---------------------------------------------------------------------------------------------------
#Whitney interpolation

function whitney_reconstruction_on_x_periodic(x::Vector{T}, p1::Vector{T}, p2::Vector{T}, p3::Vector{T}, hodge_applied=false) where T

# Use closest points for computation
x_A, y_A = p1
x_B, y_B = p2
x_C, y_C = p3
denom = (y_B - y_C) * x_A + (y_C - y_A) * x_B + (y_A - y_B) * x_C

# Barycentric coordinates using closest point
l0 = ((y_B - y_C) * x[1] + (x_C - x_B) * x[2] + x_B * y_C - x_C * y_B) / denom
l1 = ((y_C - y_A) * x[1] + (x_A - x_C) * x[2] + x_C * y_A - x_A * y_C) / denom
l2 = ((y_A - y_B) * x[1] + (x_B - x_A) * x[2] + x_A * y_B - x_B * y_A) / denom

# Define the differentials of the barycentric coordinates
dl0 = [y_B - y_C, x_C - x_B] / denom
dl1 = [y_C - y_A, x_A - x_C] / denom
dl2 = [y_A - y_B, x_B - x_A] / denom

if hodge_applied
    dl0 = [-x_C + x_B, y_B - y_C] / denom
    dl1 = [-x_A + x_C, y_C - y_A] / denom
    dl2 = [-x_B + x_A, y_A - y_B] / denom
end

# Define the Whitney 1-forms
whitney_matrix = zeros(Float64, (3,2))
whitney_matrix[1,:] = l0 * dl1 - l1 * dl0
whitney_matrix[2,:] = l1 * dl2 - l2 * dl1
whitney_matrix[3,:] = l2 * dl0 - l0 * dl2

    return whitney_matrix  
end 

whitney_reconstruction_on_x_periodic(x::Tuple{Float64, Float64},
p1::Vector{Float64}, 
p2::Vector{Float64}, 
p3::Vector{Float64}, 
hodge_applied=false) = whitney_reconstruction_on_x_periodic([x...], p1, p2, p3, hodge_applied)


function whitney_basis_functions_on_triangle_on_x(mesh::VoronoiMesh, x::Vector{T}, triangle_idx::Int, hodge_applied=false) where T

    points, edge_indices, edge_orientations = get_triangle(triangle_idx, triangle_idx)
    p = TensorsLite.Vec(x = x[1], y = x[2])
    new_x = closest(mesh.vertices.position[triangle_idx], p, mesh.x_period, mesh.y_period)
    basis_functions = whitney_reconstruction_on_x_periodic(new_x, points[1], points[2], points[3], hodge_applied)
    
    return basis_functions, edge_indices, edge_orientations
end

function whitney_basis_functions_on_triangle(mesh::VoronoiMesh, triangle_idx::Int, hodge_applied=false)

    circumcenter = [mesh.vertices.position.x[triangle_idx], mesh.vertices.position.y[triangle_idx]]
    points, edge_indices, edge_orientations = get_triangle(mesh, triangle_idx)
    basis_functions = whitney_reconstruction_on_x_periodic(circumcenter, points[1], points[2], points[3], hodge_applied)    
    return basis_functions, edge_indices, edge_orientations
end 

function whitney_basis_functions(mesh::VoronoiMesh, hodge_applied=false)
    basis_functions = Vector{Matrix{Float64}}(undef, mesh.vertices.n)
    edge_indices = Vector{Vector{Int}}(undef, mesh.vertices.n)
    edge_orientations = Vector{Vector{Int}}(undef, mesh.vertices.n)

    for i in 1:mesh.vertices.n
        basis_functions[i], edge_indices[i], edge_orientations[i] = 
        whitney_basis_functions_on_triangle(mesh, i, hodge_applied)
    end
    return basis_functions, edge_indices, edge_orientations
end

function whitney_interpolation_on_triangle(mesh::VoronoiMesh, triangle_idx::Int, v::Cochain{T}) where T
    basis_functions, edge_indices, edge_orientations = whitney_basis_functions_on_triangle(mesh, triangle_idx)
    interpolation_vector = zeros(Float64, 2)
    for i in 1:3
        product = (basis_functions[i,:] * v.values[edge_indices[i]]) * edge_orientations[i]
        interpolation_vector += product
    end
    return interpolation_vector
end

function whitney_interpolation(mesh::VoronoiMesh, v::Cochain{T}) where T
    basis_functions, edge_indices, edge_orientations = whitney_basis_functions(mesh)
    values = [v.values[[edge_indices[i]...]] for i in 1:mesh.vertices.n]
    interpolation_vectors = zeros(Float64, (mesh.vertices.n, 2))
    for i in 1:mesh.vertices.n
        # Perform element-wise multiplication
        product = basis_functions[i] .* values[i] .* edge_orientations[i]
        # Sum the rows
        interpolation_vectors[i,:] = sum(product, dims=1)
    end
    return interpolation_vectors
end

function whitney_integration_on_edge(mesh::VoronoiMesh, edge_idx::Int)
    # Get the dual edge (hexagon edge) endpoints
    xp = mesh.x_period
    yp = mesh.y_period
    edge_position = mesh.edges.position[edge_idx]
    v1, v2 = mesh.edges.vertices[edge_idx]
    
    # Get closest periodic points relative to p1
    p1_vec = closest(edge_position, mesh.vertices.position[v1], xp, yp)
    p2_vec = closest(edge_position, mesh.vertices.position[v2], xp, yp)
    
    p1 = [p1_vec[1], p1_vec[2]]
    p2 = [p2_vec[1], p2_vec[2]]
    epos = [edge_position[1], edge_position[2]]
    weights_vec = Vector{Float64}(undef, 6)
    edge_indices_vec = Vector{Int}(undef, 6)

    points1, edge_indices1, orientations1 = get_triangle(v1, 
                                                    mesh.vertices.cells, mesh.vertices.edges, mesh.edges.cells, 
                                                    mesh.vertices.position.x, mesh.vertices.position.y, 
                                                    mesh.cells.position.x, mesh.cells.position.y, 
                                                    xp, yp)
    
    vec1_p1 = TensorsLite.Vec(x=points1[1][1], y=points1[1][2])
    vec1_p2 = TensorsLite.Vec(x=points1[2][1], y=points1[2][2])
    vec1_p3 = TensorsLite.Vec(x=points1[3][1], y=points1[3][2])
    periodic1_p1 = closest(edge_position, vec1_p1, xp, yp)
    periodic1_p2 = closest(edge_position, vec1_p2, xp, yp)
    periodic1_p3 = closest(edge_position, vec1_p3, xp, yp)
    periodic1_p1 = [periodic1_p1[1], periodic1_p1[2]]
    periodic1_p2 = [periodic1_p2[1], periodic1_p2[2]]
    periodic1_p3 = [periodic1_p3[1], periodic1_p3[2]]


    integral1 = [edge_integral(x -> whitney_reconstruction_on_x_periodic(x, periodic1_p1, periodic1_p2, periodic1_p3, true)[i,:]*orientations1[i],
                                    p1, epos, 
                                    xp, yp) for i in 1:3]

    
    points2, edge_indices2, orientations2 = get_triangle(v2, 
                                                    mesh.vertices.cells, mesh.vertices.edges, mesh.edges.cells, 
                                                    mesh.vertices.position.x, mesh.vertices.position.y, 
                                                    mesh.cells.position.x, mesh.cells.position.y, 
                                                    xp, yp)

    vec2_p1 = TensorsLite.Vec(x=points2[1][1], y=points2[1][2])
    vec2_p2 = TensorsLite.Vec(x=points2[2][1], y=points2[2][2])
    vec2_p3 = TensorsLite.Vec(x=points2[3][1], y=points2[3][2])
    periodic2_p1 = closest(edge_position, vec2_p1, xp, yp)
    periodic2_p2 = closest(edge_position, vec2_p2, xp, yp)
    periodic2_p3 = closest(edge_position, vec2_p3, xp, yp)
    periodic2_p1 = [periodic2_p1[1], periodic2_p1[2]]
    periodic2_p2 = [periodic2_p2[1], periodic2_p2[2]]
    periodic2_p3 = [periodic2_p3[1], periodic2_p3[2]]

    integral2 = [edge_integral(x -> whitney_reconstruction_on_x_periodic(x, periodic2_p1, periodic2_p2, periodic2_p3, true)[i,:]*orientations2[i],
                                    epos, p2, 
                                    xp, yp) for i in 1:3]
    
    weights_vec[1:3] = integral1
    edge_indices_vec[1:3] = edge_indices1
    weights_vec[4:6] = integral2    
    edge_indices_vec[4:6] = edge_indices2

    return weights_vec, edge_indices_vec
end

function whitney_hodge_star_operator(mesh::VoronoiMesh, v::Cochain{T}) where T
    hodge_star_vectors = Vector{T}(undef, mesh.edges.n)
    for i in 1:mesh.edges.n
        weights, edge_indices = whitney_integration_on_edge(mesh, i)
        values = [v.values[edge_indices[j]] for j in 1:6]
        hodge_star_vectors[i] = sum(weights .* values)
    end
    return Cochain(hodge_star_vectors, 1, false)
end

function whitney_hodge_star_matrix(mesh::VoronoiMesh)
    # Initialize dictionary to accumulate weights for repeated indices
    weight_dict = Dict{Tuple{Int,Int}, Float64}()
    
    # For each edge i
    for i in 1:mesh.edges.n
        # Get weights and contributing edges for the i-th row
        weights, edge_indices = whitney_integration_on_edge(mesh, i)
        
        # Accumulate weights for each (i,j) pair
        for (j, idx) in enumerate(edge_indices)
            key = (i, idx)
            weight_dict[key] = get(weight_dict, key, 0.0) + weights[j]
        end
    end
    
    # Convert accumulated weights to sparse matrix format
    I = Int[]
    J = Int[]
    V = Float64[]
    
    for ((i, j), weight) in weight_dict
        push!(I, i)
        push!(J, j)
        push!(V, weight)
    end
    
    # Construct sparse matrix
    return sparse(I, J, V, mesh.edges.n, mesh.edges.n)
end

#---------------------------------------------------------------------------------------------------
# Least-squares interpolation

# Get the edges connected an edge vertex
function get_neighbors_1level_primal(verticesOnEdge::Vector{Tuple{T, T}}, edgesOnVertex::Vector{Tuple{T, T, T}}, edge_index::Int) where T
    v1, v2 = verticesOnEdge[edge_index]
    neighbors = Set{Int}()
    
    # Add the edge itself
    push!(neighbors, edge_index)
    
    # Add edges connected to v1
    for e in edgesOnVertex[v1]
        push!(neighbors, e)
    end
    
    # Add edges connected to v2
    for e in edgesOnVertex[v2]
        push!(neighbors, e)
    end
    return collect(neighbors)
end

get_neighbors_1level_primal(mesh::VoronoiMesh, edge_index::Int) = get_neighbors_1level_primal(mesh.edges.vertices, mesh.vertices.edges, edge_index)

function get_neighbors_1level_dual(mesh::VoronoiMesh)
    neighbors_list = Vector{Vector{Int}}(undef, mesh.edges.n)
    for i = 1:mesh.edges.n
        neighbors_list[i] = get_neighbors_1level_primal(mesh, i)
    end
    return neighbors_list
end

function get_neighbors_2level_primal(mesh::VoronoiMesh, edge_index::Int)
    neighbors_1level = get_neighbors_1level_primal(mesh, edge_index)
    neighbors_2level = Set{Int}()
    
    for e in neighbors_1level
        union!(neighbors_2level, get_neighbors_1level_primal(mesh, e))
    end
    return collect(neighbors_2level)
end

function get_neighbors_2level_dual(mesh::VoronoiMesh)
    neighbors_list = Vector{Vector{Int}}(undef, mesh.edges.n)
    for i in 1:mesh.edges.n
        neighbors_list[i] = get_neighbors_2level_primal(mesh, i)
    end
    return neighbors_list
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


function get_edge_info(mesh::VoronoiMesh, edge_index::Int, cochain::Cochain{T}) where T
    edge_position = mesh.edges.position[edge_index]

    if cochain.is_primal #If the cochain is primal, we need to get the triangle edges information
        v1, v2 = mesh.edges.cells[edge_index]
        p1_raw = TensorsLite.Vec(x = mesh.cells.position.x[v1], y = mesh.cells.position.y[v1])
        p2_raw = TensorsLite.Vec(x = mesh.cells.position.x[v2], y = mesh.cells.position.y[v2])
    else
        v1, v2 = mesh.edges.vertices[edge_index] #If the cochain is dual, we need to get the cells edges information
        p1_raw = TensorsLite.Vec(x = mesh.vertices.position.x[v1], y = mesh.vertices.position.y[v1])
        p2_raw = TensorsLite.Vec(x = mesh.vertices.position.x[v2], y = mesh.vertices.position.y[v2])
    end
    
    # Get closest periodic points relative to p1
    p1_periodic = closest(edge_position, p1_raw, mesh.x_period, mesh.y_period)
    p2_periodic = closest(edge_position, p2_raw, mesh.x_period, mesh.y_period)
    
    # Convert to 2D vectors
    p1 = [p1_periodic[1], p1_periodic[2]]
    p2 = [p2_periodic[1], p2_periodic[2]]
    
    edge_vector = p2 - p1
    tangent_vector = edge_vector
    midpoint = (p1 + p2) / 2
    value = cochain.values[edge_index]
    return (midpoint, value, tangent_vector)
end

function get_edge_info(mesh::VoronoiMesh, edge_index::Int, is_primal::Bool=true) 
    edge_position = mesh.edges.position[edge_index]

    if is_primal #If the cochain is primal, we need to get the triangle edges information
        v1, v2 = mesh.edges.cells[edge_index]
        p1 = mesh.cells.position[v1]
        p2 = mesh.cells.position[v2]
    else
        v1, v2 = mesh.edges.vertices[edge_index] #If the cochain is dual, we need to get the cells edges information
        p1 = mesh.vertices.position[v1]
        p2 = mesh.vertices.position[v2]
    end
    
    # Get closest periodic points relative to p1
    p1_periodic = closest(edge_position, p1, mesh.x_period, mesh.y_period)
    p2_periodic = closest(edge_position, p2, mesh.x_period, mesh.y_period)
    
    edge_vector = p2_periodic - p1_periodic
    tangent_vector = edge_vector
    midpoint = (p1_periodic + p2_periodic) / 2
    return (midpoint, tangent_vector)
end

function get_edge_info(mesh::VoronoiMesh, v::Cochain{T}) where T
    midpoints = Vector{Vector{Float64}}(undef, mesh.edges.n)
    values = Vector{Float64}(undef, mesh.edges.n)
    tangent_vectors = Vector{Vector{Float64}}(undef, mesh.edges.n)
    for i in 1:mesh.edges.n
        midpoints[i], values[i], tangent_vectors[i] = get_edge_info(mesh, i, v)
    end
    return midpoints, values, tangent_vectors
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

function rec_vector_field_lsq_5point(mesh::VoronoiMesh, cochain::Cochain{T}, edge_index::Int; verbose=true, hodge=false) where T
    """
    Reconstruct vector field using 5-point stencil least squares
    Returns coefficients [a0, a1, a2, a3, a4] for:
    v = (a0 + a2*x + a3*y, a1 + a4*x - a2*y)
    """
    neighbors = get_neighbors_1level_primal(mesh, edge_index)
    edge_infos = [get_edge_info(mesh, e, cochain) for e in neighbors]
    
    midpoints, values, unit_tangent_vectors = zip(edge_infos...)
    midpoints = hcat(midpoints...)'
    values = collect(values)
    unit_tangent_vectors = hcat(unit_tangent_vectors...)'
    
    # Construct the A matrix
    A = zeros(5, 5)
    for i in 1:5
        x, y = midpoints[i, :]
        nx, ny = unit_tangent_vectors[i, :]
        A[i, :] = [nx, ny, nx*x - ny*y, nx*y, ny*x]
    end
    
    # Solve the least squares problem
    coefs = inv(A) * values
    
    if verbose
        println("------Solving linear system-----")
        println("Residuals: ", norm(A * coefs - values))
        println("Rank: ", rank(A))
        println("Condition number: ", cond(A))
        println("----------------------------------")
    end
    
    if hodge
        reconstructed_vector = p -> [-coefs[2] - coefs[5]*p[1] + coefs[3]*p[2], 
                                     coefs[1] + coefs[3]*p[1] + coefs[4]*p[2]]
    else
        reconstructed_vector = p -> [coefs[1] + coefs[3]*p[1] + coefs[4]*p[2],
                                     coefs[2] + coefs[5]*p[1] - coefs[3]*p[2]]
    end
    
    return reconstructed_vector, coefs
end



function rec_vector_field_lsq_13point(mesh::VoronoiMesh, cochain::Cochain{T}, edge_index::Int; verbose=true, primal_to_dual=true, hodge=false, only_linear=true) where T
    x0 = mesh.edges.position[edge_index]
    neighbors = get_neighbors_2level_primal(mesh, edge_index)
    n = length(neighbors)
    edge_infos = [get_edge_info(mesh, e, cochain) for e in neighbors]
    _, values, _ = zip(edge_infos...)
    #midpoints = hcat(midpoints...)'
    values = collect(values)
    #unit_tangent_vectors = hcat(unit_tangent_vectors...)'

    # Construct the A matrix
    if only_linear
        neighbors = get_neighbors_2level_primal(mesh, edge_index)
        n = length(neighbors)
        edge_infos = [get_edge_info(mesh, e, cochain) for e in neighbors]
        _, values, _ = zip(edge_infos...)
        #midpoints = hcat(midpoints...)'
        values = collect(values)
        #unit_tangent_vectors = hcat(unit_tangent_vectors...)'
        A = zeros(n, 6)
    else
        neighbors = get_neighbors_3level_primal(mesh, edge_index)
        n = length(neighbors)
        edge_infos = [get_edge_info(mesh, e, cochain) for e in neighbors]
        _, values, _ = zip(edge_infos...)
        #midpoints = hcat(midpoints...)'
        values = collect(values)
        #unit_tangent_vectors = hcat(unit_tangent_vectors
        A = zeros(n, 12)
    end

    if primal_to_dual
        for i in 1:n
            if only_linear
                fields_values_primal = integrate_basic_fields_primal_edge(mesh, neighbors[i], x0, only_linear)
                A[i, :] = [fields_values_primal[1], fields_values_primal[2], fields_values_primal[3], fields_values_primal[4], fields_values_primal[5], fields_values_primal[6]]
            else
                fields_values_primal = integrate_basic_fields_primal_edge(mesh, neighbors[i], x0, only_linear)
                A[i, :] = [fields_values_primal[1], fields_values_primal[2], fields_values_primal[3], fields_values_primal[4], fields_values_primal[5], fields_values_primal[6], fields_values_primal[7], fields_values_primal[8], fields_values_primal[9], fields_values_primal[10], fields_values_primal[11], fields_values_primal[12]]
            end
        end
    else
        for i in 1:n
            if only_linear
                fields_values_dual = integrate_basic_fields_dual_edge(mesh, neighbors[i], x0, only_linear)
                A[i, :] = [fields_values_dual[1], fields_values_dual[2], fields_values_dual[3], fields_values_dual[4], fields_values_dual[5], fields_values_dual[6]]
            else
                fields_values_dual = integrate_basic_fields_dual_edge(mesh, neighbors[i], x0, only_linear)
                A[i, :] = [fields_values_dual[1], fields_values_dual[2], fields_values_dual[3], fields_values_dual[4], fields_values_dual[5], fields_values_dual[6], fields_values_dual[7], fields_values_dual[8], fields_values_dual[9], fields_values_dual[10], fields_values_dual[11], fields_values_dual[12]]
            end
        end
    end
    
    # Solve the least squares problem
    coefs = pinv(A) * values
    
    if verbose
        println("------Solving linear system for edge $edge_index-----")
        println("Residuals: ", norm(A * coefs - values))
        println("Rank: ", rank(A))
        println("Condition number: ", cond(A))
        println("----------------------------------")
    end
    
    if only_linear
        if hodge
            reconstructed_vector = p -> [-coefs[2] - coefs[5]*(p[1]-x0[1]) - coefs[6]*(p[2]-x0[2]), 
                                        coefs[1] + coefs[3]*(p[1]-x0[1]) + coefs[4]*(p[2]-x0[2])]
        else
            reconstructed_vector = p -> [coefs[1] + coefs[3]*(p[1]-x0[1]) + coefs[4]*(p[2]-x0[2]),
                                        coefs[2] + coefs[5]*(p[1]-x0[1]) + coefs[6]*(p[2]-x0[2])]  
        end

    else
        if hodge
            reconstructed_vector = p -> [-coefs[2] - coefs[5]*(p[1]-x0[1]) - coefs[6]*(p[2]-x0[2]) - coefs[10]*(p[1]-x0[1])^2 - coefs[11]*(p[1]-x0[1])*(p[2]-x0[2]) - coefs[12]*(p[2]-x0[2])^2, 
                                        coefs[1] + coefs[3]*(p[1]-x0[1]) + coefs[4]*(p[2]-x0[2]) + coefs[7]*(p[1]-x0[1])^2 + coefs[8]*(p[1]-x0[1])*(p[2]-x0[2]) + coefs[9]*(p[2]-x0[2])^2]
        else
            reconstructed_vector = p -> [coefs[1] + coefs[3]*(p[1]-x0[1]) + coefs[4]*(p[2]-x0[2]) + coefs[7]*(p[1]-x0[1])^2 + coefs[8]*(p[1]-x0[1])*(p[2]-x0[2]) + coefs[9]*(p[2]-x0[2])^2,
                                        coefs[2] + coefs[5]*(p[1]-x0[1]) + coefs[6]*(p[2]-x0[2]) + coefs[10]*(p[1]-x0[1])^2 + coefs[11]*(p[1]-x0[1])*(p[2]-x0[2]) + coefs[12]*(p[2]-x0[2])^2]  
        end
    end
    
    return reconstructed_vector, coefs, A
end

function integrate_lsq_reconstruction_on_edge(mesh::VoronoiMesh, cochain::Cochain{T}, edge_index::Int, primal_to_dual::Bool=true, hodge::Bool=true, only_linear::Bool=true) where T
    reconstructed_vector, _, _ = rec_vector_field_lsq_13point(mesh, cochain, edge_index, verbose=false, primal_to_dual=primal_to_dual, hodge=hodge, only_linear=only_linear)
    edge_pos = mesh.edges.position[edge_index]
    
    if primal_to_dual
        v1, v2 = mesh.edges.vertices[edge_index]
        p1_primal = closest(edge_pos, mesh.vertices.position[v1], mesh.x_period, mesh.y_period)
        p2_primal = closest(edge_pos, mesh.vertices.position[v2], mesh.x_period, mesh.y_period)
        p1_primal = [p1_primal[1], p1_primal[2]]
        p2_primal = [p2_primal[1], p2_primal[2]]

    else
        v1, v2 = mesh.edges.cells[edge_index]
        p1_primal = closest(edge_pos, mesh.cells.position[v1], mesh.x_period, mesh.y_period)
        p2_primal = closest(edge_pos, mesh.cells.position[v2], mesh.x_period, mesh.y_period)
        p1_primal = [p1_primal[1], p1_primal[2]]
        p2_primal = [p2_primal[1], p2_primal[2]]
    end
    integral_value = edge_integral(reconstructed_vector, p1_primal, p2_primal, mesh.x_period, mesh.y_period)
    return integral_value
end

function integrate_lsq_reconstruction(mesh::VoronoiMesh, cochain::Cochain{T}, primal_to_dual::Bool=true, only_linear::Bool=true) where T
    values = [integrate_lsq_reconstruction_on_edge(mesh, cochain, i, primal_to_dual, true, only_linear) for i in 1:mesh.edges.n]
    if primal_to_dual
        return Cochain(values, 1, false)
    else
        return Cochain(values, 1, true)
    end
end


function hodge_star_lsq_operator(mesh::VoronoiMesh, edge_index::Int, primal_to_dual::Bool=true, only_linear::Bool=true)
    x0 = mesh.edges.position[edge_index]
    
    # Construct the A matrix
    if only_linear==true
        neighbors = get_neighbors_2level_primal(mesh, edge_index)  #13 point stencil 
        n = length(neighbors)
        A = zeros(n, 6)
    else
        neighbors = get_neighbors_3level_primal(mesh, edge_index) #19 point stencil
        n = length(neighbors)
        A = zeros(n, 12)
    end

    if primal_to_dual #primal edge to dual edge
        for i in 1:n
            if only_linear
                fields_values_primal = integrate_basic_fields_primal_edge(mesh, neighbors[i], x0, only_linear, false)
                A[i, :] = [fields_values_primal[1], fields_values_primal[2], fields_values_primal[3], fields_values_primal[4], fields_values_primal[5], fields_values_primal[6]]
            else
                fields_values_primal = integrate_basic_fields_primal_edge(mesh, neighbors[i], x0, only_linear, false)
                A[i, :] = [fields_values_primal[1], fields_values_primal[2], fields_values_primal[3], fields_values_primal[4], fields_values_primal[5], fields_values_primal[6], fields_values_primal[7], fields_values_primal[8], fields_values_primal[9], fields_values_primal[10], fields_values_primal[11], fields_values_primal[12]]
            end
        end
        fields_values = integrate_basic_fields_dual_edge(mesh, edge_index, x0, only_linear, true)
    else #dual edge to primal edge
        for i in 1:n
            if only_linear
                fields_values_dual = integrate_basic_fields_dual_edge(mesh, neighbors[i], x0, only_linear, false)
                A[i, :] = [fields_values_dual[1], fields_values_dual[2], fields_values_dual[3], fields_values_dual[4], fields_values_dual[5], fields_values_dual[6]]
            else
                fields_values_dual = integrate_basic_fields_dual_edge(mesh, neighbors[i], x0, only_linear, false)
                A[i, :] = [fields_values_dual[1], fields_values_dual[2], fields_values_dual[3], fields_values_dual[4], fields_values_dual[5], fields_values_dual[6], fields_values_dual[7], fields_values_dual[8], fields_values_dual[9], fields_values_dual[10], fields_values_dual[11], fields_values_dual[12]]
            end
        end
        fields_values = integrate_basic_fields_primal_edge(mesh, edge_index, x0, only_linear, true)
    end 

    A_pinv = pinv(A)
    weights = fields_values' * A_pinv
    return weights, neighbors
end

function hodge_star_lsq_matrix(mesh::VoronoiMesh, primal_to_dual::Bool=true, only_linear::Bool=true)
    n = mesh.edges.n
    # Initialize arrays for sparse matrix construction
    I = Int[]  # Row indices
    J = Int[]  # Column indices
    V = Float64[]  # Values
    
    for i in 1:n
        weights, neighbors = hodge_star_lsq_operator(mesh, i, primal_to_dual, only_linear)
        # Add entries for each neighbor
        for (j, neighbor) in enumerate(neighbors)
            push!(I, i)
            push!(J, neighbor)
            push!(V, weights[j])
        end
    end
    
    return sparse(I, J, V, n, n)
end
#----------------------------------------------------------------------------------------
#Testing wedge functions
function wedge_product_01(mesh::VoronoiMesh, α::Cochain{T}, β::Cochain{T}) where T
    # Verify input forms
    if α.k != 0 || β.k != 1
        error("First input must be a 0-form and second input must be a 1-form")
    end
    
    # Initialize result array
    result = zeros(T, mesh.edges.n)
    
    # For each edge
    for i in 1:mesh.edges.n
        # Get vertices of the edge
        v1, v2 = mesh.edges.cells[i]
        
        # The wedge product on an edge is the average of the 0-form values 
        # at the vertices multiplied by the 1-form value on the edge
        α_avg = (α.values[v1] + α.values[v2]) / 2
        result[i] = α_avg * β.values[i]
    end
    return Cochain(result, 1, β.is_primal)
end

function wedge_product_10_matrix(mesh::VoronoiMesh, v::Cochain{T}) where T
    abs_d0 = abs.(d0_matrix_primal(mesh))
    return 0.5 * Diagonal(v.values) * abs_d0
end

function wedge_product_11(mesh::VoronoiMesh, α::Cochain{T}, β::Cochain{T}) where T
# Verify input forms
if α.k != 1 || β.k != 1
    error("Both inputs must be 1-forms")
end

# Initialize result array (one value per triangle/cell)
result = zeros(T, mesh.vertices.n)

# For each triangle
for i in 1:mesh.vertices.n
    # Get edges of the triangle
    _, edge_idx, edge_signs= get_triangle(mesh, i)
    
    # For a triangle with edges e1, e2, e3, the wedge product is:
    # (α₁β₂ - α₂β₁) + (α₂β₃ - α₃β₂) + (α₃β₁ - α₁β₃)
    # where subscripts refer to edge indices
    
    # Apply edge orientations to form values
    α_values = edge_signs .* α.values[edge_idx]
    β_values = edge_signs .* β.values[edge_idx]
    
    # Compute wedge product
    result[i] = (α_values[1] * β_values[2] - α_values[2] * β_values[1] +
                α_values[2] * β_values[3] - α_values[3] * β_values[2] +
                α_values[3] * β_values[1] - α_values[1] * β_values[3]) / 6
end

    return Cochain(result, 2, true)
end

function wedge_product_11_weights(mesh::VoronoiMesh, v::Cochain{T}, triangle_idx::Int) where T
    _, edge_idx, edge_signs= get_triangle(mesh, triangle_idx)

    # Apply edge orientations to form values
    v_values = edge_signs .* v.values[edge_idx]
    weights = edge_signs .* [(1/6)*(v_values[3] - v_values[2]), (1/6)*(v_values[1] - v_values[3]), (1/6)*(v_values[2] - v_values[1])]
    return weights, edge_idx
end

function wedge_product_11_matrix(mesh::VoronoiMesh, v::Cochain{T}) where T
    I = Int[]
    J = Int[]
    V = T[]
    n = mesh.vertices.n
    for i = 1:n
        weights, edge_idx = wedge_product_11_weights(mesh, v, i)
        for j = 1:3
            push!(I, i)
            push!(J, edge_idx[j])
            push!(V, weights[j])
        end
    end
    return sparse(I, J, V)
end

function wedge(mesh::VoronoiMesh, α::Cochain{T}, β::Cochain{T}) where T
    if α.k == 1 && β.k == 0
        return wedge_product_10(mesh, α, β)
    elseif α.k == 0 && β.k == 1
        return wedge_product_01(mesh, α, β)
    elseif α.k == 1 && β.k == 1
        return wedge_product_11(mesh, α, β)
        error("Invalid wedge product")
    end
end
####################################################################
######################### POST PROCESSING ##########################
####################################################################

function get_square_vertices(mesh::VoronoiMesh)
    # Get the domain dimensions from the mesh
    x_period = mesh.x_period
    y_period = mesh.y_period
    h = maximum(mesh.edges.lengthDual)

    # Calculate the number of squares in each direction
    # Each square has side length h/4
    square_size = h/4
    nx = ceil(Int, x_period / square_size)
    ny = ceil(Int, y_period / square_size)
    
    # Adjust square_size to exactly fit the domain
    square_size_x = x_period / nx
    square_size_y = y_period / ny
    
    # Initialize array to store square vertices
    squares = Vector{Vector{Vector{Float64}}}()
    
    # Generate the squares
    for i in 0:(nx-1)
        for j in 0:(ny-1)
            # Calculate the four vertices of this square
            v1 = [i * square_size_x, j * square_size_y]
            v2 = [(i+1) * square_size_x, j * square_size_y]
            v3 = [(i+1) * square_size_x, (j+1) * square_size_y]
            v4 = [i * square_size_x, (j+1) * square_size_y]
            
            # Calculate the midpoint
            midpoint = [(i+0.5) * square_size_x, (j+0.5) * square_size_y]
            
            # Add the square (vertices and midpoint)
            push!(squares, [v1, v2, v3, v4, midpoint])
        end
    end
    
    return squares
end

function assign_cochain_values_to_squares(mesh::VoronoiMesh, cochain::Cochain{T}) where T
    # Check if the cochain is defined on edges
    if cochain.k != 1
        error("This function only works with 1-cochains (defined on edges)")
    end
    
    # Get squares with their vertices and midpoints
    squares = get_square_vertices(mesh)
    
    # Initialize array to store values for each square
    square_values = Vector{T}(undef, length(squares))
    
    # For each square, find the closest edge and assign its cochain value
    for (i, square) in enumerate(squares)
        # Get the midpoint of the square (5th element in the square array)
        midpoint = square[5]
        
        # Find the closest edge
        closest_edge_idx = 1
        min_distance = Inf
        
        for edge_idx in 1:mesh.edges.n
            # Get edge position
            edge_pos = mesh.edges.position[edge_idx]
            
            # Calculate distance to midpoint
            distance = sqrt((edge_pos[1] - midpoint[1])^2 + (edge_pos[2] - midpoint[2])^2)

            # Update closest edge if this one is closer
            if distance < min_distance
                min_distance = distance
                closest_edge_idx = edge_idx
            end
        end
        
        # Assign the cochain value of the closest edge to this square
        square_values[i] = cochain.values[closest_edge_idx]
    end
    
    return squares, square_values
end

function plot_cochain_on_squares(mesh::VoronoiMesh, cochain::Cochain{T}, colormap::Symbol=:viridis, title::String="") where T
    # Get squares and their values
    squares, values = assign_cochain_values_to_squares(mesh, cochain)
    
    # Create a figure
    fig = Figure(size=(800, 800))
    ax = Axis(fig[1, 1], aspect=1, title=title)
    
    # Normalize values for coloring
    min_val = minimum(values)
    max_val = maximum(values)
    
    # Plot each square with its value as color
    for (i, square) in enumerate(squares)
        # Extract vertices
        vertices = square[1:4]
        
        # Add the first vertex again to close the polygon
        push!(vertices, vertices[1])
        
        # Extract x and y coordinates
        x_coords = [v[1] for v in vertices]
        y_coords = [v[2] for v in vertices]
        
        # Plot the square
        poly!(ax, x_coords, y_coords, color=values[i], colormap=colormap, colorrange=(min_val, max_val))
    end
    
    # Add a colorbar
    Colorbar(fig[1, 2], limits=(min_val, max_val), colormap=colormap)
    
    return fig, ax
end

#----------------------------------------------------------------------------------------------------

function compute_error_on_circumcenter(mesh::VoronoiMesh, v::Matrix{T}, field::Function) where T
    points = [[mesh.vertices.position.x[i], mesh.vertices.position.y[i]] for i in 1:mesh.vertices.n]
    field_on_points = field.(points)
    errors = [v[i,:] - field_on_points[i] for i in 1:mesh.vertices.n]   
    return norm.(errors)
end

function L2_errorOnVertices(mesh::VoronoiMesh, a::Vector{T}, b::Vector{T}) where T
    errors = [((a[i] - b[i])^2) * mesh.cells.area[i] for i in 1:mesh.cells.n]
    return sqrt(sum(errors))
end

function L2_errorOnEdges(mesh::VoronoiMesh, a::Vector{T}, b::Vector{T}) where T
    errors = [((a[i] - b[i])^2) * (mesh.edges.lengthDual[i]/mesh.edges.length[i]) for i in 1:mesh.edges.n]
    return sqrt(sum(errors))
end

function test_wedge_matrices(mesh::VoronoiMesh)
    println("Testing wedge product matrix constructions...")
    
    # Test 1: wedge_product_10_matrix
    v = Cochain(rand(mesh.cells.n), 0, true)  # Random 0-form
    w = Cochain(rand(mesh.edges.n), 1, true)  # Random 1-form
    
    direct_result = wedge_product_01(mesh, v, w)
    matrix_result = wedge_product_10_matrix(mesh, w) * v.values
    
    error_10 = maximum(abs.(direct_result.values - matrix_result))
    println("Wedge 1-0 matrix error: ", error_10)
    
    # Test 2: wedge_product_11_matrix
    v = Cochain(rand(mesh.edges.n), 1, true)  # random 1-form
    w = Cochain(rand(mesh.edges.n), 1, true)  # random 1-form
    
    direct_result = wedge_product_11(mesh, v, w)
    matrix_result = wedge_product_11_matrix(mesh, v) * w.values
    
    error_11 = maximum(abs.(direct_result.values - matrix_result))
    println("Wedge 1-1 matrix error: ", error_11)
    
    return error_10, error_11
end

function plot_neighbors(mesh::VoronoiMesh, edge_index::Int)
    neighbors = get_neighbors_3level_primal(mesh, edge_index)
    fig = Figure()
    ax  = Axis(fig[1,1], 
        aspect=DataAspect(),
        xlabel="x",
        ylabel="y",
        title="Solution",
        titlesize=20,          # Title font size
        xlabelsize=16,         # x-axis label font size
        ylabelsize=16,         # y-axis label font size
        xticklabelsize=12,     # x-axis tick label font size
        yticklabelsize=12,
        xticks = 0:0.2:1.0,
        yticks = 0:0.2:1.0)  

    for i in 1:mesh.vertices.n
        points, _, _ = get_triangle(mesh, i)
        # Add the first point again to close the triangle
        points_x = [p[1] for p in points]
        points_y = [p[2] for p in points]
        push!(points_x, points_x[1])
        push!(points_y, points_y[1])
        poly!(ax, points_x, points_y, color="white", strokewidth=0.2, strokecolor="black")
    end
    
    for neighbor in neighbors
        if neighbor != edge_index
            scatter!(ax, [mesh.edges.position[neighbor][1]], [mesh.edges.position[neighbor][2]], color="red")
        end
        scatter!(ax, [mesh.edges.position[edge_index][1]], [mesh.edges.position[edge_index][2]], color="blue")
    end
    CairoMakie.save("neighbors.png", fig)
    return fig
end
"""
    plot_cochain(cochain::Cochain, mesh::VoronoiMesh, colormap::Symbol=:viridis)

Plot the triangle mesh showing the simplices. Optionally show vertex indices and color triangles.
"""
function plot_cochain!(ax, cochain::Cochain, mesh::VoronoiMesh, colormap::Symbol=:viridis) 
    
    # Plot each triangle

    if cochain.is_primal
        
        mean_values = [sum(cochain.values[collect(mesh.vertices.cells[i])])/3 for i in 1:mesh.vertices.n]

        for i in 1:mesh.vertices.n
            points, _, _ = get_triangle(mesh, i)
            # Add the first point again to close the triangle
            points_x = [p[1] for p in points]
            points_y = [p[2] for p in points]
            push!(points_x, points_x[1])
            push!(points_y, points_y[1])
        end

        if cochain.k == 0
            vmin, vmax = extrema(mean_values)
            poly!(ax, points_x, points_y, color=mean_values[i], colormap=colormap, colorrange=(vmin, vmax))
            vlines!(ax, 0.8, color=:black, linestyle=:dash)
            Colorbar(fig[1,2], limits=(vmin, vmax), colormap=colormap)
        
        elseif cochain.k == 2
            for i in 1:mesh.vertices.n
                points, _, _ = get_triangle(mesh, i)
                # Add the first point again to close the triangle
                points_x = [p[1] for p in points]
                points_y = [p[2] for p in points]
                push!(points_x, points_x[1])
                push!(points_y, points_y[1])
                vmin, vmax = extrema(cochain.values)
                poly!(ax, points_x, points_y, color=cochain.values[i], colormap=colormap, colorrange=(vmin, vmax))
            
            end

            Colorbar(fig[1,2], limits=(vmin, vmax), colormap=colormap)
        end
    else 
        # values at vertices of dual cells. Because its value is over the triangle
        #circumcenter, we color the triangle by the value of the cochain.
        
        if cochain.k == 0
            for i in 1:mesh.vertices.n
                points, _, _ = get_triangle(mesh, i)
                # Add the first point again to close the triangle
                points_x = [p[1] for p in points]
                points_y = [p[2] for p in points]
                push!(points_x, points_x[1])
                push!(points_y, points_y[1])
            end
            vmin, vmax = extrema(cochain.values)
            poly!(ax, points_x, points_y, color=cochain.values[i], colormap=colormap, colorrange=(vmin, vmax))
            vlines!(ax, 0.8, color=:black, linestyle=:dash)
            Colorbar(fig[1,2], limits=(vmin, vmax), colormap=colormap)

        elseif cochain.k == 2
            vmin, vmax = extrema(cochain.values)
            for i in 1:mesh.cells.n
                points = get_cells_points(mesh, i)
                points_x = [p[1] for p in points]
                points_y = [p[2] for p in points]
                push!(points_x, points_x[1])
                push!(points_y, points_y[1])
                poly!(ax, points_x, points_y, color=cochain.values[i], colormap=colormap, colorrange=(vmin, vmax))
            end
            Colorbar(fig[1,2], limits=(vmin, vmax), colormap=colormap)
        end
    end
    #hidedecorations!(ax)
    return ax
end


#---------------------------------------------------------------------------------------------------
#mesh = VoronoiMesh("meshes/mesh2.nc")
#verticesOnEdge = mesh.edges.vertices
#cellsOnEdge = mesh.edges.cells
#egesOnCell = mesh.cells.edges
#xp = mesh.x_period
#yp = mesh.y_period

#checking sparsity of d0 and d1 ------------------------------------------------------------
#print(d0_matrix_primal(verticesOnEdge, mesh.nEdges))
#p = spy(Matrix(d0_matrix_primal(verticesOnEdge, mesh.nEdges)))
#savefig("spy_d0_primal.png")
#print(Matrix(d0_matrix_primal(verticesOnEdge, mesh.nEdges)))
#print(d1_matrix_primal(cellsOnEdge, edgesOnCell, mesh.nCells, mesh.nEdges))
#p = spy(Matrix(d1_matri x_primal(cellsOnEdge, edgesOnCell, mesh.nCells, mesh.nEdges)))
#savefig("spy_d1_primal.png")
#print(Matrix(d1_matrix_primal(cellsOnEdge, edgesOnCell, mesh.nCells, mesh.nEdges)))
#---------------------------------------------------------------------------------------------------

#checking if d primal and d dual are adjoint with a sign
#println(d0_matrix_dual(mesh) == transpose(d1_matrix_primal(mesh)))
#println(d1_matrix_dual(mesh) == -transpose(d0_matrix_primal(mesh)))
#---------------------------------------------------------------------------------------------------

#checking d
#println(mesh.nVertices, ' ', mesh.nEdges, ' ', mesh.nCells)
#v = dual_cochain(ones(mesh.nCells), 0)
#println(d(mesh, v))
#v = dual_cochain(ones(mesh.nEdges), 1) # It is not zero because the triangle has 3 edges
#println(d(mesh, v))
#v = primal_cochain(ones(mesh.nVertices), 0)
#println(size(d(mesh, v).values))
#v = primal_cochain(ones(mesh.nEdges), 1)
#println(size(d(mesh, v).values))
#---------------------------------------------------------------------------------------------------

# d^2 = 0
#println(size(d0_matrix_primal(mesh)), size(d1_matrix_primal(mesh)), size(d0_matrix_dual(mesh)), size(d1_matrix_dual(mesh)))#
#println(maximum(abs.(d1_matrix_dual(mesh) * d0_matrix_dual(mesh) * Cochain(rand(mesh.vertices.n), 0, false).values)))
#println(maximum(abs.(d(mesh, d(mesh, Cochain(rand(mesh.vertices.n), 0, false))).values)))

#println(maximum(abs.(d1_matrix_primal(mesh) * d0_matrix_primal(mesh) * Cochain(rand(mesh.cells.n), 0, true).values)))
#println(maximum(abs.(d(mesh, d(mesh, Cochain(rand(mesh.cells.n), 0, true))).values)))

#println(norm(d1_matrix_primal(mesh) * d0_matrix_primal(mesh) * Cochain(rand(mesh.cells.n), 0, true).values - d(mesh, d(mesh, Cochain(rand(mesh.cells.n), 0, true))).values))

#mesh = VoronoiMesh("meshes/mesh2.nc")
#xp = mesh.x_period
#yp = mesh.y_period
#u = x -> sin(2*pi*x[1]/xp) * sin(2*pi*x[2]/yp) * (1/2*pi)
#u_primal = Cochain(u.(mesh.cells.position), 0, true)
#u_dual = Cochain(u.(mesh.vertices.position), 0, false)
#du = d(mesh, u_primal)
#println("du primal: ", maximum(abs.(du.values)))
#du_dual = d(mesh, u_dual)
#println("du dual: ", maximum(abs.(du_dual.values)))
#d2u = d(mesh, d(mesh, u_primal))
#println("d2u primal: ", maximum(abs.(d2u.values)))
#d2u_dual = d(mesh, d(mesh, u_dual))
#println("d2u dual: ", maximum(abs.(d2u_dual.values)))

#print(d0_matrix_dual(mesh) == transpose(d1_matrix_primal(mesh)))
#print(d1_matrix_dual(mesh) == -transpose(d0_matrix_primal(mesh)))


#---------------------------------------------------------------------------------------------------

#Testing Hodge star
#v = primal_cochain(ones(mesh.nVertices), 0)
#println(star(mesh, star(mesh, v)).values == v.values)

#v = primal_cochain(ones(mesh.nEdges), 1)
#println(star(mesh, star(mesh, v)).values == -v.values)

#v = primal_cochain(ones(mesh.nCells), 2)
#println(star(mesh, star(mesh, v)).values == v.values)

#v = dual_cochain(ones(mesh.nCells), 0)
#println(star(mesh, star(mesh, v)).values == v.values)

#v = dual_cochain(ones(mesh.nEdges), 1)
#println(star(mesh, star(mesh, v)).values == - v.values)

#v = dual_cochain(ones(mesh.nVertices), 2)
#println(star(mesh, star(mesh, v)).values == v.values)


#---------------------------------------------------------------------------------------------------

#Testing laplace_de_rham
#v = primal_cochain(ones(mesh.nVertices), 0)
#println(laplace_de_rham(mesh, v))

#v = dual_cochain(ones(mesh.nCells), 0)
#println(laplace_de_rham(mesh, v))
#---------------------------------------------------------------------------------------------------

#Testing edge_integral
#v = x -> [0, 1]
#println(discretize_vector_field_dual(mesh, v).values)
#println("\n")
#println("values: ", d(mesh, discretize_vector_field_dual(mesh, v)).values)

#Testing whitney_basis_functions
#mesh = VoronoiMesh("meshes/mesh2_regular.nc")
#basis_functions = whitney_basis_functions(mesh, false)
#v = discretize_vector_field_dual(mesh, x -> [1.0, 0.0])
#println(whitney_interpolation(mesh, v))    

#test dual edge generation
#dual_edges = construct_dual_edges(cellsOnEdge)
#println(dual_edges)

#test plot of whitney basis functions------------------------------------------------------
#function plot_whitney_basis_functions(n=14)
#    p1 = [0.0, 0.0]
#    p2 = [1.0, 0.0]
#    p3 = [0.0, 1.0]
#    rec(x) = whitney_reconstruction_on_x(x, p1, p2, p3, false)[3,:]
#  
#    points = vec(collect(Iterators.product(range(-0.1, 1.1, length=n), range(-0.1, 1.1, length=n))))
#    values = zeros(length(points), 2)
#    for i in 1:length(points)
#        values[i,:] = rec(points[i])
#    end
#    fig = Figure(size=(800, 600))
#    ax = Axis(fig[1, 1])
#    x = map(x -> x[1], points)
#    y = map(x -> x[2], points)
#    arrows!(ax, x, y, 0.1*values[:,1], 0.1*values[:,2])
#    lines!([p1[1], p2[1], p3[1], p1[1]], [p1[2], p2[2], p3[2], p1[2]], color=:red)
#    save("whitney_basis_functions.png", fig)
#
#end
#plot_whitney_basis_functions(14)

#test get_triangle------------------------------------------------------
#points1, edge_indices1, edge_orientations1 = get_triangle(mesh, 4)
#points2, edge_indices2, edge_orientations2 = get_triangle(mesh, 5)
#x1 = map(x -> x[1], points1)
#y1 = map(x -> x[2], points1)
#x2 = map(x -> x[1], points2)
#y2 = map(x -> x[2], points2)

#-----------------------------------------------------------------------------------------------------------
#   Testing whitney basis functions
#-----------------------------------------------------------------------------------------------------------

#triangle_idx = 2
#field = x -> [sin(2*pi*x[1]/xp) * cos(2*pi*x[2]/yp), -cos(2*pi*x[1]/xp) * sin(2*pi*x[2]/yp)]
#field = x -> [1, 2]
#println(whitney_integration_on_edge(mesh, 10))
#println(whitney_integration_on_edge(mesh, 11))
#println(whitney_integration_on_edge(mesh, 12))
#v = discretize_vector_field_dual(mesh, field)
#println(whitney_interpolation_on_triangle(basis_functions, edge_indices, edge_orientations, v.values))
#println(whitney_basis_functions(mesh, false))
#v_interpolated = whitney_interpolation(mesh, v)
#println(compute_error_on_circumcenter(mesh, v_interpolated, field))

#field = x -> [sin(2*pi*x[1]/xp) * cos(2*pi*x[2]/yp), -cos(2*pi*x[1]/xp) * sin(2*pi*x[2]/yp)]
#hodge_field = x -> [cos(2*pi*x[1]/xp) * sin(2*pi*x[2]/yp), sin(2*pi*x[1]/xp) * cos(2*pi*x[2]/yp)]
#v = discretize_vector_field_dual(mesh, field)
#v_hodge = discretize_vector_field_primal(mesh, hodge_field)
#println(whitney_hodge_star_operator(mesh, v).values - v_hodge.values)

#order of convergence of whitney interpolation over circumcenter
#for i = 1:7
#    mesh = VoronoiMesh("meshes/mesh_" * string(i) * ".nc")
#    field = x -> [sin(2*pi*x[1]/mesh.attributes[:x_period]) * cos(2*pi*x[2]/mesh.attributes[:y_period]), -cos(2*pi*x[1]/mesh.attributes[:x_period]) * sin(2*pi*x[2]/mesh.attributes[:y_period])]
#    v = discretize_vector_field_dual(mesh, field)
#    v_interpolated = whitney_interpolation(mesh, v)
#    println("number of triangles: ", mesh.nCells, " number of edges: ", mesh.nEdges, " number of vertices: ", mesh.nVertices)
#    println(maximum(compute_error_on_circumcenter(mesh, v_interpolated, field)))
#end

#testing hodge star matrix
#mesh = VoronoiMesh("meshes/mesh_2.nc")
#xp = mesh.attributes[:x_period]
#yp = mesh.attributes[:y_period]
#field = x -> [sin(2*pi*x[1]/xp) * cos(2*pi*x[2]/yp), -cos(2*pi*x[1]/xp) * sin(2*pi*x[2]/yp)]
#v = discretize_vector_field_dual(mesh, field)
#hodge_star_matrix = whitney_hodge_star_matrix(mesh)
#println(norm(hodge_star_matrix * v.values - whitney_hodge_star_operator(mesh, v).values))

#fig = Figure()
#ax = Axis(fig[1,1])
#spy!(ax, Matrix(hodge_star_matrix))
#save("hodge_star_matrix.png", fig)


#-----------------------------------------------------------------------------------------------------------
#   Testing least squares hodge star
#-----------------------------------------------------------------------------------------------------------
#mesh = VoronoiMesh("meshes/mesh3.nc")
#xp = mesh.x_period
#yp = mesh.y_period
#println(typeof(mesh.edges.position))
#Testing get_neighbors_1level_primal and get_neighbors_2level_primal-------------------------------------------
#v = discretize_vector_field_primal(mesh, x -> [2, -1])
#idx = 49  
#println(get_neighbors_1level_primal(mesh, idx))
#println(get_neighbors_2level_primal(mesh, idx)) 
#println(get_neighbors_3level_primal(mesh, idx))
#println(get_edge_info(mesh, 289, v))
#plot_neighbors(mesh, idx)
# Testing rec_vector_field_lsq

#v = discretize_vector_field_dual(mesh, x -> [1 + x[1], -1 + x[2]^2])
#v_dual = discretize_vector_field_primal(mesh, x -> [1 - x[2]^2, 1 + x[1]])
#midpoints, values, tangent_vectors = get_edge_info(mesh, v)
#reconstructed_vector, coefs, _ = rec_vector_field_lsq_13point(mesh, v, 8, verbose=true, primal_to_dual=true, hodge=true, only_linear=false)
#println("Interpolation coefs: ", coefs)
#println("Interpolation coefficients: \n a0 = ", coefs[1], "\n a1 = ", coefs[2], "\n a2 = ", coefs[3], "\n a3 = ", coefs[4], "\n a4 = ", coefs[5], "\n a5 = ", coefs[6], "\n\n")
#rec_values = integrate_lsq_reconstruction(mesh, v, true, false)
# Checking for the error of the interpolation
#println(maximum(abs.(discretize_vector_field_primal(mesh, reconstructed_vector).values - v_dual.values)))
#println(maximum(abs.(rec_values.values - v_dual.values)))
#println(maximum(abs.(hodge_star_lsq_matrix(mesh, true, false) * v.values - v_dual.values)))
#println(size(hodge_star_lsq_matrix(mesh, true, false)))

#Checking order of convergence


#println(v.values)
#Checking if the hodge star operator is correct
#println(maximum(abs.(hodge_star_lsq_matrix(mesh) * v.values - integrate_lsq_field(mesh, v))))

# Testing integrate_basic_fields
#mesh = VoronoiMesh(13, 1.0, 1.0)
#println(integrate_basic_fields(mesh))

# -----------------------------------------------------------------------------------------------------------
# Testing triangle_integral and wedge products
# -----------------------------------------------------------------------------------------------------------

#testing wedge between 0-form and 1-form
#mesh = VoronoiMesh("meshes/mesh1_regular.nc")
#xp = mesh.x_period
#yp = mesh.y_period
#v = discretize_vector_field_dual(mesh, x -> [1.0, -1.0])
#w = Cochain(-2.0*ones(mesh.vertices.n), 0, true)
#ans = discretize_vector_field_dual(mesh, x -> [-2.0, 2.0])
#println(maximum(abs.(wedge_product_01(mesh, w, v).values - ans.values)))

#Test wedge between 1-form and 1-form
#field = x -> [sin(2*pi*x[1]/xp) * cos(2*pi*x[2]/yp), -cos(2*pi*x[1]/xp) * sin(2*pi*x[2]/yp)]
#v = discretize_vector_field_dual(mesh, field)
#println(maximum(abs.(wedge(mesh, v, v).values)))

#testing constant 1-forms
#v1 = discretize_vector_field_dual(mesh, x -> [1.0, 1.0])
#v2 = discretize_vector_field_dual(mesh, x -> [-1.0, 1.0])
#ans = discretize_scalar_field_triangles(mesh, x -> 2.0)
#println(triangle_integral(x->x[1]^2 + x[2]^2, [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]))
#println(abs.(wedge(mesh, v1, v2).values - ans.values))

#Check for consistency of wedge product for wedge 0-form and 1-form
#println("-------------------------------------------------------------------------------------------")
#println("Checking for consistency of wedge product for wedge 0-form and 1-form...")
#error_list_Linf = []
#edge_length_list = []

#for i = 1:7
#    mesh = VoronoiMesh("meshes/mesh" * string(i) * "_regular.nc")
#    field1 = x -> sin(2*pi*x[1]/mesh.x_period) * cos(2*pi*x[2]/mesh.y_period)
#    field2 = x -> [10.0, -1.0]
#    field3 = x -> [10*sin(2*pi*x[1]/mesh.x_period) * cos(2*pi*x[2]/mesh.y_period), -sin(2*pi*x[1]/mesh.x_period) * cos(2*pi*x[2]/mesh.y_period)]
#    u = Cochain(field1.(mesh.cells.position), 0, true)
#    v = discretize_vector_field_dual(mesh, field2)
#    ans = discretize_vector_field_dual(mesh, field3)
#    push!(error_list_Linf, maximum(abs.(wedge(mesh, u, v).values - ans.values)))   
#    println(maximum(abs.(wedge(mesh, u, v).values - ans.values))) 
#    push!(edge_length_list, maximum(mesh.edges.lengthDual))
#end

#for i = 2:length(error_list_Linf)
#    order = log(error_list_Linf[i]/error_list_Linf[i-1]) / log(edge_length_list[i]/edge_length_list[i-1])
#    println("Order Linf between meshes $(i-1) and $i: $order")
#end

#Check for consistency of wedge product for wedge 1-form and 1-form
#println("-------------------------------------------------------------------------------------------")
#println("Checking for consistency of wedge product for wedge 1-form and 1-form...")
#error_list_Linf = []
#edge_length_list = []

#for i = 1:7
#    mesh = VoronoiMesh("meshes/mesh" * string(i) * "_regular.nc")
#    field1 = x -> [sin(2*pi*x[1]/mesh.x_period) * cos(2*pi*x[2]/mesh.y_period), -cos(2*pi*x[1]/mesh.x_period) * sin(2*pi*x[2]/mesh.y_period)]
#    field2 = x -> [0.0, 20.0]
#    field3 = x -> 20*sin(2*pi*x[1]/mesh.x_period) * cos(2*pi*x[2]/mesh.y_period)
#    u = discretize_vector_field_dual(mesh, field1)
#    v = discretize_vector_field_dual(mesh, field2)
#    ans = discretize_scalar_field_triangles(mesh, field3)
#    push!(error_list_Linf, maximum(abs.(wedge(mesh, u, v).values - ans.values)))    
#    println(maximum(abs.(wedge(mesh, u, v).values - ans.values)))
#    push!(edge_length_list, maximum(mesh.edges.lengthDual))
#end

#for i = 2:length(error_list_Linf)
#    order = log(error_list_Linf[i]/error_list_Linf[i-1]) / log(edge_length_list[i]/edge_length_list[i-1])
#    println("Order Linf between meshes $(i-1) and $i: $order")
#end

#Another check for consistency of wedge product for wedge 1-form and 1-form
#println("-------------------------------------------------------------------------------------------")
#println("Another check for consistency of wedge product for wedge 1-form and 1-form...")
#error_list_Linf = []
#edge_length_list = []

#for i = 1:7
#    mesh = VoronoiMesh("meshes/mesh" * string(i) * "_regular.nc")
    #field1 = x -> [sin(2*pi*x[1]/mesh.x_period) * cos(2*pi*x[2]/mesh.y_period), -cos(2*pi*x[1]/mesh.x_period) * sin(2*pi*x[2]/mesh.y_period)]
    #field2 = x -> [1.0, 2.0]
    #field3 = x -> cos(2*pi*x[1]/mesh.x_period) * sin(2*pi*x[2]/mesh.y_period) + 2.0*sin(2*pi*x[1]/mesh.x_period) * cos(2*pi*x[2]/mesh.y_period)
    
#    field1 = x -> [x[1] + x[2], x[2] - x[1]]
#    field2 = x -> [x[2], x[1]]
#    field3 = x -> x[1]^2 - x[2]^2 + 2*x[1]*x[2]
    
#    u = discretize_vector_field_dual(mesh, field1)
#    v = discretize_vector_field_dual(mesh, field2)
#    ans = discretize_scalar_field_triangles(mesh, field3)
#    push!(error_list_Linf, maximum(abs.(wedge(mesh, u, v).values - ans.values)))    
#    println(maximum(abs.(wedge(mesh, u, v).values - ans.values)))
#    push!(edge_length_list, maximum(mesh.edges.lengthDual))
#end
#
#for i = 2:length(error_list_Linf)
#    order = log(error_list_Linf[i]/error_list_Linf[i-1]) / log(edge_length_list[i]/edge_length_list[i-1])
#    println("Order Linf between meshes $(i-1) and $i: $order")
#end
#
#mesh = VoronoiMesh("meshes/mesh2.nc")
#println(test_wedge_matrices(mesh))
