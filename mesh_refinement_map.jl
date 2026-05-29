using VoronoiMeshes, TensorsLite, TensorsLiteGeometry, LinearAlgebra

"""
    is_point_in_triangle(mesh::VoronoiMesh, tri_idx::Int, point::Vector{Float64})

Checks if a point is inside a specific triangle of the mesh.
"""
function is_point_in_triangle(mesh::VoronoiMesh, tri_idx::Union{Int32, Int64}, point::Vector{Float64})
    points, _, _ = get_triangle(mesh, tri_idx)
    p1, p2, p3 = points
    
    x, y = point
    
    # Barycentric coordinates
    # P = l0*p1 + l1*p2 + l2*p3
    
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    
    # Actually reusing the formula from whitney code is safer:
    denom = (y2 - y3) * x1 + (x3 - x1) * y2 + (x1 - x2) * y3
    
    if abs(denom) < 1e-10 return false end
    
    l0 = ((y2 - y3) * x + (x3 - x2) * y + x2 * y3 - x3 * y2) / denom
    l1 = ((y3 - y1) * x + (x1 - x3) * y + x3 * y1 - x1 * y3) / denom
    l2 = 1.0 - l0 - l1
    
    # Allow small epsilon for boundary cases
    eps = -1e-10
    return l0 >= eps && l1 >= eps && l2 >= eps
end

"""
    find_coarse_triangle_index(mesh::VoronoiMesh, point::Vector{Float64})

Finds the index of the triangle in `mesh` that contains `point`.
Uses neighborhood search optimization.
"""
function find_triangle_index(mesh::VoronoiMesh, point::Vector{Float64})
    xp, yp = mesh.x_period, mesh.y_period
    p_vec = TensorsLite.Vec(x=point[1], y=point[2])

    # Find nearest vertex (generator) in mesh to use as a seed
    min_dist = Inf
    nearest_v = 1

    for i in 1:mesh.cells.n
        d = norm(closest(mesh.cells.position[i], p_vec, xp, yp) - p_vec)
        if d < min_dist
            min_dist = d
            nearest_v = i
        end
    end
    
    incident_triangles = mesh.cells.vertices[nearest_v]
    
    for tri_idx in incident_triangles
        tri_center = TensorsLite.Vec(x=mesh.vertices.position.x[tri_idx], 
                                     y=mesh.vertices.position.y[tri_idx])
        p_mapped_vec = closest(tri_center, p_vec, xp, yp)
        p_mapped = [p_mapped_vec.x, p_mapped_vec.y]
        
        if is_point_in_triangle(mesh, tri_idx, p_mapped)
            return tri_idx
        end
    end
    
    return 0 #Not found
end

"""
    map_fine_to_coarse(coarse_mesh::VoronoiMesh, fine_mesh::VoronoiMesh)

Returns a vector where mapping[i] is the index of the coarse triangle containing fine triangle i.
"""
function map_fine_to_coarse(coarse_mesh::VoronoiMesh, fine_mesh::VoronoiMesh)
    n_fine = fine_mesh.vertices.n
    mapping = zeros(Int, n_fine)
    
    println("Mapping $n_fine fine triangles to coarse mesh...")
    
    for i in 1:n_fine
        # Use centroid of fine triangle for robustness
        points, _, _ = get_triangle(fine_mesh, i)
        centroid = (points[1] + points[2] + points[3]) / 3.0
        
        coarse_idx = find_coarse_triangle_index(coarse_mesh, centroid)
        mapping[i] = coarse_idx
    end
    
    return mapping
end

"""
    get_children_triangles(coarse_mesh, fine_mesh)

Returns a Dictionary: Coarse Triangle Index -> List of Fine Triangle Indices
"""
function get_children_triangles(coarse_mesh::VoronoiMesh, fine_mesh::VoronoiMesh)
    mapping = map_fine_to_coarse(coarse_mesh, fine_mesh)
    children = Dict{Int, Vector{Int}}()
    
    for (fine_idx, coarse_idx) in enumerate(mapping)
        if coarse_idx != 0
            if !haskey(children, coarse_idx)
                children[coarse_idx] = []
            end
            push!(children[coarse_idx], fine_idx)
        end
    end
    
    return children
end



