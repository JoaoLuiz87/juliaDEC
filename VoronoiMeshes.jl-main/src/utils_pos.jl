function graph_partition(cellsOnCell::AbstractVector{<:SmallVector},nEdges::Integer)
    nCells = length(cellsOnCell)
    outIO = IOBuffer()
    println(outIO, nCells,' ',nEdges)

    @inbounds for i in eachindex(cellsOnCell)
        map(x->print(outIO,x,' '),cellsOnCell[i])
        skip(outIO,-1)
        println(outIO)
    end
    seekstart(outIO)
    return outIO
end

graph_partition(cells::Cells,edges::Edges) = graph_partition(cells.cells,edges.n)

graph_partition(mesh::AbstractVoronoiMesh) = graph_partition(mesh.cells,mesh.edges)

function find_obtuse_triangles_periodic(cpos,cellsOnVertex,xp::Number,yp::Number)
    r = Int[]
    lk = ReentrantLock()
    @parallel for v in eachindex(cellsOnVertex)
        @inbounds begin
        c1,c2,c3 = cellsOnVertex[v]
        c1pos = cpos[c1]
        c2pos = closest(c1pos,cpos[c2],xp,yp)
        c3pos = closest(c1pos,cpos[c3],xp,yp)
        if is_obtuse(c1pos,c2pos,c3pos)
            lock(lk) do
                push!(r,v)
            end
        end
        end #inbounds
    end
    return r
end

find_obtuse_triangles(vertices::Vertices{false},cells::Cells{false},xp::Number,yp::Number) = find_obtuse_triangles_periodic(cells.position,vertices.cells,xp,yp)

find_obtuse_triangles(mesh::AbstractVoronoiMesh{false}) = find_obtuse_triangles(mesh.vertices,mesh.cells,mesh.x_period, mesh.y_period)

"""
    periodic_edges_mask(dcEdge::Vector,cellsOnEdge,cell_positions)
    periodic_edges_mask(mesh::AbstractVoronoiMesh)

Returns a BitArray that masks periodic edges (interior edges = true, boundary edges = false).
"""
function periodic_edges_mask(dc,cellsOnEdge,c_pos)
    mask = BitArray(undef,length(dc))
    @inbounds for e in eachindex(cellsOnEdge)
        dc_e = dc[e]
        c1,c2 = cellsOnEdge[e]
        dc_2 = norm(c_pos[c2] - c_pos[c1])
        mask[e] = dc_e ≈ dc_2
    end
    return mask
end

periodic_edges_mask(mesh::AbstractVoronoiMesh{false}) = periodic_edges_mask(mesh.edges.lengthDual, mesh.edges.cells, mesh.cells.position)

function periodic_edges_mask(mesh::AbstractVoronoiMesh{true})
    a = BitArray(mesh.edges.n)
    a .= true
    return a
end

"""
    periodic_vertices_mask(mesh::AbstractVoronoiMesh)

Returns a BitArray that masks vertices that belong to a periodic edge (interior vertices = true, boundary vertices = false).
"""
function periodic_vertices_mask(edgesOnVertex, mask_edges)
    mask = BitArray(undef,length(edgesOnVertex))
    @inbounds for v in eachindex(edgesOnVertex)
        e1, e2, e3 = edgesOnVertex[v]
        mask[v] = mask_edges[e1] && mask_edges[e2] && mask_edges[e3]
    end
    return mask
end

periodic_vertices_mask(mesh::AbstractVoronoiMesh{false}) = periodic_vertices_mask(mesh.vertices.edges, periodic_edges_mask(mesh))

function periodic_vertices_mask(mesh::AbstractVoronoiMesh{true})
    a = BitArray(mesh.vertices.n)
    a .= true
    return a
end

@inline function edge_normal_and_tanget_requirement(normal::AbstractVector, tangent::AbstractVector, surface_normal::AbstractVector)
    isapprox(((normal × tangent) ⋅ surface_normal), 1.0, atol = 0.001)
end

function check_edge_normal_and_tangent_spherical(R::Number, normals, tangents, edge_positions)
    r = Int[]
    lk = ReentrantLock()
    @parallel for e in eachindex(edge_positions)
        @inbounds begin
            if !edge_normal_and_tanget_requirement(normals[e], tangents[e], edge_positions[e] / R)
                lock(lk) do
                    push!(r,e)
                end
            end
        end #inbounds
    end
    n_problems = length(r)
    return n_problems == 0 ? nothing : r
end

function check_edge_normal_and_tangent_periodic(normals, tangents)
    r = Int[]
    lk = ReentrantLock()
    @parallel for e in eachindex(normals)
        @inbounds begin
            if !edge_normal_and_tanget_requirement(normals[e], tangents[e], 𝐤)
                lock(lk) do
                    push!(r,e)
                end
            end
        end #inbounds
    end
    n_problems = length(r)
    return n_problems == 0 ? nothing : r
end

check_edge_normal_and_tangent(edges::Edges{true}) = check_edge_normal_and_tangent_spherical(edges.sphere_radius, edges.normal, edges.tangent, edges.position)
check_edge_normal_and_tangent(edges::Edges{false}) = check_edge_normal_and_tangent_periodic(edges.normal, edges.tangent)

"""
    check_edge_normal_and_tangent(mesh::AbstractVoronoiMesh) -> result::Union{Nothing, Vector{<:Integer}}

Check if `(mesh.edges.normal[e] × mesh.edges.tanget[e]) ≈ surface_normal[e])` for all edges `e`.
If `true` for all edges, return `nothing`, otherwise return a vector of indices where the requirement doesn't hold.
"""
check_edge_normal_and_tangent(mesh::AbstractVoronoiMesh) = check_edge_normal_and_tangent(mesh.edges)

function check_edge_indexing_spherical(R::Number, edge_pos, cellsOnEdge, cell_pos, verticesOnEdge, vert_pos)
    r = Int[]
    lk = ReentrantLock()
    @parallel for e in eachindex(cellsOnEdge)
        @inbounds begin
            n = edge_pos[e] / R
            c1, c2 = cellsOnEdge[e]
            v1, v2 = verticesOnEdge[e]

            c1p = cell_pos[c1]
            c2p = cell_pos[c2]
            v1p = vert_pos[v1]
            v2p = vert_pos[v2]
            if signbit(((c2p - c1p) × (v2p - v1p)) ⋅ n)
                lock(lk) do
                    push!(r,e)
                end
            end
        end #inbounds
    end
    n_problems = length(r)
    return n_problems == 0 ? nothing : r
end

function check_edge_indexing_periodic(cellsOnEdge::AbstractVector, cell_pos, verticesOnEdge, vert_pos, xp::Number, yp::Number)
    r = Int[]
    lk = ReentrantLock()
    @parallel for e in eachindex(cellsOnEdge)
        @inbounds begin
            c1, c2 = cellsOnEdge[e]
            v1, v2 = verticesOnEdge[e]

            c1p = cell_pos[c1]
            c2p = closest(c1p, cell_pos[c2], xp, yp)
            v1p = closest(c1p, vert_pos[v1], xp, yp)
            v2p = closest(c1p, vert_pos[v2], xp, yp)
            if signbit(((c2p - c1p) × (v2p - v1p)).z)
                lock(lk) do
                    push!(r,e)
                end
            end
        end #inbounds
    end
    n_problems = length(r)
    return n_problems == 0 ? nothing : r
end

precompile(Tuple{typeof(check_edge_indexing_periodic), Array{Tuple{Int32, Int32}, 1}, Vec2DxyArray{Float64, 1}, Array{Tuple{Int32, Int32}, 1}, Vec2DxyArray{Float64, 1}, Float64, Float64})

"""
    check_edge_indexing(mesh::AbstractVoronoiMesh)

Check if the vector formed by
`(edges.cells[e][2] - edges.cells[e][1]) × (edges.vertices[e][2] - edges.vertices[e][1]))`
points at the same direction as the surface normal for every edge `e`.
"""
function check_edge_indexing end

function check_edge_indexing(mesh::AbstractVoronoiMesh{false})
    ordering = check_edge_indexing_periodic(mesh.edges.cells, mesh.cells.position,
                                            mesh.edges.vertices, mesh.vertices.position,
                                            mesh.x_period, mesh.y_period)
    isnothing(ordering) ? nothing : (;ordering)
end

function check_edge_indexing(mesh::AbstractVoronoiMesh{true})
    ordering = check_edge_indexing_spherical(mesh.sphere_radius, mesh.edges.position,
                                            mesh.edges.cells, mesh.cells.position,
                                            mesh.edges.vertices, mesh.vertices.position)
    isnothing(ordering) ? nothing : (;ordering)
end

function check_vertex_indexing_spherical(R::Number, vpos, cellsOnVertex, cpos, edgesOnVertex, epos)
    cells = check_indices_ordering(R, vpos, edgesOnVertex, epos, cellsOnVertex, cpos)
    isnothing(cells) ? nothing : (; cells)
end

function check_vertex_indexing_periodic(vpos, cellsOnVertex, cpos, edgesOnVertex, epos, xp::Number, yp::Number)
    cells = check_indices_ordering(vpos, edgesOnVertex, epos, cellsOnVertex, cpos, xp, yp)
    isnothing(cells) ? nothing : (; cells)
end

precompile(Tuple{typeof(check_vertex_indexing_periodic), Vec2DxyArray{Float64, 1}, Array{Tuple{Int32, Int32, Int32}, 1}, Vec2DxyArray{Float64, 1}, Array{Tuple{Int32, Int32, Int32}, 1}, Vec2DxyArray{Float64, 1}, Float64, Float64})

"""
    check_vertex_indexing(mesh::AbstractVoronoiMesh)

Check if `cells` and `edges` indexing arrays in `mesh.vertices` follow
counter-clockwise ordering.
Also check if `vertices.cells[v][n]` is between `vertices.edges[v][n]` and
`vertices.edges[v][n+1]` for every vertex `v`.
"""
function check_vertex_indexing end

function check_vertex_indexing(mesh::AbstractVoronoiMesh{false})
    ordering = check_vertex_indexing_periodic(mesh.vertices.position,
                                   mesh.vertices.cells,
                                   mesh.cells.position,
                                   mesh.vertices.edges,
                                   mesh.edges.position,
                                   mesh.x_period, mesh.y_period)
    xp = mesh.x_period
    yp = mesh.y_period
    cc_c = check_if_counter_clockwise(mesh.vertices.position,
                                      mesh.vertices.cells, mesh.cells.position, xp, yp)
    cc_e = check_if_counter_clockwise(mesh.vertices.position,
                                      mesh.vertices.edges, mesh.edges.position, xp, yp)

    return all(isnothing, (cc_c, cc_e, ordering)) ? 
        nothing :
        (ordering = ordering, counter_clockwise = (edges = cc_e, cells = cc_c))
end

function check_vertex_indexing(mesh::AbstractVoronoiMesh{true})
    R = mesh.sphere_radius
    ordering = check_vertex_indexing_spherical(R, mesh.vertices.position,
                                    mesh.vertices.cells,
                                    mesh.cells.position,
                                    mesh.vertices.edges,
                                    mesh.edges.position)
    cc_c = check_if_counter_clockwise(R, mesh.vertices.position,
                                      mesh.vertices.cells, mesh.cells.position)
    cc_e = check_if_counter_clockwise(R, mesh.vertices.position,
                                      mesh.vertices.edges, mesh.edges.position)

    return all(isnothing, (cc_c, cc_e, ordering)) ? 
        nothing :
        (ordering = ordering, counter_clockwise = (edges = cc_e, cells = cc_c))
end

function check_cell_indexing_spherical(R::Number, cpos, cellsOnCell, verticesOnCell, vpos, edgesOnCell, epos)
    edges = check_indices_ordering(R, cpos, edgesOnCell, epos, verticesOnCell, vpos)
    cells = check_indices_ordering(R, cpos, cellsOnCell, cpos, verticesOnCell, vpos)
    return all(isnothing, (edges, cells)) ? nothing : (;edges, cells)
end

function check_cell_indexing_periodic(cpos, cellsOnCell, verticesOnCell, vpos, edgesOnCell, epos, xp::Number, yp::Number)
    edges = check_indices_ordering(cpos, edgesOnCell, epos, verticesOnCell, vpos, xp, yp)
    cells = check_indices_ordering(cpos, cellsOnCell, cpos, verticesOnCell, vpos, xp, yp)
    return all(isnothing, (edges, cells)) ? nothing : (;edges, cells)
end

precompile(Tuple{typeof(check_cell_indexing_periodic), Vec2DxyArray{Float64, 1}, SmallVectorArray{6, Int32, 1, Array{FixedVector{6, Int32}, 1}}, SmallVectorArray{6, Int32, 1, Array{FixedVector{6, Int32}, 1}}, Vec2DxyArray{Float64, 1}, SmallVectorArray{6, Int32, 1, Array{FixedVector{6, Int32}, 1}}, Vec2DxyArray{Float64, 1}, Float64, Float64})

"""
    check_cell_indexing(mesh::AbstractVoronoiMesh)

Check if `cells`, `edges` and `vertices` indexing arrays in `mesh.cells` follow
counter-clockwise ordering.
Also check if `cell.vertices[c][n]` is between `cell.{edges|cells}[c][n]` and
`cell.{edges|cells}[c][n+1]` for every cell `c`.
"""
function check_cell_indexing end

function check_cell_indexing(mesh::AbstractVoronoiMesh{true})
    ordering  = check_cell_indexing_spherical(mesh.sphere_radius, mesh.cells.position,
        mesh.cells.cells,
        mesh.cells.vertices,
        mesh.vertices.position,
        mesh.cells.edges,
        mesh.edges.position)

    R = mesh.sphere_radius
    cc_vert = check_if_counter_clockwise(R, mesh.cells.position,
                                         mesh.cells.vertices, mesh.vertices.position)
    cc_edg = check_if_counter_clockwise(R, mesh.cells.position,
                                        mesh.cells.edges, mesh.edges.position)
    cc_cells = check_if_counter_clockwise(R, mesh.cells.position,
                                          mesh.cells.cells, mesh.cells.position)
    return all(isnothing, (cc_vert, cc_edg, cc_cells, ordering)) ? 
        nothing :
        (ordering = ordering, counter_clockwise = (vertices = cc_vert, edges = cc_edg, cells = cc_cells))
end

function check_cell_indexing(mesh::AbstractVoronoiMesh{false})
    ordering = check_cell_indexing_periodic(mesh.cells.position,
        mesh.cells.cells,
        mesh.cells.vertices,
        mesh.vertices.position,
        mesh.cells.edges,
        mesh.edges.position,
        mesh.x_period, mesh.y_period)

    xp = mesh.x_period
    yp = mesh.y_period
    cc_vert = check_if_counter_clockwise(mesh.cells.position,
                                         mesh.cells.vertices, mesh.vertices.position,
                                         xp, yp)
    cc_edg = check_if_counter_clockwise(mesh.cells.position,
                                         mesh.cells.edges, mesh.edges.position,
                                         xp, yp)
    cc_cells = check_if_counter_clockwise(mesh.cells.position,
                                         mesh.cells.cells, mesh.cells.position,
                                         xp, yp)
    return all(isnothing, (cc_vert, cc_edg, cc_cells, ordering)) ? 
        nothing :
        (ordering = ordering, counter_clockwise = (vertices = cc_vert, edges = cc_edg, cells = cc_cells))
end

function check_mesh(mesh::AbstractVoronoiMesh)
    edges = check_edge_indexing(mesh)
    vertices = check_vertex_indexing(mesh)
    cells = check_cell_indexing(mesh)
    return all(isnothing, (edges, vertices, cells)) ? nothing : (;edges, vertices, cells)
end

warn_mesh_issues(::Nothing, ::AbstractVoronoiMesh) = nothing

function warn_mesh_issues(nt::NamedTuple, mesh::AbstractVoronoiMesh)
    warn_edge_issues(nt.edges, mesh)
    warn_vertex_issues(nt.vertices, mesh)
    warn_cell_issues(nt.cells, mesh)
end

warn_mesh_issues(mesh::AbstractVoronoiMesh) = warn_mesh_issues(check_mesh(mesh), mesh)

warn_edge_issues(::Nothing, ::AbstractVoronoiMesh) = nothing

function warn_edge_issues(nt, mesh::AbstractVoronoiMesh)
    if !isnothing(nt)
        @warn "The edges indexing arrays from this mesh do not follow the mesh specification on $(length(nt.ordering)) edges. The indices of the offending edges can be given by running `check_edge_indexing(mesh).ordering`."
    end
end

warn_vertex_issues(::Nothing, ::AbstractVoronoiMesh) = nothing

function warn_vertex_issues(nt, mesh::AbstractVoronoiMesh)
    if !isnothing(nt)
       if !isnothing(nt.counter_clockwise)
            if !isnothing(nt.counter_clockwise.cells)
                @warn "The `vertices.cells` indexing array from this mesh is not in counter-clockwise order on $(length(nt.counter_clockwise.cells)) vertices. The indices of the offending vertices can be given by running `check_vertex_indexing(mesh).counter_clockwise.cells`."
            end
            if !isnothing(nt.counter_clockwise.edges)
                @warn "The `vertices.edges` indexing array from this mesh is not in counter-clockwise order on $(length(nt.counter_clockwise.edges)) vertices. The indices of the offending vertices can be given by running `check_vertex_indexing(mesh).counter_clockwise.edges`."
            end
        end
        if !isnothing(nt.ordering)
            @warn "The `vertices.edges` and vertices.cells` indexing arrays for this mesh do not follow the mesh specification on $(length(nt.ordering)) vertices. The indices of the offending vertices can be given by running `check_vertex_indexing(mesh).ordering`."
        end
    end
end

warn_cell_issues(::Nothing, ::AbstractVoronoiMesh) = nothing

function warn_cell_issues(nt, mesh::AbstractVoronoiMesh)
    if !isnothing(nt)
       if !isnothing(nt.counter_clockwise)
            if !isnothing(nt.counter_clockwise.cells)
                @warn "The `cells.cells` indexing array from this mesh is not in counter-clockwise order on $(length(nt.counter_clockwise.cells)) cells. The indices of the offending cells can be given by running `check_cell_indexing(mesh).counter_clockwise.cells`."
            end
            if !isnothing(nt.counter_clockwise.edges)
                @warn "The `cells.edges` indexing array from this mesh is not in counter-clockwise order on $(length(nt.counter_clockwise.edges)) cells. The indices of the offending cells can be given by running `check_cell_indexing(mesh).counter_clockwise.edges`."
            end
            if !isnothing(nt.counter_clockwise.vertices)
                @warn "The `cells.vertices` indexing array from this mesh is not in counter-clockwise order on $(length(nt.counter_clockwise.vertices)) cells. The indices of the offending cells can be given by running `check_cell_indexing(mesh).counter_clockwise.vertices`."
            end
        end
        if !isnothing(nt.ordering)
            @warn "The `cells.edges` and `cells.cells` indexing arrays for this mesh do not follow the mesh specification on $(length(nt.ordering)) cells. The indices of the offending cells can be given by running `check_cell_indexing(mesh).ordering`."
        end
    end
end

function fix_diagram!(d::PlanarVoronoiDiagram)
    verticesOnCell = d.verticesOnCell
    verticesOnCell_inverted = check_if_counter_clockwise(d.generators, verticesOnCell, d.vertices, d.x_period, d.y_period)
    if !isnothing(verticesOnCell_inverted)
        let voc = verticesOnCell_inverted::Vector{Int}
            @parallel for i in voc
                @inbounds verticesOnCell[i] = reverse(verticesOnCell[i])
            end
        end
    end

    cellsOnVertex = d.cellsOnVertex
    cellsOnVertex_inverted = check_if_counter_clockwise(d.vertices, cellsOnVertex, d.generators, d.x_period, d.y_period)
    if !isnothing(cellsOnVertex_inverted)
        let cov = cellsOnVertex_inverted::Vector{Int}
            @parallel for i in cov
                @inbounds cellsOnVertex[i] = reverse(cellsOnVertex[i])
            end
        end
    end

    return d
end

function fix_diagram!(d::SphericalVoronoiDiagram)
    verticesOnCell = d.verticesOnCell
    verticesOnCell_inverted = check_if_counter_clockwise(d.sphere_radius, d.generators, verticesOnCell, d.vertices)
    if !isnothing(verticesOnCell_inverted)
        let voc = verticesOnCell_inverted::Vector{Int}
            @parallel for i in voc
                @inbounds verticesOnCell[i] = reverse(verticesOnCell[i])
            end
        end
    end

    cellsOnVertex = d.cellsOnVertex
    cellsOnVertex_inverted = check_if_counter_clockwise(d.sphere_radius, d.vertices, cellsOnVertex, d.generators)
    if !isnothing(cellsOnVertex_inverted)
        let cov = cellsOnVertex_inverted::Vector{Int}
            @parallel for i in cov
                @inbounds cellsOnVertex[i] = reverse(cellsOnVertex[i])
            end
        end
    end

    return d
end

"""
    fix_diagram!(d::AbstractVoronoiDiagram) -> d

Fix any elements in `d.cellsOnVertex` and `d.verticesOnCell` that is not in counter-clockwise order
"""
fix_diagram!(d::VoronoiDiagram) = (fix_diagram!(get_diagram(d)); d )
