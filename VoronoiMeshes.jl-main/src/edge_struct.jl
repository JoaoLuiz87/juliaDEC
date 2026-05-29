mutable struct EdgeInfo{S, NE, TI, TF, Tz}
    const diagram::VoronoiDiagram{S, NE, TI, TF, Tz}
    midpoint::TensorsLite.VecMaybe2DxyArray{TF, Tz, 1}
    length::Vector{TF}
    lengthDual::Vector{TF}
    angle::Vector{TF}
    longitude::Vector{TF}
    latitude::Vector{TF}
    normal::TensorsLite.VecMaybe2DxyArray{TF, Tz, 1}
    tangent::TensorsLite.VecMaybe2DxyArray{TF, Tz, 1}

    function EdgeInfo(diagram::VoronoiDiagram{S, NE, TI, TF, Tz}) where {S, NE, TI, TF, Tz}
        return new{S, NE, TI, TF, Tz}(diagram)
    end
end

const planar_edgeinfo_names = (:midpoint, :length, :lengthDual, :angle, :normal, :tangent, :x_period, :y_period)
const spherical_edgeinfo_names = (filter(!=(:diagram), fieldnames(EdgeInfo))..., :sphere_radius)

"""
    Edges{OnSphere, max_nEdges, <:Integer, <:Float, <:Union{Float, Zeros.Zero}} 

Struct that holds all edges (both primal (of the Voronoi cell) and dual
(of the Delaunay triangle) edges) information in a struct of arrays (SoA) layout, that is,
each field is usually an array with the requested data for each edge.

Data should be accessed using the `getproperty` function, preferably through the "dot" syntax.
For example, to fetch an array with each edges length, use `edge.length`.

When the struct is initialized only part of its data is actually existent. We refer to
those fields as the "Base Data". Other fields are only computed and stored if they are called at least once.
We refer to those as the "Computed Data".

## Base Data
- `n::Int`: The total number of edges.
- `position::TensorArray`: An array with each edge position vector, that is, the position of the intersection
  between the Delaunay triangle and Voronoi cell edges.
  This always lies in the midpoint of the Delaunay triangle edge.
An array with a particular coordinate can also be extracted throught the dot
   syntax. For example, an array with `x` coordinates of the edge is given by `edges.position.x`.
- `vertices::Vector`: A Vector of tuples with the index ID of the vertices that form the edge.
- `cells::Vector`: A Vector of tuples with the index ID of the cells divided by the edge.
- `sphere_radius::Real` (Spherical meshes only): The sphere radius.
- `x_period::Real` (Planar meshes only): The domain `x` direction period.
- `y_period::Real` (Planar meshes only): The domain `y` direction period.

## Computed Data
- `length::Vector`: An array with the length of each edge (the Voronoi cell edge).
- `lengthDual::Vector`: An array with the length of the dual edge (the Delaunay triangle edge) that crosses the Voronoi cell edge.
  This is the distance between the Voronoi cells generators points that are divided by the edge.
- `midpoint::TensorArray`: An array with the position vector of the Voronoi cell edge midpoint.
- `normal::TensorArray`: An array with unit vectors that are normal to the edge and tangent to the mesh, at the edge `position`.
- `tangent::TensorArray`: An array with unit vectors that are tangent both to the the edge and to the mesh, at the edge `position`.
- `angle::Vector`: Angle in radians between local north and the positive tangential direction of an edge.
  Which is the same as the angle between the edge normal and the eastward direction.
- `longitude::Vector`(Spherical meshes only): The longitude in radians of the edge `position` vector.
- `latitude::Vector`(Spherical meshes only): The latitude in radians of the edge `position` vector.
"""
struct Edges{S, NE, TI, TF, Tz}
    n::Int
    position::TensorsLite.VecMaybe2DxyArray{TF, Tz, 1}
    vertices::Vector{FixedVector{2, TI}}
    cells::Vector{FixedVector{2, TI}}
    info::EdgeInfo{S, NE, TI, TF, Tz}
end

function Base.show(io::IO, edges::Edges{S}) where {S}
    s = """$(typeof(edges))
    $(edges.n) $(S ? "Spherical" : "Planar") Edges"""
    print(io, s)
end

get_diagram(e::Edges) = getfield(e, :info).diagram

const edge_names = (:n, :position, :vertices, :cells)

Base.propertynames(::Edges{false}) = (edge_names..., planar_edgeinfo_names...)
Base.propertynames(::Edges{true}) = (edge_names..., spherical_edgeinfo_names...)

Base.getproperty(edge::Edges, s::Symbol) = _getproperty(edge, Val(s))
_getproperty(edge::Edges, ::Val{s}) where {s} = getfield(edge, s)
_getproperty(edge::Edges{false}, ::Val{:x_period}) = get_diagram(edge).x_period
_getproperty(edge::Edges{false}, ::Val{:y_period}) = get_diagram(edge).y_period
_getproperty(edge::Edges{true}, ::Val{:sphere_radius}) = get_diagram(edge).sphere_radius

include("edge_info_creation.jl")

for s in fieldnames(EdgeInfo)
    if s !== :diagram
        func = Symbol(string("compute_edge_", s))
        @eval function _getproperty(edge::Edges, ::Val{$(QuoteNode(s))})
            info = getfield(edge, :info)
            if !isdefined(info, $(QuoteNode(s)))
                setfield!(info, $(QuoteNode(s)), $func(edge))
            end
            return getfield(info, $(QuoteNode(s)))
        end
        for nEdges in 6:10
            for TI in (Int32, Int64)
                for TF in (Float32, Float64)
                    @eval precompile($func, (Edges{true, $nEdges, $TI, $TF, $TF},))
                    @eval precompile($func, (Edges{false, $nEdges, $TI, $TF, Zero},))
                end
            end
        end
    else
        _getproperty(edges::Edges, ::Val{:diagram}) = getfield(edges, :info).diagram
    end
end

function build_edges(diagram::VoronoiDiagram{false, NE, TI, TF}) where {NE, TI, TF}

    cpos = diagram.generators
    verticesOnCell = diagram.verticesOnCell
    cellsOnVertex = diagram.cellsOnVertex
    nVertex = length(cellsOnVertex)
    nCells = length(verticesOnCell)
    nEdges = nVertex + nCells

    position = similar(diagram.vertices, nEdges)
    vertices = Vector{FixedVector{2, TI}}(undef, nEdges)
    cells = Vector{FixedVector{2, TI}}(undef, nEdges)

    xp = diagram.x_period
    yp = diagram.y_period

    vertex_pair_to_edge = Dict{FixedVector{2, TI}, TI}()
    cell_pair_to_edge = Dict{FixedVector{2, TI}, TI}()
    e = 0
    @inbounds for c in eachindex(cpos)
        cell_vertices = verticesOnCell[c]
        cp = cpos[c]
        v_1 = cell_vertices[1]
        v2 = v_1
        for v in 2:length(cell_vertices)
            v1 = v2
            v2 = cell_vertices[v]
            pair = ordered(v1, v2)
            if !(pair in keys(vertex_pair_to_edge))
                e += one(TI)
                vertex_pair_to_edge[pair] = e
                c2 = find_cellOnCell(c, v2, v1, cellsOnVertex)
                position[e] = periodic_to_base_point(((cp + closest(cp, cpos[c2], xp, yp)) / 2), xp, yp)
                vertices[e] = FixedVector((v1, v2))
                cells[e] = FixedVector((c, c2))
                cell_pair_to_edge[ordered(c,c2)] = e
            end
        end
        v1 = v2
        v2 = v_1
        pair = ordered(v1, v2)
        if !(pair in keys(vertex_pair_to_edge))
            e += one(TI)
            vertex_pair_to_edge[pair] = e
            c2 = find_cellOnCell(c, v2, v1, cellsOnVertex)
            position[e] = periodic_to_base_point(((cp + closest(cp, cpos[c2], xp, yp)) / 2), xp, yp)
            vertices[e] = FixedVector((v1, v2))
            cells[e] = FixedVector((c, c2))
            cell_pair_to_edge[ordered(c,c2)] = e
        end
    end

    @assert e == nEdges

    return Edges(e, position, vertices, cells, EdgeInfo(diagram)), vertex_pair_to_edge, cell_pair_to_edge
end

function build_edges(diagram::VoronoiDiagram{true, NE, TI, TF}) where {NE, TI, TF}

    cpos = diagram.generators
    verticesOnCell = diagram.verticesOnCell
    cellsOnVertex = diagram.cellsOnVertex
    nVertex = length(cellsOnVertex)
    nCells = length(verticesOnCell)
    nEdges = nVertex + nCells - 2

    position = similar(diagram.vertices, nEdges)
    vertices = Vector{FixedVector{2, TI}}(undef, nEdges)
    cells = Vector{FixedVector{2, TI}}(undef, nEdges)

    R = diagram.sphere_radius

    vertex_pair_to_edge = Dict{FixedVector{2, TI}, TI}()
    cell_pair_to_edge = Dict{FixedVector{2, TI}, TI}()
    e = 0
    @inbounds for c in eachindex(cpos)
        cell_vertices = verticesOnCell[c]
        cp = cpos[c]
        v_1 = cell_vertices[1]
        v2 = v_1
        for v in 2:length(cell_vertices)
            v1 = v2
            v2 = cell_vertices[v]
            pair = ordered(v1, v2)
            if !(pair in keys(vertex_pair_to_edge))
                e += one(TI)
                vertex_pair_to_edge[pair] = e
                c2 = find_cellOnCell(c, v2, v1, cellsOnVertex)
                position[e] = arc_midpoint(R, cp, cpos[c2])
                vertices[e] = FixedVector((v1, v2))
                cells[e] = FixedVector((c, c2))
                cell_pair_to_edge[ordered(c,c2)] = e
            end
        end
        v1 = v2
        v2 = v_1
        pair = ordered(v1, v2)
        if !(pair in keys(vertex_pair_to_edge))
            e += one(TI)
            vertex_pair_to_edge[pair] = e
            c2 = find_cellOnCell(c, v2, v1, cellsOnVertex)
            position[e] = arc_midpoint(R, cp, cpos[c2])
            vertices[e] = FixedVector((v1, v2))
            cells[e] = FixedVector((c, c2))
            cell_pair_to_edge[ordered(c,c2)] = e
        end
    end

    @assert e == nEdges

    return Edges(e, position, vertices, cells, EdgeInfo(diagram)), vertex_pair_to_edge, cell_pair_to_edge
end

function Edges(diagram::VoronoiDiagram)
    edges,_,_ = build_edges(diagram)
    return edges
end
