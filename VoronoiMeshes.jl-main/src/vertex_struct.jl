mutable struct VertexInfo{S, NE, TI, TF, Tz}
    const diagram::VoronoiDiagram{S, NE, TI, TF, Tz}
    centroid::TensorsLite.VecMaybe2DxyArray{TF, Tz, 1}
    area::Vector{TF}
    areaMimetic::Vector{TF}
    kiteAreas::Vector{FixedVector{3, TF}}
    kiteAreasMimetic::Vector{FixedVector{3, TF}}
    edgesSign::Vector{FixedVector{3, TF}}
    longitude::Vector{TF}
    latitude::Vector{TF}
    verticesOnEdge::Vector{FixedVector{2, TI}}

    function VertexInfo(diagram::VoronoiDiagram{S, NE, TI, TF, Tz}) where {S, NE, TI, TF, Tz}
        return new{S, NE, TI, TF, Tz}(diagram)
    end
end

const planar_vertexinfo_names = (:centroid, :area, :kiteAreas, :edgesSign, :x_period, :y_period)
const spherical_vertexinfo_names = (filter(x -> ((x != :diagram) & (x != :verticesOnEdge)), fieldnames(VertexInfo))..., :sphere_radius)

"""
    Vertices{OnSphere, max_nEdges, <:Integer, <:Float, <:Union{Float, Zeros.Zero}}

Struct that holds all vertices (and dual cells (triangles)) information in a struct of arrays (SoA) layout, that is,
each field is usually an array with the requested data for each vertex.

Data should be accessed using the `getproperty` function, preferably through the "dot" syntax.
For example, to fetch an array with each vertex position, use `vertices.position`.

When the struct is initialized only part of its data is actually existent. We refer to
those fields as the "Base Data". Other fields are only computed and stored if they are called at least once.
We refer to those as the "Computed Data".

## Base Data
- `n::Int`: The total number of vertices (and dual cells (triangles)).
- `position::TensorArray`: An array with each vertex position vector, that is, the position of the Delaunay
  triangle circumcenter. An array with a particular coordinate can also be extracted throught the dot
   syntax. For example, an array with `x` coordinates of the vertex is given by `vertices.position.x`.
- `edges::Vector`: A Vector of tuples with the index ID of the edges that meet at the vertex.
- `cells::Vector`: A Vector of tuples with the index ID of the cells sharing the vertex.
- `sphere_radius::Real` (Spherical meshes only): The sphere radius.
- `x_period::Real` (Planar meshes only): The domain `x` direction period.
- `y_period::Real` (Planar meshes only): The domain `y` direction period.

## Computed Data
- `area::Vector`: A vector with the area of each Delaunay triangle.
- `kiteAreas::Vector`: A vector of tuples containing the intersection areas between primal (Voronoi) and dual (triangular) cells.
- `centroid::TensorArray`: A vector with each Delaunay triangle centroid position vector.
- `edgesSign::Vector`: A vector of tuples associated with the edges that meet at the vertex.
  Has value of 1 if the edge normal is oriented in the counter-clockwise direction and -1 otherwise.
- `longitude::Vector` (Spherical meshes only): The longitude in radians of the vertex `position` vector.
- `latitude::Vector` (Spherical meshes only): The latitude in radians of the vertex `position` vector.
- `areaMimetic::Vector` (Spherical meshes only): A vector with an area value for each Delaunay triangle suitable for the mimetic TRiSK scheme.
- `kiteAreasMimetic::Vector` (Spherical meshes only): A vector of tuples with values related to the intersection areas between primal (Voronoi) and dual (triangular) cells, calculated in a manner consistent with `areaMimetic`.
"""
struct Vertices{S, NE, TI, TF, Tz}
    n::Int
    position::TensorsLite.VecMaybe2DxyArray{TF, Tz, 1}
    edges::Vector{FixedVector{3, TI}}
    cells::Vector{FixedVector{3, TI}}
    info::VertexInfo{S, NE, TI, TF, Tz}
end

function Base.show(io::IO, vertices::Vertices{S}) where {S}
    s = """$(typeof(vertices))
    $(vertices.n) $(S ? "Spherical" : "Planar") Vertices"""
    print(io, s)
end

get_diagram(v::Vertices) = getfield(v, :info).diagram

const vertex_names = (:n, :position, :edges, :cells)

Base.propertynames(::Vertices{false}) = (vertex_names..., planar_vertexinfo_names...)
Base.propertynames(::Vertices{true}) = (vertex_names..., spherical_vertexinfo_names...)

Base.getproperty(vertex::Vertices, s::Symbol) = _getproperty(vertex, Val(s))
_getproperty(vertex::Vertices, ::Val{s}) where {s} = getfield(vertex, s)
_getproperty(vertex::Vertices{false}, ::Val{:x_period}) = get_diagram(vertex).x_period
_getproperty(vertex::Vertices{false}, ::Val{:y_period}) = get_diagram(vertex).y_period
_getproperty(vertex::Vertices{true}, ::Val{:sphere_radius}) = get_diagram(vertex).sphere_radius

include("vertex_info_creation.jl")

for s in fieldnames(VertexInfo)
    if s === :diagram

        _getproperty(vertex::Vertices, ::Val{:diagram}) = getfield(vertex, :info).diagram

        for nEdges in 6:10
            for TI in (Int32, Int64)
                for TF in (Float32, Float64)
                    precompile(_getproperty, (Vertices{true, nEdges, TI, TF, TF}, Val{:diagram}))
                    precompile(_getproperty, (Vertices{false, nEdges, TI, TF, Zero}, Val{:diagram}))
                end
            end
        end

    elseif s !== :verticesOnEdge
        func = Symbol(string("compute_vertex_", s))
        @eval function _getproperty(vertex::Vertices, ::Val{$(QuoteNode(s))})
            info = getfield(vertex, :info)
            if !isdefined(info, $(QuoteNode(s)))
                setfield!(info, $(QuoteNode(s)), $func(vertex))
            end
            return getfield(info, $(QuoteNode(s)))
        end
        for nEdges in 6:10
            for TI in (Int32, Int64)
                for TF in (Float32, Float64)
                    @eval precompile($func, (Vertices{true, $nEdges, $TI, $TF, $TF},))
                    @eval precompile($func, (Vertices{false, $nEdges, $TI, $TF, Zero},))

                    @eval precompile(_getproperty, (Vertices{true, $nEdges, $TI, $TF, $TF}, Val{$(QuoteNode(s))}))
                    @eval precompile(_getproperty, (Vertices{false, $nEdges, $TI, $TF, Zero}, Val{$(QuoteNode(s))}))
                end
            end
        end
    end
end
