module VoronoiMeshes

using LinearAlgebra
using Zeros, TensorsLite, SmallCollections, TensorsLiteGeometry
using PrecompileTools

export @parallel

export SmallVectorArray, SmVecArray
export AbstractVoronoiDiagram, VoronoiDiagram, PlanarVoronoiDiagram, SphericalVoronoiDiagram
export CellInfo, Cells, VertexInfo, Vertices, EdgeInfo, Edges
export AbstractVoronoiMesh, VoronoiMesh, on_sphere, max_edges, integer_type, float_type, get_diagram
export plotmesh, plotmesh!, plotdiagram, plotdiagram!, plotdualmesh, plotdualmesh!
export create_cell_polygons, create_dual_triangles, create_edge_quadrilaterals
export graph_partition, find_obtuse_triangles, periodic_edges_mask, periodic_vertices_mask
export check_mesh, check_edge_normal_and_tangent, check_vertex_indexing, check_cell_indexing
export save
export circular_refinement_function, y_refinement_function, x_refinement_function
export fix_diagram!, create_planar_hex_mesh

# NCDatasetsKExt functions
export save_to_netcdf, save_to_netcdf!, read_from_netcdf, write_field_to_netcdf!, write_field_to_netcdf

# WriteVTKExt and ReadVTKExt functions
export save_voronoi_to_vtu, save_triangulation_to_vtu, read_from_vtu   

const VecMaybe1DxArray{TX, TYZ, N} = TensorsLite.TensorArray{Tensor{Union{TX, TYZ}, 1, TX, TYZ, TYZ}, N, Array{TX, N}, Array{TYZ, N}, Array{TYZ, N}}
const Vec1DxOr2DxyArray{TX, TXY, N} = TensorsLite.TensorArray{Tensor{Union{TX, Zero}, 1, TX, TXY, Zero}, N, Array{TX, N}, Array{TXY, N}, Array{Zero, N}}
const VecMaybe1DyArray{TY, TXZ, N} = TensorsLite.TensorArray{Tensor{Union{TY, TXZ}, 1, TXZ, TY, TXZ}, N, Array{TXZ, N}, Array{TY, N}, Array{TXZ, N}}

include("imvecarray.jl")

include("utils_pre.jl")

include("voronoi_diagrams.jl")

include("cell_struct.jl")

include("vertex_struct.jl")

include("edge_struct.jl")

function set_info_fields!(c::Cells, v::Vertices, e::Edges)
    ci = getfield(c, :info)
    if !isdefined(ci, :cellsOnEdge)
        ci.cellsOnEdge = e.cells
    end
    vi = getfield(v, :info)
    if !isdefined(vi, :verticesOnEdge)
        vi.verticesOnEdge = e.vertices
    end
    return nothing
end

abstract type AbstractVoronoiMesh{S, max_nEdges, TI, TF, TZ} end

"""
    VoronoiMesh{OnSphere?, max_nEdges, <:Integer, <:Float, <:Union{<:Float, Zeros.Zero}}

Struct that holds all information of a mesh which is constructed on top of
either a spherical or a planar biperiodic Voronoi diagram.
The `OnSphere?` type parameter is a `Bool` that is `true` if the mesh is on the sphere,
and `false` otherwise.
The `max_nEdges` type parameters is an integer that specifies the largest number of sides
present in the Voronoi cells of the mesh.

## Fields
- `cells::Cells`: a `Cell` struct that holds all Voronoi cells information.
  see available cell data with ?[`Cells`](@ref).
- `vertices::Vertices`: a `Vertices` struct that holds all vertices (and Delaunay triangle cells) information.
  see available vertex data with ?[`Vertices`](@ref).
- `edges::Edges`: a `Edges` struct that holds all edges (both primal (Voronoi) and dual (Delaunay) cells edges) information.
  see available edge data with ?[`Edges`](@ref).

## Other properties
- `diagram::VoronoiDiagram`: the Voronoi diagram from which the mesh is constructed.
- `sphere_radius::Float`(Spherical meshes only): the sphere radius in meters.
- `x_period::Float`(Planar meshes only): the `x` direction domain period.
- `y_period::Float`(Planar meshes only): the `y` direction domain period.

# Constructors

    VoronoiMesh(filename::String)

The `NCDatasets` or the `ReadVTK` package must be loaded to use this constructor.
Read the mesh from a NetCDF/VTU file.

---

    VoronoiMesh(N::Integer, x_period::Real, y_perid::Real; density=(x -> 1), rtol=1e-6, max_iter=20000)

The `DelaunayTriangulation` package must be loaded in order to use this method.
Construct a biperiodic planar Voronoi mesh with `N` cells in the `[0, x_period] × [0, y_period]` domain.
The `N` generators points are randomly generated and Lloyd's iteration are performed util the `max_iter`
number of iterations or the `rtol` relative tolerance is reached.
Optionally, a `density` function can be specified, to generate a mesh with variable resolution.
The `density` function should accept a `TensorsLite.Vec` and return a scalar.

---

    VoronoiMesh(points::TensorArray, x_period::Real, y_perid::Real; density=(x -> 1), rtol=1e-6, max_iter=20000)

The `DelaunayTriangulation` package must be loaded in order to use this method.
Construct a biperiodic planar Voronoi mesh with in the `[0, x_period] × [0, y_period]` domain using `points`
as starting voronoi generator points.
Lloyd's iteration are performed util the `max_iter` number of iterations
or the `rtol` relative tolerance is reached.
Optionally, a `density` function can be specified, to generate a mesh with variable resolution.
The `density` function should accept a `TensorsLite.Vec` and return a scalar.

---

    VoronoiMesh(diagram::VoronoiDiagram)

Construct a `VoronoiMesh` based on the Voronoi `diagram`.

"""
struct VoronoiMesh{S, maxEdges, TI, TF, Tz} <: AbstractVoronoiMesh{S, maxEdges, TI, TF, Tz}
    cells::Cells{S, maxEdges, TI, TF, Tz}
    vertices::Vertices{S, maxEdges, TI, TF, Tz}
    edges::Edges{S, maxEdges, TI, TF, Tz}

    function VoronoiMesh(
                c::Cells{S, NE, TI, TF, Tz},
                v::Vertices{S, NE, TI, TF, Tz},
                e::Edges{S, NE, TI, TF, Tz}
            ) where {S, NE, TI, TF, Tz}

        if !(get_diagram(c) === get_diagram(v) === get_diagram(e))
            throw(DimensionMismatch("`Cell`, `Vertices` and `Edges` structs are not based on the same Voronoi diagram"))
        end
        set_info_fields!(c, v, e)
        return new{S, NE, TI, TF, Tz}(c, v, e)
    end
end

function Base.show(io::IO, mesh::AbstractVoronoiMesh{false})
    s = """$(typeof(mesh))
    - Domain: Periodic in [0, $(mesh.x_period)] × [0, $(mesh.y_period)]
    - Number of Cells: $(mesh.cells.n)
    - Number of Vertices: $(mesh.vertices.n)
    - Number of Edges: $(mesh.edges.n)"""
    print(io, s)
end

function Base.show(io::IO, mesh::AbstractVoronoiMesh{true})
    s = """$(typeof(mesh))
    - Domain: Spherical with sphere radius $(mesh.sphere_radius)m
    - Number of Cells: $(mesh.cells.n)
    - Number of Vertices: $(mesh.vertices.n)
    - Number of Edges: $(mesh.edges.n)"""
    print(io, s)
end

get_diagram(mesh::AbstractVoronoiMesh) = get_diagram(mesh.cells)

Base.propertynames(m::AbstractVoronoiMesh{false}) = (fieldnames(typeof(m))..., :diagram, :x_period, :y_period)
Base.propertynames(m::AbstractVoronoiMesh{true}) = (fieldnames(typeof(m))..., :diagram, :sphere_radius)

Base.getproperty(mesh::AbstractVoronoiMesh, s::Symbol) = _getproperty(mesh, Val(s))
_getproperty(mesh::AbstractVoronoiMesh, ::Val{s}) where {s} = getfield(mesh, s)
_getproperty(mesh::AbstractVoronoiMesh, ::Val{:diagram}) = get_diagram(mesh)
_getproperty(mesh::AbstractVoronoiMesh{false}, ::Val{:x_period}) = get_diagram(mesh).x_period
_getproperty(mesh::AbstractVoronoiMesh{false}, ::Val{:y_period}) = get_diagram(mesh).y_period
_getproperty(mesh::AbstractVoronoiMesh{true}, ::Val{:sphere_radius}) = get_diagram(mesh).sphere_radius

for N in 6:9
    precompile(Tuple{typeof(_getproperty), VoronoiMeshes.VoronoiMesh{false, N, Int32, Float64, Zeros.Zero}, Val{:cells}})
    precompile(Tuple{typeof(_getproperty), VoronoiMeshes.VoronoiMesh{true, N, Int32, Float64, Float64}, Val{:cells}})
end

for T in (
        :AbstractVoronoiMesh,
        :Cells, :CellInfo,
        :Vertices, :VertexInfo,
        :Edges, :EdgeInfo,
        :VoronoiDiagram,
    )
    @eval begin
        on_sphere(::Type{<:$T{S}}) where {S} = S
        max_edges(::Type{<:$T{S, N}}) where {S, N} = N
        integer_type(::Type{<:$T{S, N, TI}}) where {S, N, TI} = TI
        float_type(::Type{<:$T{S, N, TI, TF}}) where {S, N, TI, TF} = TF
    end
    for func in (:on_sphere, :max_edges, :integer_type, :float_type)
        @eval $func(field::$T) = $func(typeof(field))
    end
end

function VoronoiMesh(voronoi::VoronoiDiagram)
    verticesOnCell = voronoi.verticesOnCell
    cellsOnVertex = voronoi.cellsOnVertex
    cpos = voronoi.generators
    vpos = voronoi.vertices

    edges, vertex_pair_to_edge, cell_pair_to_edge = build_edges(voronoi)

    cellsOnCell = compute_cellsOnCell(verticesOnCell, cellsOnVertex)
    edgesOnCell = compute_edgesOnCell(cellsOnCell, cell_pair_to_edge)

    cells = Cells(length(cpos), cpos, verticesOnCell.length, verticesOnCell, edgesOnCell, cellsOnCell, CellInfo(voronoi))

    edgesOnVertex = compute_edgesOnVertex(cellsOnVertex, cell_pair_to_edge)

    vertices = Vertices(length(vpos), vpos, edgesOnVertex, cellsOnVertex, VertexInfo(voronoi))

    return VoronoiMesh(cells, vertices, edges)
end

include("utils_pos.jl")

include("save_func.jl")

include("refinement_functions.jl")

include("read_func.jl")

#Definitions are in ext/NCDatasetsExt.jl
function save_to_netcdf end
function save_to_netcdf! end
function write_field_to_netcdf! end
function write_field_to_netcdf end
function read_from_netcdf end

#Definitions are in ext/DelaunayTriangulationExt.jl
function create_planar_hex_mesh end


#functions to be used when Makie is loaded
#Definitions are in ext/GeometryBasicsExt.jl
function create_cell_polygons end
function create_dual_triangles end
function create_edge_quadrilaterals end
function create_cell_linesegments end
function create_dual_triangles_linesegments end
function create_diagram_linesegments end

# Defined in ext/MakieExt.jl
function plotmesh end
function plotmesh! end
function plotdualmesh end
function plotdualmesh! end

function plotdiagram end
function plotdiagram! end

#Definitions are in ext/WriteVTKExt.jl
function save_voronoi_to_vtu end
function save_triangulation_to_vtu end

#Definitions are in ext/ReadVTKExt.jl
function read_from_vtu end

include("precompile.jl")

end # Module
