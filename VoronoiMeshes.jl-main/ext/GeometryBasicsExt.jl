module GeometryBasicsExt

using VoronoiMeshes
using LinearAlgebra: normalize
using TensorsLite: Vec
using TensorsLiteGeometry, SmallCollections
using GeometryBasics: GeometryBasics, Polygon, Point2f

@static if pkgversion(GeometryBasics) < v"0.5"
    using GeometryBasics: LineString, Line, TupleView
    const PolType = Polygon{2, Float32, Point2f, LineString{2, Float32, Point2f, Base.ReinterpretArray{Line{2, Float32}, 1, Tuple{Point2f, Point2f}, TupleView{Tuple{Point2f, Point2f}, 2, 1, Vector{Point2f}}, false}}, Vector{LineString{2, Float32, Point2f, Base.ReinterpretArray{Line{2, Float32}, 1, Tuple{Point2f, Point2f}, TupleView{Tuple{Point2f, Point2f}, 2, 1, Vector{Point2f}}, false}}}}
else
    const PolType = Polygon{2, Float32}
end

@inline _capacity(::Type{<:SmallVector{N}}) where {N} = N
@inline _capacity(::Type{<:FixedVector{N}}) where {N} = N

function create_polygons_periodic(vert_pos, polygon_pos, verticesOnPolygon, x_period, y_period)

    polygons = Vector{PolType}(undef, length(polygon_pos))
    NE = _capacity(eltype(verticesOnPolygon))
    @parallel for i in eachindex(polygon_pos)
        @inbounds begin
            ppos = polygon_pos[i]
            local_vertices = SmallVector{NE, Point2f}()
            for i_v in verticesOnPolygon[i]
                vpos = closest(ppos, vert_pos[i_v], x_period, y_period)
                local_vertices = @inbounds push(local_vertices, Point2f(vpos.x, vpos.y))
            end
            polygons[i] = Polygon(Array(local_vertices))
        end
    end

    return polygons
end

function VoronoiMeshes.create_cell_polygons(mesh::AbstractVoronoiMesh{false})
    return create_polygons_periodic(mesh.vertices.position, mesh.cells.position, mesh.cells.vertices, mesh.x_period, mesh.y_period)
end

function VoronoiMeshes.create_dual_triangles(mesh::AbstractVoronoiMesh{false})
    return create_polygons_periodic(mesh.cells.position, mesh.vertices.position, mesh.vertices.cells, mesh.x_period, mesh.y_period)
end

function create_polygons_sphere(vert_lon::Vector{T}, vert_lat, base_lon, edge_lon, edge_lat, verticesOnElement, edgesOnElement, clip::Bool = false) where {T<:Number}
    x_period = T(360)

    polygons = Vector{PolType}(undef, length(base_lon))

    @parallel for i in eachindex(base_lon)
        @inbounds begin
            c_lon_aux = rad2deg(base_lon[i])
            c_lon = c_lon_aux > 180 ? c_lon_aux - x_period : c_lon_aux
            #local_vertices = SmallVector{NE, Point2f}()
            local_vertices = Point2f[]

            edgesOnEl = edgesOnElement[i]
            verticesOnEl = verticesOnElement[i]
            i_v1 = verticesOnEl[end]
            for j in eachindex(edgesOnElement[i])
                v1lon_aux_1 = rad2deg(vert_lon[i_v1])
                v1lon_aux = v1lon_aux_1 > 180 ? v1lon_aux_1 - x_period : v1lon_aux_1
                v1lat = rad2deg(vert_lat[i_v1])
                v1lon = closest(c_lon, v1lon_aux)

                i_e = edgesOnEl[j]
                elon_aux_1 = rad2deg(edge_lon[i_e])
                elon_aux = elon_aux_1 > 180 ? elon_aux_1 - x_period : elon_aux_1
                elat = rad2deg(edge_lat[i_e])
                elon = closest(c_lon, elon_aux)

                i_v2 = verticesOnEl[j]
                v2lon_aux_1 = rad2deg(vert_lon[i_v2])
                v2lon_aux = v2lon_aux_1 > 180 ? v2lon_aux_1 - x_period : v2lon_aux_1
                v2lat = rad2deg(vert_lat[i_v2])
                v2lon = closest(c_lon, v2lon_aux)



                minlon = 3

                difflon = abs(v1lon - elon)
                if difflon <= minlon
                    #Assuming vertex n in between edge n and n+1
                    push!(local_vertices, Point2f(v1lon, v1lat), Point2f(elon, elat))
                else
                    nparts = div(difflon, minlon) + 2
                    v1pos = lonlat_to_position(1, deg2rad(v1lon), deg2rad(v1lat))
                    epos = lonlat_to_position(1, deg2rad(elon), deg2rad(elat))
                    for l in 1:nparts
                        w = (nparts - l)/(nparts - 1)
                        p = normalize(w*v1pos + (1-w)*epos)
                        lonr, latr = position_to_lonlat(p)
                        lon_aux_1 = rad2deg(lonr)
                        lon_aux = lon_aux_1 > 180 ? lon_aux_1 - x_period : lon_aux_1
                        lon = closest(c_lon, lon_aux)
                        push!(local_vertices, Point2f(lon, rad2deg(latr)))
                    end
                end

                difflon = abs(v2lon - elon)
                if difflon > minlon
                    nparts = div(difflon, minlon) + 2
                    epos = lonlat_to_position(1, deg2rad(elon), deg2rad(elat))
                    v2pos = lonlat_to_position(1, deg2rad(v2lon), deg2rad(v2lat))
                    for l in 2:(nparts-1)
                        w = (nparts - l)/(nparts - 1)
                        p = normalize(w*epos + (1-w)*v2pos)
                        lonr, latr = position_to_lonlat(p)
                        lon_aux_1 = rad2deg(lonr)
                        lon_aux = lon_aux_1 > 180 ? lon_aux_1 - x_period : lon_aux_1
                        lon = closest(c_lon, lon_aux)
                        push!(local_vertices, Point2f(lon, rad2deg(latr)))
                    end
                end

                i_v1 = i_v2

            end
            polygons[i] = Polygon(local_vertices)
        end
    end

    return polygons
end

function VoronoiMeshes.create_cell_polygons(mesh::AbstractVoronoiMesh{true}, clip::Bool = false)
    return create_polygons_sphere(mesh.vertices.longitude, mesh.vertices.latitude,
                                  mesh.cells.longitude,
                                  mesh.edges.longitude, mesh.edges.latitude,
                                  mesh.cells.vertices, mesh.cells.edges, clip)
end

function VoronoiMeshes.create_dual_triangles(mesh::AbstractVoronoiMesh{true}, clip::Bool = false)
    return create_polygons_sphere(mesh.cells.longitude, mesh.cells.latitude,
                                  mesh.vertices.longitude,
                                  mesh.edges.longitude, mesh.edges.latitude,
                                  mesh.vertices.cells, mesh.vertices.edges, clip)
end

function create_edge_quadrilaterals_periodic(edge_pos, vert_pos, cell_pos, verticesOnEdge, cellsOnEdge, x_period, y_period)

    edge_quadrilaterals = Vector{PolType}(undef, length(edge_pos))

    @parallel for i in eachindex(edge_pos)
        @inbounds begin
            epos = edge_pos[i]
            iv1, iv2 = verticesOnEdge[i]
            ic1, ic2 = cellsOnEdge[i]

            cpos1 = closest(epos, cell_pos[ic1], x_period, y_period)

            vpos1 = closest(epos, vert_pos[iv1], x_period, y_period)

            cpos2 = closest(epos, cell_pos[ic2], x_period, y_period)

            vpos2 = closest(epos, vert_pos[iv2], x_period, y_period)

            edge_quadrilaterals[i] = Polygon([Point2f(cpos1.x, cpos1.y), Point2f(vpos1.x, vpos1.y), Point2f(cpos2.x, cpos2.y), Point2f(vpos2.x, vpos2.y)])
        end
    end

    return edge_quadrilaterals
end

function VoronoiMeshes.create_edge_quadrilaterals(mesh::AbstractVoronoiMesh{false})
    return create_edge_quadrilaterals_periodic(mesh.edges.position, mesh.vertices.position, mesh.cells.position, mesh.edges.vertices, mesh.edges.cells, mesh.x_period, mesh.y_period)
end



"""
        create_polygon_linesegments_periodic(vert_pos, edge_pos, polygon_pos, edgesOnPolygon, verticesOnEdge, x_period, y_period)

Build line-segment coordinate lists for polygons while taking periodic
domain into account. This helper is used by the exported wrappers
`create_cell_linesegments` and `create_dual_triangles_linesegments`.

Arguments
- `vert_pos` : array-like of vertex positions (objects with `.x, .y` fields)
- `edge_pos` : array-like of edge midpoints/positions (with `.x, .y`)
- `polygon_pos` : array-like of polygon centers (used as reference to pick
    the closest periodic images)
- `edgesOnPolygon` : topology mapping polygon -> list of incident edges
- `verticesOnEdge` : topology mapping edge -> tuple/list of its two vertices
- `x_period, y_period` : periodic domain lengths (used by `closest` to
    compute wrapped images)

Returns
- A tuple `((x, y), (x_periodic, y_periodic))` where:
    - `(x, y)` are vectors of x/y coordinates for interior (non-wrapping)
        line-segment endpoints (interleaved pairs represent segment endpoints)
    - `(x_periodic, y_periodic)` are vectors of coordinates for segments
        that cross periodic boundaries (wrapped pieces)

Notes
- The function avoids duplicate segments by tracking processed edge
    positions via a `Set`. It uses `closest(...)` to select the image of
    vertices/edges nearest the polygon center so that segments are drawn
    consistently across periodic boundaries.
"""
function create_polygon_linesegments_periodic(vert_pos, edge_pos, polygon_pos, edgesOnPolygon, verticesOnEdge, x_period, y_period)
    x = eltype(edge_pos.x)[]
    y = eltype(edge_pos.y)[]
    nEdges = length(verticesOnEdge)
    sizehint!(x, 2 * nEdges)
    sizehint!(y, 2 * nEdges)
    x_periodic = eltype(edge_pos.x)[]
    y_periodic = eltype(edge_pos.y)[]
    sizehint!(x_periodic, nEdges ÷ 2)
    sizehint!(y_periodic, nEdges ÷ 2)

    touched_edges_pos = Set{eltype(edge_pos)}()

    @inbounds for i in eachindex(edgesOnPolygon)
        c_pos = polygon_pos[i]
        edges_ind = edgesOnPolygon[i]

        for j in edges_ind
            e_pos = edge_pos[j]
            closest_e_pos = closest(c_pos, e_pos, x_period, y_period)
            if !(closest_e_pos in touched_edges_pos)
                p = verticesOnEdge[j]

                is_interior = closest_e_pos == e_pos
                for k in p
                    v_pos = vert_pos[k]
                    closest_v_pos = closest(closest_e_pos, v_pos, x_period, y_period)
                    if (is_interior)
                        push!(x, closest_v_pos.x)
                        push!(y, closest_v_pos.y)
                    else
                        push!(x_periodic, closest_v_pos.x)
                        push!(y_periodic, closest_v_pos.y)
                    end
                end

                push!(touched_edges_pos, closest_e_pos)
            end

        end

    end

    return ((x, y), (x_periodic, y_periodic))
end

function VoronoiMeshes.create_cell_linesegments(mesh::AbstractVoronoiMesh{false})
    return create_polygon_linesegments_periodic(mesh.vertices.position, mesh.edges.position, mesh.cells.position, mesh.cells.edges, mesh.edges.vertices, mesh.x_period, mesh.y_period)
end

function VoronoiMeshes.create_dual_triangles_linesegments(mesh::AbstractVoronoiMesh{false})
    return create_polygon_linesegments_periodic(mesh.cells.position, mesh.edges.position, mesh.vertices.position, mesh.vertices.edges, mesh.edges.cells, mesh.x_period, mesh.y_period)
end

function create_diagram_linesegments_periodic(vert_pos, cell_pos, verticesOnCell, cellsOnVertex, x_period, y_period)
    x = eltype(vert_pos.x)[]
    y = eltype(vert_pos.y)[]
    nEdges = length(verticesOnCell) + length(cellsOnVertex)
    sizehint!(x, 2 * nEdges)
    sizehint!(y, 2 * nEdges)

    touched_edges_pos = Set{eltype(vert_pos)}()

    @inbounds for i in eachindex(verticesOnCell)
        c_pos = cell_pos[i]
        vertices_ind = verticesOnCell[i]

        L = length(vertices_ind)
        for j in Base.OneTo(L)
            j
            jp1 = j + 1
            j2 = ifelse(jp1 > L, jp1 - L, jp1)
            v1 = vertices_ind[j]
            v2 = vertices_ind[j2]
            v1_pos = closest(c_pos, vert_pos[v1], x_period, y_period)
            v2_pos = closest(c_pos, vert_pos[v2], x_period, y_period)
            e_pos = (v1_pos + v2_pos) / 2
            if !(e_pos in touched_edges_pos)
                push!(x, v1_pos.x)
                push!(x, v2_pos.x)
                push!(y, v1_pos.y)
                push!(y, v2_pos.y)
                push!(touched_edges_pos, e_pos)
            end
        end
    end
    return (x, y)
end

function VoronoiMeshes.create_diagram_linesegments(diagram::VoronoiDiagram{false})
    return create_diagram_linesegments_periodic(diagram.vertices, diagram.generators, diagram.verticesOnCell, diagram.cellsOnVertex, diagram.x_period, diagram.y_period)
end

end

