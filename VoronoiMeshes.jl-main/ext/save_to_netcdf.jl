using NCDatasets.CommonDataModel

import VoronoiMeshes: write_field_to_netcdf!, write_field_to_netcdf

function write_field_to_netcdf!(ds::NCDataset, data::AbstractVector{T}, name::String, dim_name::String, attrib::AbstractVector{Pair{String,String}}) where {T<:Number}

    if !haskey(ds.dim, dim_name)
        ds.dim[dim_name] = length(data)
    end

    fdata = T === Int16 ? Int32.(data) : data
    if !haskey(ds, name)
        defVar(ds, name, fdata, (dim_name,); attrib = attrib)
    else
        throw(ArgumentError("Field \"$name\" already present in the NetCDF file"))
    end

    return ds
end

function write_field_to_netcdf!(ds::NCDataset, vec_data::VecArray{T,1}, names::NTuple{N,String}, dim_name::String, attrib::NTuple{N, AbstractVector{Pair{String,String}}}) where {T<:Number, N}

    write_field_to_netcdf!(ds, vec_data.x, names[1], dim_name, attrib[1])
    write_field_to_netcdf!(ds, vec_data.y, names[2], dim_name, attrib[2])

    if N == 3
        if eltype(vec_data.z) == Zero
            write_field_to_netcdf!(ds, zeros(eltype(vec_data.x), length(vec_data)), names[3], dim_name, attrib[3])
        else
            write_field_to_netcdf!(ds, vec_data.z, names[3], dim_name, attrib[3])
        end
    end

    return ds
end

function write_field_to_netcdf!(ds::NCDataset, vec_data::VecArray{T,1}, name::String, dim_names::NTuple{2, String}, attrib::AbstractVector{Pair{String,String}}) where {T}

    if !haskey(ds.dim, dim_names[1])
        ds.dim[dim_names[1]] = 3
    end
    if !haskey(ds.dim, dim_names[2])
        ds.dim[dim_names[2]] = length(vec_data)
    end

    if haskey(ds, name)
        throw(ArgumentError("Field \"$name\" already present in the NetCDF file"))
    end

    TF = nonzero_eltype(eltype(vec_data))

    dataArray = zeros(TF, 3, length(vec_data))

    @inbounds for i in eachindex(vec_data)
        v = vec_data[i]
        dataArray[1, i] = v.x
        dataArray[2, i] = v.y
        dataArray[3, i] = v.z
    end

    defVar(ds, name, dataArray, dim_names; attrib = attrib)

    return ds
end

base_data(d) = d
base_data(d::SmallVectorArray) = d.data

function write_field_to_netcdf!(ds::NCDataset, data::AbstractVector{T}, name::String, dim_names::NTuple{2,String}, attrib::AbstractVector{Pair{String,String}}) where {T<:Union{<:Tuple,<:AbstractVector}}

    TI = eltype(T)
    fdata = reinterpret(reshape, TI, base_data(data))

    if !haskey(ds.dim, dim_names[1])
        ds.dim[dim_names[1]] = size(fdata,1)
    end

    if !haskey(ds.dim, dim_names[2])
        ds.dim[dim_names[2]] = size(fdata,2)
    end

    if !haskey(ds, name)
        defVar(ds, name, fdata, dim_names; attrib = attrib)
    else
        throw(ArgumentError("Field \"$name\" already present in the NetCDF file"))
    end

    return ds
end

function write_field_to_netcdf!(ds::NCDataset, data::AbstractVector{T}, name::String, dim_names::NTuple{3,String}, attrib::AbstractVector{Pair{String,String}}) where {T<:Union{<:NTuple, <:AbstractVector}}

    TV = eltype(T)
    fdata = reinterpret(reshape, TV, base_data(data))

    if !haskey(ds.dim, dim_names[1])
        ds.dim[dim_names[1]] = 3
    end

    if !haskey(ds.dim, dim_names[2])
        ds.dim[dim_names[2]] = size(fdata, 1)
    end

    if !haskey(ds.dim, dim_names[3])
        ds.dim[dim_names[3]] = size(fdata, 2)
    end

    TF = nonzero_eltype(TV)

    dataArray = zeros(TF, 3, size(fdata, 1), size(fdata, 2))

    @inbounds for k in axes(dataArray, 3)
        for j in axes(dataArray, 2)
            v = fdata[j, k]
            dataArray[1, j, k] = v.x
            dataArray[2, j, k] = v.y
            dataArray[3, j, k] = v.z
        end
    end

    if !haskey(ds, name)
        defVar(ds, name, dataArray, dim_names; attrib=attrib)
    else
        throw(ArgumentError("Field \"$name\" already present in the NetCDF file"))
    end

    return ds
end

function write_field_to_netcdf(filename::String, data::AbstractArray, names, dim_names, attrib; format = :netcdf5_64bit_data)

    mode = isfile(filename) ? "a" : "c"

    NCDataset(filename, mode, format = format) do ds
        write_field_to_netcdf!(ds, data, names, dim_names, attrib)
    end
end

function write_diagram_fields!(ds::NCDataset, cpos, vpos, meshDensity, force3D::Bool = false)

    is3D = eltype(cpos.z) !== Zeros.Zero

    xcattr = ["units" => "m", "long_name" => "Cartesian x-coordinate of cells"]
    ycattr = ["units" => "m", "long_name" => "Cartesian y-coordinate of cells"]
    zcattr = ["units" => "m", "long_name" => "Cartesian z-coordinate of cells"]

    xvattr = ["units" => "m", "long_name" => "Cartesian x-coordinate of vertices"]
    yvattr = ["units" => "m", "long_name" => "Cartesian y-coordinate of vertices"]
    zvattr = ["units" => "m", "long_name" => "Cartesian z-coordinate of vertices"]

    if (is3D | force3D)
        write_field_to_netcdf!(ds, cpos, ("xCell", "yCell", "zCell"), "nCells", (xcattr, ycattr, zcattr))
        write_field_to_netcdf!(ds, vpos, ("xVertex", "yVertex", "zVertex"), "nVertices", (xvattr, yvattr, zvattr))
    else
        write_field_to_netcdf!(ds, cpos, ("xCell", "yCell"), "nCells", (xcattr, ycattr))
        write_field_to_netcdf!(ds, vpos, ("xVertex", "yVertex"), "nVertices", (xvattr, yvattr))
    end

    write_field_to_netcdf!(ds, meshDensity, "meshDensity", "nCells", ["units" => "-", "long_name" => "Mesh density function (used when generating the mesh) evaluated at a cell"])

    return ds
end

function write_diagram_indexing_fields!(ds, diag::AbstractVoronoiDiagram{S, mE, TI, TF}) where {S, mE, TI, TF}

    write_field_to_netcdf!(ds, diag.cellsOnVertex, "cellsOnVertex", ("vertexDegree", "nVertices"), ["units" => "-", "long_name" => "IDs of the cells that meet at a vertex"])

    write_field_to_netcdf!(ds, diag.verticesOnCell, "verticesOnCell", ("maxEdges", "nCells"), ["units" => "-", "long_name" => "IDs of vertices (corner points) of a cell"])

    write_field_to_netcdf!(ds, diag.verticesOnCell.length, "nEdgesOnCell", "nCells", ["units" => "-", "long_name" => "Number of edges forming the boundary of a cell"])

    return ds
end

function write_diagram_data!(ds::NCDataset, diag::AbstractVoronoiDiagram, force3D::Bool = false)
    cpos = diag.generators
    vpos = diag.vertices
    mD = diag.meshDensity
    write_diagram_fields!(ds, cpos, vpos, mD, force3D)
    write_diagram_indexing_fields!(ds, diag)
end

function save_to_netcdf!(ds::NCDataset, diag::PlanarVoronoiDiagram{maxEdges, TI}, force3D::Bool = false) where {maxEdges, TI}
    attrib = ds.attrib
    attrib["on_a_sphere"] = "NO"
    attrib["is_periodic"] = "YES"
    attrib["x_period"] = diag.x_period
    attrib["y_period"] = diag.y_period
    write_diagram_data!(ds, diag, force3D)
end

function save_to_netcdf!(ds::NCDataset, diag::SphericalVoronoiDiagram{maxEdges, TI}, force3D::Bool = false) where {maxEdges, TI}
    attrib = ds.attrib
    attrib["on_a_sphere"] = "YES"
    attrib["sphere_radius"] = diag.sphere_radius
    write_diagram_data!(ds, diag, force3D)
end

save_to_netcdf!(ds::NCDataset, diag::VoronoiDiagram, force3D::Bool = false) = save_to_netcdf!(ds, get_diagram(diag), force3D)

function save_to_netcdf(filename::String, diag::AbstractVoronoiDiagram; format = :netcdf5_64bit_data)
    NCDataset(filename, "c", format = format) do ds
        save_to_netcdf!(ds, diag)
    end
end

function write_base_cell_data!(ds::NCDataset, cells::Cells{S, mE, TI}) where {S, mE, TI}

    write_field_to_netcdf!(ds, cells.edges, "edgesOnCell", ("maxEdges", "nCells"), ["units" => "-", "long_name" => "IDs of edges forming the boundary of a cell"])

    write_field_to_netcdf!(ds, cells.cells, "cellsOnCell", ("maxEdges", "nCells"), ["units" => "-", "long_name" => "IDs of cells neighboring a cell"])

     return ds
end

function write_computed_cell_data!(ds::NCDataset, cells::Cells{S, mE, TI, TF}, force3D::Bool = false) where {S, mE, TI, TF}
    cinfo = cells.info
    ncells = cells.n

    if isdefined(cinfo, :centroid)
        xattrib = ["units" => "m", "long_name" => "Cartesian x-coordinate of cells centroid"]
        yattrib = ["units" => "m", "long_name" => "Cartesian y-coordinate of cells centroid"]
        zattrib = ["units" => "m", "long_name" => "Cartesian z-coordinate of cells centroid"]

        if (S | force3D)
            write_field_to_netcdf!(ds, cinfo.centroid, ("xCellCentroid", "yCellCentroid", "zCellCentroid"), "nCells", (xattrib, yattrib, zattrib))
        else
            write_field_to_netcdf!(ds, cinfo.centroid, ("xCellCentroid", "yCellCentroid"), "nCells", (xattrib, yattrib))
        end
    end

    if isdefined(cinfo, :area)
        write_field_to_netcdf!(ds, cinfo.area, "areaCell", "nCells", ["units" => "m^2", "long_name" => ( S ? "Spherical area of a Voronoi cell" : "Area of a Voronoi cell")])
    end

    if isdefined(cinfo, :longitude)
        write_field_to_netcdf!(ds, cinfo.longitude, "lonCell", "nCells", ["units" => "rad", "long_name" => "Longitude of cells"])
    end

    if isdefined(cinfo, :latitude)
        write_field_to_netcdf!(ds, cinfo.latitude, "latCell", "nCells", ["units" => "rad", "long_name" => "Latitude of cells"])
    end

    if isdefined(cinfo, :normal)
        write_field_to_netcdf!(ds, cinfo.normal, "localVerticalUnitVectors", ("R3", "nCells"), ["units" => "unitless", "long_name" => "Cartesian components of the vector pointing in the local vertical direction for a cell"])
    end

    if (isdefined(cinfo, :zonalVector) & isdefined(cinfo, :meridionalVector))
        cellTangent = zeros(TF, 3, 2, ncells)
        zVector = cinfo.zonalVector
        mVector = cinfo.meridionalVector
        @inbounds for c in 1:ncells
            zv = zVector[c]
            cellTangent[1, 1, c] = zv.x
            cellTangent[2, 1, c] = zv.y
            cellTangent[3, 1, c] = zv.z

            mv = mVector[c]
            cellTangent[1, 2, c] = mv.x
            cellTangent[2, 2, c] = mv.y
            cellTangent[3, 2, c] = mv.z
        end
        defVar(ds, "cellTangentPlane", cellTangent, ("R3", "TWO", "nCells"); attrib = [
            "units" => "unitless",
            "long_name" => "Components of a pair of vectors defining the tangent plane at a cell"
        ])
    end
end

function write_cell_data!(ds::NCDataset, cells::Cells; force3D::Bool=false, write_computed::Bool = false)
    write_base_cell_data!(ds, cells)
    write_computed && write_computed_cell_data!(ds, cells, force3D)
    return ds
end

function write_base_vertex_data!(ds::NCDataset, vertices::Vertices{S, mE, TI}) where {S, mE, TI}
    write_field_to_netcdf!(ds,  vertices.edges, "edgesOnVertex", ("vertexDegree", "nVertices"), ["units" => "-", "long_name" => "IDs of the edges that meet at a vertex"])
     return ds
end

function write_computed_vertex_data!(ds::NCDataset, vertices::Vertices{S, mE, TI, TF}, force3D::Bool = false) where {S, mE, TI, TF}
    vinfo = vertices.info

    if isdefined(vinfo, :centroid)
        xattrib = ["units" => "m", "long_name" => "Cartesian x-coordinate of triangles centroid"]
        yattrib = ["units" => "m", "long_name" => "Cartesian y-coordinate of triangles centroid"]
        zattrib = ["units" => "m", "long_name" => "Cartesian z-coordinate of triangles centroid"]

        if (S | force3D)
            write_field_to_netcdf!(ds, vinfo.centroid, ("xVertexCentroid", "yVertexCentroid", "zVertexCentroid"), "nVertices", (xattrib, yattrib, zattrib))
        else
            write_field_to_netcdf!(ds, vinfo.centroid, ("xVertexCentroid", "yVertexCentroid"), "nVertices", (xattrib, yattrib))
        end
    end

    if isdefined(vinfo, :area)
        write_field_to_netcdf!(ds, vinfo.area, "areaTriangle", "nVertices", ["units" => "m^2", "long_name" => ( S ? "Spherical area of Delaunay Triangle" : "Area of a Delaunay Triangle")])
    end

    if isdefined(vinfo, :kiteAreas)
        write_field_to_netcdf!(ds, vinfo.kiteAreas, "kiteAreasOnVertex", ("vertexDegree", "nVertices"), ["units" => "m^2", "long_name" => "Intersection areas between primal (Voronoi) and dual (triangular) mesh cells"])
    end

    if isdefined(vinfo, :longitude)
        write_field_to_netcdf!(ds, vinfo.longitude, "lonVertex", "nVertices", ["units" => "rad", "long_name" => "Longitude of vertices"])
    end

    if isdefined(vinfo, :latitude)
        write_field_to_netcdf!(ds, vinfo.latitude, "latVertex", "nVertices", ["units" => "rad", "long_name" => "Latitude of vertices"])
    end

end

function write_vertex_data!(ds::NCDataset, vertices::Vertices; force3D::Bool=false, write_computed::Bool = false)
    write_base_vertex_data!(ds, vertices)
    write_computed && write_computed_vertex_data!(ds, vertices, force3D)
    return ds
end

function write_base_edge_data!(ds::NCDataset, edges::Edges{S, mE, TI, TF}, force3D::Bool = false) where {S, mE, TI, TF}

    write_field_to_netcdf!(ds, edges.vertices, "verticesOnEdge", ("TWO", "nEdges"), ["units" => "-", "long_name" => "IDs of the two vertex endpoints of an edge"])

    write_field_to_netcdf!(ds, edges.cells, "cellsOnEdge", ("TWO", "nEdges"), ["units" => "-", "long_name" => "IDs of cells divided by an edge"])

    xattrib = ["units" => "m", "long_name" => "Cartesian x-coordinate of edges"]
    yattrib = ["units" => "m", "long_name" => "Cartesian y-coordinate of edges"]
    zattrib = ["units" => "m", "long_name" => "Cartesian z-coordinate of edges"]

    if (S | force3D)
        write_field_to_netcdf!(ds, edges.position, ("xEdge", "yEdge", "zEdge"), "nEdges", (xattrib, yattrib, zattrib))
    else
        write_field_to_netcdf!(ds, edges.position, ("xEdge", "yEdge"), "nEdges", (xattrib, yattrib))
    end

    return ds
end

function write_computed_edge_data!(ds::NCDataset, edges::Edges{S, mE, TI, TF}, force3D::Bool = false) where {S, mE, TI, TF}

    einfo = edges.info

    if isdefined(einfo, :midpoint)
        xattrib = ["units" => "m", "long_name" => "Cartesian x-coordinate of edges midpoint"]
        yattrib = ["units" => "m", "long_name" => "Cartesian y-coordinate of edges midpoint"]
        zattrib = ["units" => "m", "long_name" => "Cartesian z-coordinate of edges midpoint"]

        if (S | force3D)
            write_field_to_netcdf!(ds, edges.midpoint, ("xEdgeMidpoint", "yEdgeMidpoint", "zEdgeMidpoint"), "nEdges", (xattrib, yattrib, zattrib))
        else
            write_field_to_netcdf!(ds, edges.midpoint, ("xEdgeMidpoint", "yEdgeMidpoint"), "nEdges", (xattrib, yattrib))
        end
    end

    if isdefined(einfo, :length)
        write_field_to_netcdf!(ds, einfo.length, "dvEdge", "nEdges", ["units" => "m", "long_name" => (S ? "Spherical distance between vertex endpoints of an edge" : "Distance between vertex endpoints of an edge")])
    end

    if isdefined(einfo, :lengthDual)
        write_field_to_netcdf!(ds, einfo.lengthDual, "dcEdge", "nEdges", ["units" => "m", "long_name" => (S ? "Spherical distance between cells separated by an edge" : "Distance between cells separated by an edge")])
    end

    if isdefined(einfo, :angle)
        write_field_to_netcdf!(ds, einfo.angle, "angleEdge", "nEdges", ["units" => "rad", "long_name" => "Angle between local north and the positive tangential direction of an edge"])
    end

    if isdefined(einfo, :longitude)
        write_field_to_netcdf!(ds, einfo.longitude, "lonEdge", "nEdges", ["units" => "rad", "long_name" => "Longitude of edges"])
    end

    if isdefined(einfo, :latitude)
        write_field_to_netcdf!(ds, einfo.latitude, "latEdge", "nEdges", ["units" => "rad", "long_name" => "Latitude of edges"])
    end

    if isdefined(einfo, :normal)
        write_field_to_netcdf!(ds, einfo.normal, "edgeNormalVectors", ("R3", "nEdges"), ["units" => "unitless", "long_name" => "Cartesian components of the vector normal to an edge and tangential to the surface of the sphere"])
    end

    if isdefined(einfo, :tangent)
        write_field_to_netcdf!(ds, einfo.tangent, "edgeTangentialVectors", ("R3", "nEdges"), ["units" => "unitless", "long_name" => "Cartesian components of the vector tangential to an edge and tangential to the surface of the sphere"])
    end

    return ds
end

function write_edge_data!(ds::NCDataset, edges::Edges; force3D::Bool = false, write_computed::Bool = false)
    write_base_edge_data!(ds, edges, force3D)
    write_computed && write_computed_edge_data!(ds, edges, force3D)
    return ds
end

function save_to_netcdf!(ds::NCDataset, mesh::AbstractVoronoiMesh; force3D::Bool = false, write_computed::Bool= false)
    save_to_netcdf!(ds, mesh.diagram, force3D)
    write_cell_data!(ds, mesh.cells; force3D = force3D, write_computed = write_computed)
    write_vertex_data!(ds, mesh.vertices; force3D = force3D, write_computed = write_computed)
    write_edge_data!(ds, mesh.edges; force3D = force3D, write_computed = write_computed)
end

function save_to_netcdf(filename::String, mesh::AbstractVoronoiMesh; force3D::Bool=false, write_computed::Bool=false, format = :netcdf5_64bit_data)
    NCDataset(filename, "c", format = format) do ds
        save_to_netcdf!(ds, mesh; force3D = force3D, write_computed = write_computed)
    end
end

