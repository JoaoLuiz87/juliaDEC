"""
    save(filename::String, mesh::AbstractVoronoiMesh; format=:netcdf5_64bit_data, write_computed=false)

Save a Voronoi mesh or diagram to disk in either NetCDF or VTU (VTK) format, based on the file extension:

* If `filename` ends with `.nc`, saves the mesh in NetCDF format using NCDatasets.jl.
    - By default, the CDF5 NetCDF file format is used. Other formats can be specified with the `format` keyword. See other available options with ?[`NCDataset`](@ref).
    - Only the base fields are stored by default. To write computed fields, use `write_computed=true`.
    - For meshes, also writes a graph partition file (`.graph.info`).

* If `filename` ends with `.vtu`, saves the mesh in VTK Unstructured Grid (VTU) format using WriteVTK.jl.
    - Two files are written: one for the Voronoi grid and one for the Delaunay triangulation, with `_vor` and `_tri` in the filenames.
    - Handles periodic ghost vertices automatically for correct visualization in VTK/ParaView.

* If the extension is not recognized, an error is thrown.

---


    save(filename::String, diagram::VoronoiDiagram; format=:netcdf5_64bit_data)

Save the `diagram` in a NetCDF file named `filename` (see above for options).
"""
function save(filename::String, obj::T; kwds...) where {T<:Union{<:AbstractVoronoiDiagram,<:AbstractVoronoiMesh}}
    name, ext = Base.Filesystem.splitext(filename)
    if ext == ".nc" # Save to NetCDF using NCDatasets
        save_to_netcdf(filename, obj; kwds...)
        T<:AbstractVoronoiMesh && write(name*".graph.info", String(take!(graph_partition(obj))))
    elseif ext == ".vtu" #Save to VTU using VTKExt
        name_vor = name*"_vor"*ext
        name_tri = name*"_tri"*ext
        println("Saving to VTU $name : " , name_vor, ", ", name_tri)
        save_voronoi_to_vtu(name_vor, obj)
        save_triangulation_to_vtu(name_tri, obj)
    else
        error("Unsupported file extension: $filename")
    end
    return nothing
end

for N in 6:9
    precompile(Tuple{typeof(save), String, VoronoiMesh{false, N, Int32, Float64, Zeros.Zero}})
    precompile(Tuple{typeof(save), String, VoronoiMesh{true, N, Int32, Float64, Float64}})
end

