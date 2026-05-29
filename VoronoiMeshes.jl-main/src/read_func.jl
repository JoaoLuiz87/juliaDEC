function VoronoiMesh(filename::String; kwds...)

    _, ext = Base.Filesystem.splitext(filename)

    return if ext == ".nc" # Save to NetCDF using NCDatasets
        read_from_netcdf(filename; kwds...)
    elseif ext == ".vtu" #Save to VTU using WriteVTK
        read_from_vtu(filename)
    else
        error("Unsupported file extension: $filename")
    end
end

