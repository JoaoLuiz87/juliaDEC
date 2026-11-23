# gh_export_scvt.jl -- export the juliaDEC SCVT meshes (Delaunay vertices and triangles) to plain text for gh_lab.jl
#     julia scripts/archive/gh_export_scvt.jl
using VoronoiMeshes, NCDatasets
const OUT = joinpath(@__DIR__, "..", "..", "output", "gh_data"); mkpath(OUT)
for mn in ("mesh2", "mesh3", "mesh4", "mesh5")
    m = VoronoiMesh("meshes/$mn.nc")          # relative path: NetCDF fails on non-ASCII absolute paths
    open(joinpath(OUT, "$mn.txt"), "w") do io
        println(io, m.x_period, " ", m.y_period); println(io, m.cells.n)
        for p in m.cells.position
            println(io, Float64(p[1]), " ", Float64(p[2]))
        end
        println(io, m.vertices.n)
        for c in m.vertices.cells
            println(io, Int(c[1]), " ", Int(c[2]), " ", Int(c[3]))
        end
    end
    println("wrote $mn: ", m.cells.n, " vertices, ", m.vertices.n, " triangles")
end
