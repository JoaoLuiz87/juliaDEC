using MPASTools, NCDatasets, VoronoiMeshes, DelaunayTriangulation, GLMakie, TensorsLite, TensorsLiteGeometry
using Random
Random.seed!(1111)

function create_cell_linesegments_periodic(vert_pos,edge_pos,cell_pos,edgesOnCell,verticesOnEdge,x_period,y_period)
    x = eltype(edge_pos.x)[]
    y = eltype(edge_pos.y)[]
    nEdges = length(verticesOnEdge)
    sizehint!(x,2*nEdges)
    sizehint!(y,2*nEdges)
    x_periodic = eltype(edge_pos.x)[]
    y_periodic = eltype(edge_pos.y)[]
    sizehint!(x_periodic,nEdges÷2) 
    sizehint!(y_periodic,nEdges÷2)

    touched_edges_pos = Set{eltype(edge_pos)}()

    @inbounds for i in eachindex(edgesOnCell)
        c_pos = cell_pos[i]
        edges_ind = edgesOnCell[i]

        for j in edges_ind
            e_pos=edge_pos[j]
            closest_e_pos = closest(c_pos,e_pos,x_period,y_period)
            if !(closest_e_pos in touched_edges_pos)
                p = verticesOnEdge[j]

                for k in p
                    v_pos = vert_pos[k]
                    closest_v_pos = closest(closest_e_pos,v_pos,x_period,y_period)
                    if (closest_e_pos == e_pos)
                        push!(x, closest_v_pos.x)
                        push!(y, closest_v_pos.y)
                    else
                        push!(x_periodic, closest_v_pos.x)
                        push!(y_periodic, closest_v_pos.y)
                    end
                end

                push!(touched_edges_pos,closest_e_pos)
            end

        end

    end

    return ((x, y), (x_periodic, y_periodic))
end

cell_linesegments(mesh::VoronoiMesh) = create_cell_linesegments_periodic(mesh.vertices.position, mesh.edges.position, mesh.cells.position, mesh.cells.edges, mesh.edges.vertices, mesh.x_period, mesh.y_period)

println("Distorted mesh? (y/n)")
dist = readline() == "y"

if dist
    mesh1 = VoronoiMesh(16, 1.0, 1.0, density = x-> 1.0, max_iter=10000)
    VoronoiMeshes.save("mesh1.nc", mesh1)

    mesh2 = VoronoiMesh(vcat(mesh1.cells.position, mesh1.edges.position), 1.0, 1.0, max_iter=20000)
    VoronoiMeshes.save("mesh2.nc", mesh2)

    mesh3 = VoronoiMesh(vcat(mesh2.cells.position, mesh2.edges.position), 1.0, 1.0, max_iter=20000)
    VoronoiMeshes.save("mesh3.nc", mesh3)

    mesh4 = VoronoiMesh(vcat(mesh3.cells.position, mesh3.edges.position), 1.0, 1.0, max_iter=20000)
    VoronoiMeshes.save("mesh4.nc", mesh4)

    mesh5 = VoronoiMesh(vcat(mesh4.cells.position, mesh4.edges.position), 1.0, 1.0, max_iter=20000)
    VoronoiMeshes.save("mesh5.nc", mesh5)

    mesh6 = VoronoiMesh(vcat(mesh5.cells.position, mesh5.edges.position), 1.0, 1.0, max_iter=10000)
    VoronoiMeshes.save("mesh6.nc", mesh6)

    mesh7 = VoronoiMesh(vcat(mesh6.cells.position, mesh6.edges.position), 1.0, 1.0, max_iter=3000)
    VoronoiMeshes.save("mesh7.nc", mesh7)

else
    mesh1 = VoronoiMesh(12, 1.0, 1.0)
    VoronoiMeshes.save("mesh1_regular.nc", mesh1)

    mesh2 = VoronoiMesh(vcat(mesh1.cells.position, mesh1.edges.position), 1.0, 1.0)
    VoronoiMeshes.save("mesh2_regular.nc", mesh2)

    mesh3 = VoronoiMesh(vcat(mesh2.cells.position, mesh2.edges.position), 1.0, 1.0)
    VoronoiMeshes.save("mesh3_regular.nc", mesh3)

    mesh4 = VoronoiMesh(vcat(mesh3.cells.position, mesh3.edges.position), 1.0, 1.0)
    VoronoiMeshes.save("mesh4_regular.nc", mesh4)

    mesh5 = VoronoiMesh(vcat(mesh4.cells.position, mesh4.edges.position), 1.0, 1.0)
    VoronoiMeshes.save("mesh5_regular.nc", mesh5)

    mesh6 = VoronoiMesh(vcat(mesh5.cells.position, mesh5.edges.position), 1.0, 1.0)
    VoronoiMeshes.save("mesh6_regular.nc", mesh6)

    mesh7 = VoronoiMesh(vcat(mesh6.cells.position, mesh6.edges.position), 1.0, 1.0)
    VoronoiMeshes.save("mesh7_regular.nc", mesh7)

    mesh8 = VoronoiMesh(vcat(mesh7.cells.position, mesh7.edges.position), 1.0, 1.0)
    VoronoiMeshes.save("mesh8_regular.nc", mesh8)

    mesh9 = VoronoiMesh(vcat(mesh8.cells.position, mesh8.edges.position), 1.0, 1.0)
    VoronoiMeshes.save("mesh9_regular.nc", mesh9)

    mesh10 = VoronoiMesh(vcat(mesh9.cells.position, mesh9.edges.position), 1.0, 1.0)
    VoronoiMeshes.save("mesh10_regular.nc", mesh10)

    println("done")
end


println("done")

