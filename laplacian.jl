using VoronoiMeshes,TensorsLite, TensorsLiteGeometry, ImmutableVectors, LinearAlgebra, NCDatasets, MPASTools
using StaticArrays, SparseArrays
using CairoMakie 
include("juliaDEC.jl")

"""
    laplacian_diagonal(mesh::VoronoiMesh, u::Cochain{T}) where T

Compute the Laplacian using the diagonal Hodge star operator:
Δu = *d*du
"""
function laplacian_diagonal(mesh::VoronoiMesh, u::Cochain{T}) where T
    if u.k != 0 || !u.is_primal
        error("Input must be a primal 0-form")
    end
    # First d: 0-form -> 1-form on triangle edges
    du = d(mesh, u)
    
    # First Hodge: 1-form on triangle edges -> 1-form on hexagon edges
    hodge_du = Cochain(star(mesh, 1, true) * du.values, 1, false)
    # Second d: 1-form on hexagon edges -> 2-form on hexagon cells
    d_hodge_du = d(mesh, hodge_du)
    
    # Final Hodge: 2-form on hexagon cells -> 0-form on triangle vertices
    #return Cochain(hodge_star_2_dual(mesh.cells.area) * d_hodge_du.values, 0, true)
    return Cochain(star(mesh, 2, false) * d_hodge_du.values, 0, true)
end

"""
    laplacian_whitney(mesh::VoronoiMesh, u::Cochain{T}) where T

Compute the Laplacian using the Whitney Hodge star operator:
Δu = *d*du
"""
function laplacian_whitney(mesh::VoronoiMesh, u::Cochain{T}) where T
    if u.k != 0 || !u.is_primal
        error("Input must be a primal 0-form")
    end
    
    # First d: 0-form -> 1-form on triangle edges
    du = d(mesh, u)
    
    # First Hodge: 1-form on triangle edges -> 1-form on hexagon edges
    hodge_du = Cochain(whitney_hodge_star_matrix(mesh) * du.values, 1, false)
    
    # Second d: 1-form on hexagon edges -> 2-form on hexagon cells
    d_hodge_du = d(mesh, hodge_du)
    
    # Final Hodge: 2-form on hexagon cells -> 0-form on triangle vertices
    #return Cochain(hodge_star_2_dual(mesh.cells.area) * d_hodge_du.values, 0, true)
    return Cochain(star(mesh, 2, false) * d_hodge_du.values, 0, true)
end

"""
    laplacian_lsq(mesh::VoronoiMesh, u::Cochain{T}) where T

Compute the Laplacian using the least squares Hodge star operator:
Δu = *d*du
"""
function laplacian_lsq(mesh::VoronoiMesh, u::Cochain{T}) where T
    if u.k != 0 || !u.is_primal
        error("Input must be a primal 0-form")
    end
    
    # First d: 0-form -> 1-form on triangle edges
    du = d(mesh, u)
    
    # First Hodge: 1-form on triangle edges -> 1-form on hexagon edges
    hodge_du = Cochain(hodge_star_lsq_matrix(mesh, true, false) * du.values, 1, false)
    
    # Second d: 1-form on hexagon edges -> 2-form on hexagon cells
    d_hodge_du = d(mesh, hodge_du)
    
    # Final Hodge: 2-form on hexagon cells -> 0-form on triangle vertices
    #return Cochain(hodge_star_2_dual(mesh.cells.area) * d_hodge_du.values, 0, true)
    return Cochain(star(mesh, 2, false) * d_hodge_du.values, 0, true)
end

function test_laplacian()
    errorL2_list_diagonal = []
    errorL2_list_whitney = []
    errorL2_list_lsq = []
    errorInf_list_diagonal = []
    errorInf_list_whitney = []
    errorInf_list_lsq = []
    edge_length_list = []
    for i = 1:7
        #mesh = VoronoiMesh("meshes/mesh" * string(i) * "_regular.nc")
        mesh = VoronoiMesh("meshes/mesh" * string(i) * ".nc")
        println("i = ", i)
        println("nCells: ", mesh.cells.n)
        println("nEdges: ", mesh.edges.n)
        println("nVertices: ", mesh.vertices.n)
        xp = mesh.x_period
        yp = mesh.y_period

        a = Vector{Bool}(undef, mesh.cells.n)
        for i = 1:mesh.cells.n
            if mesh.cells.position[i][1] > 0.25 && mesh.cells.position[i][1] < 0.75 && mesh.cells.position[i][2] > 0.25 && mesh.cells.position[i][2] < 0.75
                a[i] = true
            else
                a[i] = false
            end
        end

        #mesh = VoronoiMesh("meshes/mesh" * string(i) * "_regular.nc")
        u = x -> sin(4*pi*x[1]/xp) * sin(4*pi*x[2]/yp)
        true_laplacian = x -> -16*(pi^2)*sin(4*pi*x[1]/xp) * sin(4*pi*x[2]/yp)*(xp^(-2) + yp^(-2))

        u_cochain = Cochain(u.(mesh.cells.position), 0, true)
        lap_cochain = Cochain(true_laplacian.(mesh.cells.position), 0, true)

        lap_diagonal = laplacian_diagonal(mesh, u_cochain)
        lap_whitney = laplacian_whitney(mesh, u_cochain)
        lap_lsq = laplacian_lsq(mesh, u_cochain)

        push!(errorInf_list_diagonal, maximum(abs.(lap_diagonal.values - lap_cochain.values)[a]))
        push!(errorL2_list_diagonal, norm((lap_diagonal.values - lap_cochain.values)[a] .* sqrt.(mesh.cells.area)[a]))
        push!(errorInf_list_whitney, maximum(abs.(lap_whitney.values - lap_cochain.values)[a]))
        push!(errorL2_list_whitney, norm((lap_whitney.values - lap_cochain.values)[a] .* sqrt.(mesh.cells.area)[a]))
        push!(errorInf_list_lsq, maximum(abs.(lap_lsq.values - lap_cochain.values)[a]))
        push!(errorL2_list_lsq, norm((lap_lsq.values - lap_cochain.values)[a] .* sqrt.(mesh.cells.area)[a]))
        push!(edge_length_list, maximum(mesh.edges.lengthDual))
        println("i = ", i)
    end

    for i = 2:length(errorInf_list_diagonal)
        orderInf = log(errorInf_list_diagonal[i]/errorInf_list_diagonal[i-1]) / log(edge_length_list[i]/edge_length_list[i-1])
        println("Order Inf diagonal between meshes $(i-1) and $i: $orderInf")
    end

    for i = 2:length(errorL2_list_diagonal)
        orderL2 = log(errorL2_list_diagonal[i]/errorL2_list_diagonal[i-1]) / log(edge_length_list[i]/edge_length_list[i-1])
        println("Order L2 diagonal between meshes $(i-1) and $i: $orderL2")
    end

    println("--------------------------------")
    for i = 2:length(errorInf_list_whitney)
        orderInf = log(errorInf_list_whitney[i]/errorInf_list_whitney[i-1]) / log(edge_length_list[i]/edge_length_list[i-1])
        println("Order Inf whitney between meshes $(i-1) and $i: $orderInf")
    end

    for i = 2:length(errorL2_list_whitney)
        orderL2 = log(errorL2_list_whitney[i]/errorL2_list_whitney[i-1]) / log(edge_length_list[i]/edge_length_list[i-1])
        println("Order L2 whitney between meshes $(i-1) and $i: $orderL2")
    end 

    println("--------------------------------")
    for i = 2:length(errorInf_list_lsq)
        orderInf = log(errorInf_list_lsq[i]/errorInf_list_lsq[i-1]) / log(edge_length_list[i]/edge_length_list[i-1])
        println("Order Inf lsq between meshes $(i-1) and $i: $orderInf")
    end

    for i = 2:length(errorL2_list_lsq)
        orderL2 = log(errorL2_list_lsq[i]/errorL2_list_lsq[i-1]) / log(edge_length_list[i]/edge_length_list[i-1])
        println("Order L2 lsq between meshes $(i-1) and $i: $orderL2")
    end
end


test_laplacian()