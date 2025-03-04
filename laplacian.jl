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
    error_list_diagonal = []
    error_list_whitney = []
    error_list_lsq = []
    edge_length_list = []
    for i = 1:6
        #mesh = VoronoiMesh("meshes/mesh" * string(i) * "_regular.nc")
        mesh = VoronoiMesh("meshes/mesh" * string(i) * "_regular.nc")
        println("i = ", i)
        println("nCells: ", mesh.cells.n)
        println("nEdges: ", mesh.edges.n)
        println("nVertices: ", mesh.vertices.n)
        xp = mesh.x_period
        yp = mesh.y_period
        #mesh = VoronoiMesh("meshes/mesh" * string(i) * "_regular.nc")
        u = x -> sin(2*pi*x[1]/xp) * sin(2*pi*x[2]/yp)
        true_laplacian = x -> -8*(pi^2)*sin(2*pi*x[1]/xp) * sin(2*pi*x[2]/yp)

        u_cochain = Cochain(u.(mesh.cells.position), 0, true)
        lap_cochain = Cochain(true_laplacian.(mesh.cells.position), 0, true)

        lap_diagonal = laplacian_diagonal(mesh, u_cochain)
        lap_whitney = laplacian_whitney(mesh, u_cochain)
        lap_lsq = laplacian_lsq(mesh, u_cochain)

        push!(error_list_diagonal, maximum(abs.(lap_diagonal.values - lap_cochain.values)))
        #push!(error_list_diagonal, norm((lap_diagonal.values - lap_cochain.values) .* sqrt.(mesh.cells.area)))
        push!(error_list_whitney, maximum(abs.(lap_whitney.values - lap_cochain.values)))
        #push!(error_list_whitney, norm((lap_whitney.values - lap_cochain.values) .* sqrt.(mesh.cells.area)))
        push!(error_list_lsq, maximum(abs.(lap_lsq.values - lap_cochain.values)))
        #push!(error_list_lsq, norm((lap_lsq.values - lap_cochain.values) .* sqrt.(mesh.cells.area)))
        push!(edge_length_list, maximum(mesh.edges.lengthDual))
        println("--------------------------------")
    end

    println("Error list diagonal: ", error_list_diagonal)
    println("Error list whitney: ", error_list_whitney)
    println("Error list lsq: ", error_list_lsq)
    println("Edge length list: ", edge_length_list)
end

test_laplacian()