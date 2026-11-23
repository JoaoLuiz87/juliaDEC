using VoronoiMeshes,TensorsLite, TensorsLiteGeometry, ImmutableVectors, LinearAlgebra, NCDatasets, MPASTools, JLD2
using StaticArrays, SparseArrays
using CairoMakie 
include("juliaDEC.jl")
@load "lsq_matrices.jld2" matrices_regular matrices_irregular

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
    laplacian_lsq(mesh::VoronoiMesh, u::Cochain{T}, linear::Bool=false) where T

Compute the Laplacian using the least squares Hodge star operator:
Δu = *d*du
linear: if true, uses linear LSQ, if false uses quadratic LSQ
"""
function laplacian_lsq(mesh::VoronoiMesh, mesh_level::Int, u::Cochain{T}, regular::Bool, linear::Bool=false) where T
    if u.k != 0 || !u.is_primal
        error("Input must be a primal 0-form")
    end
    
    # First d: 0-form -> 1-form on triangle edges
    du = d(mesh, u)
    
    # First Hodge: 1-form on triangle edges -> 1-form on hexagon edges
    if regular
        if linear
            hodge_du = Cochain(matrices_regular[mesh_level]["lsq_linear"] * du.values, 1, false)
        else
            hodge_du = Cochain(matrices_regular[mesh_level]["lsq_quadratic"] * du.values, 1, false)
        end
    else
        if linear
            hodge_du = Cochain(matrices_irregular[mesh_level]["lsq_linear"] * du.values, 1, false)
        else
            hodge_du = Cochain(matrices_irregular[mesh_level]["lsq_quadratic"] * du.values, 1, false)
        end
    end
    
    # Second d: 1-form on hexagon edges -> 2-form on hexagon cells
    d_hodge_du = d(mesh, hodge_du)
    
    # Final Hodge: 2-form on hexagon cells -> 0-form on triangle vertices
    return Cochain(star(mesh, 2, false) * d_hodge_du.values, 0, true)
end

function test_laplacian()
    error_Linf_diagonal = []
    error_Linf_lsq1 = []
    error_Linf_lsq2 = []
    error_l2_diagonal = []
    error_l2_lsq1 = []
    error_l2_lsq2 = []
    order_diagonal_Linf = []
    order_lsq1_Linf = []
    order_lsq2_Linf = []
    order_diagonal_l2 = []
    order_lsq1_l2 = []
    order_lsq2_l2 = []
    edge_length_list = []

    # First test with regular meshes
    for i = 1:7
        mesh = VoronoiMesh("meshes/mesh" * string(i) * "_regular.nc")
        xp = mesh.x_period
        yp = mesh.y_period
        println("Mesh ", i,"     vertices: ", mesh.cells.n,"    edges: ", mesh.edges.n, "   cells: ", mesh.vertices.n)

        # Define test function and its true Laplacian
        u = x -> cos(4*pi*x[1]/xp) * cos(4*pi*x[2]/yp)
        true_laplacian = x -> -16*(pi^2)*cos(4*pi*x[1]/xp) * cos(4*pi*x[2]/yp)*(xp^(-2) + yp^(-2))

        u_cochain = Cochain(u.(mesh.cells.position), 0, true)
        lap_cochain = Cochain(true_laplacian.(mesh.cells.position), 0, true)

        # Compute numerical Laplacians
        lap_diagonal = laplacian_diagonal(mesh, u_cochain)
        lap_lsq1 = laplacian_lsq(mesh, i, u_cochain, true, true)  # linear LSQ
        lap_lsq2 = laplacian_lsq(mesh, i, u_cochain, true, false) # quadratic LSQ

        # Compute errors
        push!(error_Linf_diagonal, maximum(abs.(lap_diagonal.values - lap_cochain.values)))
        push!(error_Linf_lsq1, maximum(abs.(lap_lsq1.values - lap_cochain.values)))
        push!(error_Linf_lsq2, maximum(abs.(lap_lsq2.values - lap_cochain.values)))
        push!(error_l2_diagonal, norm((lap_diagonal.values - lap_cochain.values)) .* sqrt.(maximum(mesh.cells.area)))
        push!(error_l2_lsq1, norm((lap_lsq1.values - lap_cochain.values)) .* sqrt.(maximum(mesh.cells.area)))
        push!(error_l2_lsq2, norm((lap_lsq2.values - lap_cochain.values)) .* sqrt.(maximum(mesh.cells.area)))
        push!(edge_length_list, maximum(mesh.edges.lengthDual))
    end

    # Compute convergence orders
    for i = 2:6
        push!(order_diagonal_Linf, log(error_Linf_diagonal[i]/error_Linf_diagonal[i-1]) / log(edge_length_list[i]/edge_length_list[i-1]))
        push!(order_lsq1_Linf, log(error_Linf_lsq1[i]/error_Linf_lsq1[i-1]) / log(edge_length_list[i]/edge_length_list[i-1]))
        push!(order_lsq2_Linf, log(error_Linf_lsq2[i]/error_Linf_lsq2[i-1]) / log(edge_length_list[i]/edge_length_list[i-1]))
        push!(order_diagonal_l2, log(error_l2_diagonal[i]/error_l2_diagonal[i-1]) / log(edge_length_list[i]/edge_length_list[i-1]))
        push!(order_lsq1_l2, log(error_l2_lsq1[i]/error_l2_lsq1[i-1]) / log(edge_length_list[i]/edge_length_list[i-1]))
        push!(order_lsq2_l2, log(error_l2_lsq2[i]/error_l2_lsq2[i-1]) / log(edge_length_list[i]/edge_length_list[i-1]))
    end

    # Create plots
    fig = Figure()
    ax = Axis(fig[1,1], xlabel = L"h", ylabel = L"Error $||⋅||_{\infty}$", 
              title = L"Laplacian Error $||⋅||_{\infty}$ Regular Mesh", 
              yscale = log10, xscale = log2)
    scatterlines!(ax, edge_length_list, error_Linf_diagonal, label="Diagonal", color = :red, marker = :circle)
    scatterlines!(ax, edge_length_list, error_Linf_lsq1, label="LSQ1", color = :green, marker = :diamond)
    scatterlines!(ax, edge_length_list, error_Linf_lsq2, label="LSQ2", color = :orange, marker = :star5)
    scatterlines!(ax, [2^(-4), 2^(-5)], [2.0, 1.0], label="First order", color = :gray)
    scatterlines!(ax, [2^(-4), 2^(-5)], [1.0, 0.25], label="Second order", color = :black)
    axislegend(ax, position = :rb)
    CairoMakie.save("results/laplacian/laplacian_error_Linf_regular.pdf", fig)    

    fig = Figure()
    ax = Axis(fig[1,1], xlabel = L"h", ylabel = L"Error $||⋅||_2$", 
              title = L"Laplacian Error $||⋅||_2$ Regular Mesh", 
              yscale = log10, xscale = log2)
    scatterlines!(ax, edge_length_list, error_l2_diagonal, label="Diagonal", color = :red, marker = :circle)
    scatterlines!(ax, edge_length_list, error_l2_lsq1, label="LSQ1", color = :green, marker = :diamond)
    scatterlines!(ax, edge_length_list, error_l2_lsq2, label="LSQ2", color = :orange, marker = :star5)
    scatterlines!(ax, [2^(-4), 2^(-5)], [2.0, 1.0], label="First order", color = :gray)
    scatterlines!(ax, [2^(-4), 2^(-5)], [1.0, 0.25], label="Second order", color = :black)
    axislegend(ax, position = :rb)
    CairoMakie.save("results/laplacian/laplacian_error_L2_regular.pdf", fig)

    println("TESTE COM MALHA REGULAR")
    println("Order of Linf error for diagonal Laplacian: ", order_diagonal_Linf[end])
    println("Order of Linf error for LSQ1 Laplacian: ", order_lsq1_Linf[end])
    println("Order of Linf error for LSQ2 Laplacian: ", order_lsq2_Linf[end])
    println("Order of l2 error for diagonal Laplacian: ", order_diagonal_l2[end])
    println("Order of l2 error for LSQ1 Laplacian: ", order_lsq1_l2[end])
    println("Order of l2 error for LSQ2 Laplacian: ", order_lsq2_l2[end])

    # Reset arrays for irregular mesh tests
    error_Linf_diagonal = []
    error_Linf_lsq1 = []
    error_Linf_lsq2 = []
    error_l2_diagonal = []
    error_l2_lsq1 = []
    error_l2_lsq2 = []
    order_diagonal_Linf = []
    order_lsq1_Linf = []
    order_lsq2_Linf = []
    order_diagonal_l2 = []
    order_lsq1_l2 = []
    order_lsq2_l2 = []
    edge_length_list = []

    # Test with irregular meshes
    for i = 1:7
        mesh = VoronoiMesh("meshes/mesh" * string(i) * ".nc")
        xp = mesh.x_period
        yp = mesh.y_period
        println("Mesh ", i,"     vertices: ", mesh.cells.n,"    edges: ", mesh.edges.n, "   cells: ", mesh.vertices.n)

        u = x -> cos(4*pi*x[1]/xp) * cos(4*pi*x[2]/yp)
        true_laplacian = x -> -16*(pi^2)*cos(4*pi*x[1]/xp) * cos(4*pi*x[2]/yp)*(xp^(-2) + yp^(-2))

        u_cochain = Cochain(u.(mesh.cells.position), 0, true)
        lap_cochain = Cochain(true_laplacian.(mesh.cells.position), 0, true)

        lap_diagonal = laplacian_diagonal(mesh, u_cochain)
        lap_lsq1 = laplacian_lsq(mesh, i, u_cochain, false, true)  # linear LSQ
        lap_lsq2 = laplacian_lsq(mesh, i, u_cochain, false, false) # quadratic LSQ

        push!(error_Linf_diagonal, maximum(abs.(lap_diagonal.values - lap_cochain.values)))
        push!(error_Linf_lsq1, maximum(abs.(lap_lsq1.values - lap_cochain.values)))
        push!(error_Linf_lsq2, maximum(abs.(lap_lsq2.values - lap_cochain.values)))
        push!(error_l2_diagonal, norm((lap_diagonal.values - lap_cochain.values)) .* sqrt.(maximum(mesh.cells.area)))
        push!(error_l2_lsq1, norm((lap_lsq1.values - lap_cochain.values)) .* sqrt.(maximum(mesh.cells.area)))
        push!(error_l2_lsq2, norm((lap_lsq2.values - lap_cochain.values)) .* sqrt.(maximum(mesh.cells.area)))
        push!(edge_length_list, maximum(mesh.edges.lengthDual))
    end

    for i = 2:6
        push!(order_diagonal_Linf, log(error_Linf_diagonal[i]/error_Linf_diagonal[i-1]) / log(edge_length_list[i]/edge_length_list[i-1]))
        push!(order_lsq1_Linf, log(error_Linf_lsq1[i]/error_Linf_lsq1[i-1]) / log(edge_length_list[i]/edge_length_list[i-1]))
        push!(order_lsq2_Linf, log(error_Linf_lsq2[i]/error_Linf_lsq2[i-1]) / log(edge_length_list[i]/edge_length_list[i-1]))
        push!(order_diagonal_l2, log(error_l2_diagonal[i]/error_l2_diagonal[i-1]) / log(edge_length_list[i]/edge_length_list[i-1]))
        push!(order_lsq1_l2, log(error_l2_lsq1[i]/error_l2_lsq1[i-1]) / log(edge_length_list[i]/edge_length_list[i-1]))
        push!(order_lsq2_l2, log(error_l2_lsq2[i]/error_l2_lsq2[i-1]) / log(edge_length_list[i]/edge_length_list[i-1]))
    end

    fig = Figure()
    ax = Axis(fig[1,1], xlabel = L"h", ylabel = L"Error $||⋅||_{\infty}$", 
              title = L"Laplacian Error $||⋅||_{\infty}$ Irregular Mesh", 
              yscale = log10, xscale = log2)
    scatterlines!(ax, edge_length_list, error_Linf_diagonal, label="Diagonal", color = :red, marker = :circle)
    scatterlines!(ax, edge_length_list, error_Linf_lsq1, label="LSQ1", color = :green, marker = :diamond)
    scatterlines!(ax, edge_length_list, error_Linf_lsq2, label="LSQ2", color = :orange, marker = :star5)
    scatterlines!(ax, [2^(-4), 2^(-5)], [2.0, 1.0], label="First order", color = :gray)
    scatterlines!(ax, [2^(-4), 2^(-5)], [1.0, 0.25], label="Second order", color = :black)
    axislegend(ax, position = :rb)
    CairoMakie.save("results/laplacian/laplacian_error_Linf_irregular.pdf", fig)

    fig = Figure()
    ax = Axis(fig[1,1], xlabel = L"h", ylabel = L"Error $||⋅||_2$", 
              title = L"Laplacian Error $||⋅||_2$ Irregular Mesh", 
              yscale = log10, xscale = log2)
    scatterlines!(ax, edge_length_list, error_l2_diagonal, label="Diagonal", color = :red, marker = :circle)
    scatterlines!(ax, edge_length_list, error_l2_lsq1, label="LSQ1", color = :green, marker = :diamond)
    scatterlines!(ax, edge_length_list, error_l2_lsq2, label="LSQ2", color = :orange, marker = :star5)
    scatterlines!(ax, [2^(-4), 2^(-5)], [2.0, 1.0], label="First order", color = :gray)
    scatterlines!(ax, [2^(-4), 2^(-5)], [1.0, 0.25], label="Second order", color = :black)
    axislegend(ax, position = :rb)
    CairoMakie.save("results/laplacian/laplacian_error_L2_irregular.pdf", fig)

    println("TESTE COM MALHA IRREGULAR")
    println("Order of Linf error for diagonal Laplacian: ", order_diagonal_Linf[end])
    println("Order of Linf error for LSQ1 Laplacian: ", order_lsq1_Linf[end])
    println("Order of Linf error for LSQ2 Laplacian: ", order_lsq2_Linf[end])
    println("Order of l2 error for diagonal Laplacian: ", order_diagonal_l2[end])
    println("Order of l2 error for LSQ1 Laplacian: ", order_lsq1_l2[end])
    println("Order of l2 error for LSQ2 Laplacian: ", order_lsq2_l2[end])
end

test_laplacian()