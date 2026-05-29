using VoronoiMeshes, TensorsLite, TensorsLiteGeometry, ImmutableVectors, LinearAlgebra, NCDatasets, MPASTools
using StaticArrays, SparseArrays
using CairoMakie, Printf, DataFrames, CSV, JLD2
include("juliaDEC.jl")

# Load pre-computed matrices
@load "lsq_matrices.jld2" matrices_regular matrices_irregular

function compute_advection_operator(mesh::VoronoiMesh, velocity::Function, hodge_type::String, mesh_level::Int, regular::Bool=true)
    # Compute components
    d1 = d1_matrix_dual(mesh)
    v = discretize_vector_field_dual(mesh, velocity)
    wedge10 = wedge_product_10_matrix(mesh, v)
    star2 = star(mesh, 2, false)
    
    # Choose Hodge star operator based on type
    if regular
        if hodge_type == "diagonal"
            star1 = star(mesh, 1, true)
        elseif hodge_type == "whitney"
            star1 = whitney_hodge_star_matrix(mesh)
        elseif hodge_type == "lsq_linear"
            star1 = matrices_regular[mesh_level]["lsq_linear"]
        elseif hodge_type == "lsq_quadratic"
            star1 = matrices_regular[mesh_level]["lsq_quadratic"]
        end
    else
        if hodge_type == "diagonal"
            star1 = star(mesh, 1, true)
        elseif hodge_type == "whitney"
            star1 = whitney_hodge_star_matrix(mesh)
        elseif hodge_type == "lsq_linear"
            star1 = matrices_irregular[mesh_level]["lsq_linear"]
            star1 = 0.5*(star1 + star1')
        elseif hodge_type == "lsq_quadratic"
            star1 = matrices_irregular[mesh_level]["lsq_quadratic"]
            star1 = 0.5*(star1 + star1')
        end
    end
    
    # Compute full operator
    return -d1 * star1 * wedge10 * star2
end

function plot_eigenvalues(mesh::VoronoiMesh, velocity::Function, mesh_level::Int, regular::Bool=true)
    # Create figure
    fig = Figure(size=(1200, 800))
    
    # Create a layout with 2x2 subplots
    axs = [Axis(fig[i,j], 
                aspect=1,
                xlabel="Re(λ)",
                ylabel="Im(λ)",
                title=title) 
           for (i,j,title) in [(1,1,"Diagonal"), (1,2,"Whitney"), 
                              (2,1,"LSQ Linear"), (2,2,"LSQ Quadratic")]]
    
    # Colors for different operators
    colors = [:red, :blue, :green, :orange]
    
    # Compute and plot eigenvalues for each operator
    for (i, hodge_type) in enumerate(["diagonal", "whitney", "lsq_linear", "lsq_quadratic"])
        # Compute operator
        op = compute_advection_operator(mesh, velocity, hodge_type, mesh_level, regular)
        
        # Compute eigenvalues
        λ = eigvals(Matrix(op))
        
        # Plot eigenvalues
        scatter!(axs[i], real.(λ), imag.(λ), 
                color=colors[i],
                markersize=5)
        
        # Add zero reference lines
        hlines!(axs[i], [0], color=:gray, linestyle=:dash)
        vlines!(axs[i], [0], color=:gray, linestyle=:dash)
        
    end
    
    return fig
end

# Test with constant velocity field
velocity = x -> [1.0, 1.0]

# Plot eigenvalues for different mesh levels
for i in 1:5
    println("Computing eigenvalues for mesh level $i")
    
    # Regular mesh
    mesh = VoronoiMesh("meshes/mesh$(i)_regular.nc")
    fig = plot_eigenvalues(mesh, velocity, i, true)
    CairoMakie.save("results/eigenvalues/eigenvalues_mesh$(i)_regular.pdf", fig)    
    # Irregular mesh
    mesh = VoronoiMesh("meshes/mesh$(i).nc")
    fig = plot_eigenvalues(mesh, velocity, i, false)
    CairoMakie.save("results/eigenvalues/eigenvalues_mesh$(i)_irregular.pdf", fig)
end

# Test with rotating velocity field
velocity2 = x -> [sin(2pi*x[1])*cos(2pi*x[2]), -sin(2pi*x[2])*cos(2pi*x[1])]

# Plot eigenvalues for different mesh levels with rotating field
for i in 1:5
    println("Computing eigenvalues for mesh level $i with rotating field")
    
    # Regular mesh
    mesh = VoronoiMesh("meshes/mesh$(i)_regular.nc")
    fig = plot_eigenvalues(mesh, velocity2, i, true)
    CairoMakie.save("results/eigenvalues/eigenvalues_rotating_mesh$(i)_regular.pdf", fig)
    
    # Irregular mesh
    mesh = VoronoiMesh("meshes/mesh$(i).nc")
    fig = plot_eigenvalues(mesh, velocity2, i, false)
    CairoMakie.save("results/eigenvalues/eigenvalues_rotating_mesh$(i)_irregular.pdf", fig)
end 