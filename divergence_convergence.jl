using VoronoiMeshes, TensorsLite, TensorsLiteGeometry, ImmutableVectors, LinearAlgebra, NCDatasets, MPASTools
using StaticArrays, SparseArrays
using CairoMakie, Printf, DataFrames, CSV, JLD2
include("juliaDEC.jl")

# Load pre-computed matrices
@load "lsq_matrices.jld2" matrices_regular matrices_irregular

# Define the analytical solution
function analytical_divergence(x)
    # For u(x,y) = 1 and v = [sin(2pi*x)cos(2pi*y), -cos(2pi*x)sin(2pi*y)]
    return 0.0
end

function compute_discrete_divergence(mesh::VoronoiMesh, velocity::Function, hodge_type::String, mesh_level::Int, regular::Bool=true)
    
    # Create the function u(x,y) 
    field = x -> 1.0
    u = Cochain(field.(mesh.cells.position) .* mesh.cells.area, 2, false);
    
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
        elseif hodge_type == "lsq_quadratic"
            star1 = matrices_irregular[mesh_level]["lsq_quadratic"]
        end
    end
    
    # Compute the discrete divergence operator
    div_op = d1 * star1 * wedge10 * star2
    
    # Apply the operator to u
    discrete_div = div_op * u.values
    
    return discrete_div
end

function analyze_convergence(regular::Bool=true)
    # Initialize arrays to store results
    mesh_levels = 1:7
    errors = Dict(
        "diagonal" => Float64[],
        "whitney" => Float64[],
        "lsq_linear" => Float64[],
        "lsq_quadratic" => Float64[]
    )
    mesh_sizes = Float64[]
    
    #velocity field
    velocity = x -> [sin(2pi*x[1])*cos(2pi*x[2]), -cos(2pi*x[1])*sin(2pi*x[2])]
    
    # Analyze convergence for each mesh level
    for level in mesh_levels
        println("Computing convergence for mesh level $level")
        
        # Load mesh
        mesh = regular ? 
            VoronoiMesh("meshes/mesh$(level)_regular.nc") :
            VoronoiMesh("meshes/mesh$(level).nc")
        
        # Store mesh size
        push!(mesh_sizes, maximum(mesh.edges.lengthDual))
        
        # Compute errors for each Hodge star type
        for hodge_type in ["diagonal", "whitney", "lsq_linear", "lsq_quadratic"]
            discrete_div = compute_discrete_divergence(mesh, velocity, hodge_type, level, regular)
            error = maximum(abs.(discrete_div - analytical_divergence.(mesh.cells.position)))
            push!(errors[hodge_type], error)
        end
    end
    
    # Create convergence plot
    fig = Figure(size=(800, 600))
    ax = Axis(fig[1,1],
              xlabel="Mesh Size (h)",
              ylabel="Maximum Error",
              xscale=log2,
              yscale=log10,
              title="Convergence of Divergence Operator $regular mesh")
    
    # Plot errors for each Hodge star type
    colors = [:red, :blue, :green, :orange]
    for (i, hodge_type) in enumerate(["diagonal", "whitney", "lsq_linear", "lsq_quadratic"])
        scatterlines!(ax, mesh_sizes, errors[hodge_type], 
               color=colors[i], 
               label=hodge_type,
               marker=:circle)
    end
    axislegend(ax, position = :rb)
    
    # Save plot
    mesh_type = regular ? "regular" : "irregular"
    CairoMakie.save("results/divergence_convergence_$(mesh_type).pdf", fig)
    
    # Compute convergence rates
    println("\nConvergence rates:")
    for hodge_type in ["diagonal", "whitney", "lsq_linear", "lsq_quadratic"]
        # Compute slope using linear regression in log-log space
        e = errors[hodge_type]
        h = mesh_sizes
        slope = log(e[end]/e[end-1])/log(h[end]/h[end-1])
        println("$hodge_type: $(slope)")  # Negative because we want positive convergence rate
    end
    
    return errors, mesh_sizes
end

# Plot error distribution for mesh level 6
function plot_error_distribution(regular::Bool=true)
    # Load mesh
    mesh = regular ? 
        VoronoiMesh("meshes/mesh6_regular.nc") :
        VoronoiMesh("meshes/mesh6.nc")
    
    # Velocity field
    velocity = x -> [sin(2pi*x[1])*cos(2pi*x[2]), -cos(2pi*x[1])*sin(2pi*x[2])]
    
    # Create figure for each Hodge star type
    for hodge_type in ["diagonal", "whitney", "lsq_linear", "lsq_quadratic"]
        # Compute discrete divergence
        discrete_div = compute_discrete_divergence(mesh, velocity, hodge_type, 6, regular)
        
        # Compute error
        error = discrete_div - analytical_divergence.(mesh.cells.position)
        
        # Create error cochain
        error_cochain = Cochain(error, 2, false)
        
        # Create figure
        fig = Figure(size=(800, 800))
        ax = Axis(fig[1,1], 
                 aspect=1,
                 title="Error Distribution - $regular mesh - $hodge_type")
        
        # Plot error distribution
        plot_cochain(fig, ax, error_cochain, mesh, :viridis)
        
        # Save plot
        mesh_type = regular ? "regular" : "irregular"
        CairoMakie.save("results/divergence_error_mesh6_$(mesh_type)_$(hodge_type).pdf", fig)
    end
end

# Run convergence analysis for both regular and irregular meshes
println("Analyzing convergence for regular meshes...")
regular_errors, regular_sizes = analyze_convergence(true)

println("\nAnalyzing convergence for irregular meshes...")
irregular_errors, irregular_sizes = analyze_convergence(false)

# Plot error distributions for mesh level 6
println("\nPlotting error distributions for mesh level 6...")
plot_error_distribution(true)  # Regular mesh
plot_error_distribution(false) # Irregular mesh 