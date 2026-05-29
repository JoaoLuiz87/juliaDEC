using VoronoiMeshes, TensorsLite, TensorsLiteGeometry, SmallCollections, LinearAlgebra, NCDatasets, MPASTools
using StaticArrays, SparseArrays
using CairoMakie, JLD2
include("juliaDEC.jl")

# Load pre-computed LSQ matrices
@load "lsq_matrices.jld2" matrices_regular matrices_irregular

"""
    compute_laplacian_matrix(mesh::VoronoiMesh, hodge_type::String, mesh_level::Int, regular::Bool=true)

Compute the Laplacian matrix using different Hodge star operators:
- diagonal: Standard diagonal Hodge star
- whitney: Whitney Hodge star
- lsq1: Linear least squares Hodge star
- lsq2: Quadratic least squares Hodge star
"""
function compute_laplacian_matrix(mesh::VoronoiMesh, hodge_type::String, mesh_level::Int, regular::Bool=true)
    # Get the differential operators
    d0 = d0_matrix_primal(mesh)  # 0-form to 1-form
    d1 = d1_matrix_dual(mesh)    # 1-form to 2-form
    star2 = star(mesh, 2, false)
    
    # Choose Hodge star operator for 1-forms based on type
    if regular
        if hodge_type == "diagonal"
            star1 = star(mesh, 1, true)
        elseif hodge_type == "whitney"
            star1 = whitney_hodge_star_matrix(mesh)
        elseif hodge_type == "lsq1"
            star1 = matrices_regular[mesh_level]["lsq_linear"]
            star1 = 0.5*(star1 + star1')
        elseif hodge_type == "lsq2"
            star1 = matrices_regular[mesh_level]["lsq_quadratic"]
            star1 = 0.5*(star1 + star1')
        else
            error("Unknown hodge_type: $hodge_type")
        end
    else
        if hodge_type == "diagonal"
            star1 = star(mesh, 1, true)
        elseif hodge_type == "whitney"
            star1 = whitney_hodge_star_matrix(mesh)
        elseif hodge_type == "lsq1"
            star1 = matrices_irregular[mesh_level]["lsq_linear"]
            star1 = 0.5*(star1 + star1')
        elseif hodge_type == "lsq2"
            star1 = matrices_irregular[mesh_level]["lsq_quadratic"]
            star1 = 0.5*(star1 + star1')
        else
            error("Unknown hodge_type: $hodge_type")
        end
    end
    
    # Construct the Laplacian matrix: Δ = *d*d
    # For 0-forms: Δ = *d₁*d₀ where d₀: 0-form → 1-form, d₁: 1-form → 2-form
    laplacian = star2 * d1 * star1 * d0
    
    return laplacian
end

"""
    compute_eigenvalues(laplacian_matrix::SparseMatrixCSC)

Compute eigenvalues of the Laplacian matrix.
Returns all eigenvalues (real and complex).
"""
function compute_eigenvalues(laplacian_matrix::SparseMatrixCSC)
    # Convert to dense matrix for eigenvalue computation
    L = Matrix(laplacian_matrix)
    
    # Compute eigenvalues
    λ = eigvals(L)
    
    # Sort by magnitude
    sort!(λ, by=abs)
    
    return λ
end

"""
    plot_eigenvalue_distribution(eigenvalues_dict::Dict, mesh_type::String, mesh_level::Int, mesh_size::Float64)

Plot the distribution of eigenvalues for different Hodge star operators.
"""
function plot_eigenvalue_distribution(eigenvalues_dict::Dict, mesh_type::String, mesh_level::Int, mesh_size::Float64)
    # Create figure with multiple subplots
    fig = Figure(size=(1600, 1200))
    
    # Create subplots with larger font sizes and mesh size in title
    ax1 = Axis(fig[1,1], 
                xlabel="Re(λ)", 
                ylabel="Im(λ)",
                title="Eigenvalues in Complex Plane - $mesh_type Mesh $mesh_level (h = $(round(mesh_size, digits=4)))",
                aspect=1,
                titlesize=18,
                xlabelsize=16,
                ylabelsize=16,
                xticklabelsize=14,
                yticklabelsize=14)
    
    ax2 = Axis(fig[1,2], 
                xlabel="Eigenvalue Index", 
                ylabel="|λ|",
                title="Eigenvalue Magnitudes - $mesh_type Mesh $mesh_level (h = $(round(mesh_size, digits=4)))",
                yscale=log10,
                titlesize=18,
                xlabelsize=16,
                ylabelsize=16,
                xticklabelsize=14,
                yticklabelsize=14)
    
    ax3 = Axis(fig[2,1], 
                xlabel="Re(λ)", 
                ylabel="Frequency",
                title="Real Part Distribution - $mesh_type Mesh $mesh_level (h = $(round(mesh_size, digits=4)))",
                titlesize=18,
                xlabelsize=16,
                ylabelsize=16,
                xticklabelsize=14,
                yticklabelsize=14)
    
    ax4 = Axis(fig[2,2], 
                xlabel="|λ|", 
                ylabel="Frequency",
                title="Magnitude Distribution - $mesh_type Mesh $mesh_level (h = $(round(mesh_size, digits=4)))",
                titlesize=18,
                xlabelsize=16,
                ylabelsize=16,
                xticklabelsize=14,
                yticklabelsize=14)
    
    # Colors for different operators
    colors = [:red, :blue, :green, :orange]
    markers = [:circle, :diamond, :hexagon, :star5]
    
    # Plot eigenvalues for each operator
    for (i, (hodge_type, eigenvalues)) in enumerate(eigenvalues_dict)
        # Plot eigenvalues in complex plane
        scatter!(ax1, real.(eigenvalues), imag.(eigenvalues), 
                color=colors[i], 
                marker=markers[i],
                label=hodge_type,
                markersize=4)
        
        # Plot eigenvalue magnitudes vs index
        scatterlines!(ax2, 1:length(eigenvalues), abs.(eigenvalues), 
                     color=colors[i], 
                     marker=markers[i],
                     label=hodge_type,
                     markersize=3)
        
        # Plot histogram of real parts
        hist!(ax3, real.(eigenvalues), 
              color=colors[i], 
              label=hodge_type,
              bins=30)
        
        # Plot histogram of magnitudes
        hist!(ax4, abs.(eigenvalues), 
              color=colors[i], 
              label=hodge_type,
              bins=30)
    end
    
    # Add legends
    axislegend(ax1, position=:lt)
    axislegend(ax2, position=:lt)
    axislegend(ax3, position=:rt)
    axislegend(ax4, position=:rt)
    
    # Add reference lines for complex plane
    hlines!(ax1, [0], color=:gray, linestyle=:dash, alpha=0.5)
    vlines!(ax1, [0], color=:gray, linestyle=:dash, alpha=0.5)
    
    return fig
end

"""
    plot_complex_eigenvalue_comparison(eigenvalues_dict::Dict, mesh_type::String, mesh_level::Int, mesh_size::Float64)

Create a detailed comparison of eigenvalues in the complex plane.
"""
function plot_complex_eigenvalue_comparison(eigenvalues_dict::Dict, mesh_type::String, mesh_level::Int, mesh_size::Float64)
    fig = Figure(size=(1200, 800))
    
    # Create subplots for each Hodge star operator
    hodge_types = ["diagonal", "whitney", "lsq1", "lsq2"]
    colors = [:red, :blue, :green, :orange]
    
    for (i, hodge_type) in enumerate(hodge_types)
        ax = Axis(fig[div(i-1,2)+1, mod(i-1,2)+1],
                  xlabel="Re(λ)",
                  ylabel="Im(λ)",
                  title="$hodge_type Hodge Star - $mesh_type Mesh $mesh_level (h = $(round(mesh_size, digits=4)))",
                  aspect=1,
                  titlesize=16,
                  xlabelsize=14,
                  ylabelsize=14,
                  xticklabelsize=12,
                  yticklabelsize=12)
        
        eigenvalues = eigenvalues_dict[hodge_type]
        
        # Plot all eigenvalues in complex plane
        scatter!(ax, real.(eigenvalues), imag.(eigenvalues), 
                color=colors[i], 
                marker=:circle,
                markersize=4)
        
        # Add reference lines
        hlines!(ax, [0], color=:gray, linestyle=:dash, alpha=0.5)
        vlines!(ax, [0], color=:gray, linestyle=:dash, alpha=0.5)
    end
    
    return fig
end

"""
    analyze_laplacian_eigenvalues(mesh_level::Int=4)

Main function to analyze Laplacian eigenvalues for mesh 4 (both regular and irregular).
"""
function analyze_laplacian_eigenvalues(mesh_level::Int=4)
    println("Computing Laplacian eigenvalues for mesh level $mesh_level")
    
    # Test both regular and irregular meshes
    for (mesh_type, regular) in [("Regular", true), ("Irregular", false)]
        println("\nProcessing $mesh_type mesh...")
        
        # Load mesh
        mesh_filename = regular ? "meshes/mesh$(mesh_level)_regular.nc" : "meshes/mesh$(mesh_level).nc"
        mesh = VoronoiMesh(mesh_filename)
        
        println("Mesh loaded: $(mesh.cells.n) cells, $(mesh.edges.n) edges, $(mesh.vertices.n) vertices")
        
        # Dictionary to store eigenvalues for each Hodge star operator
        eigenvalues_dict = Dict{String, Vector{ComplexF64}}()
        
        # Compute eigenvalues for each Hodge star operator
        for hodge_type in ["diagonal", "whitney", "lsq1", "lsq2"]
            println("Computing eigenvalues for $hodge_type Hodge star...")
            
            # Compute Laplacian matrix
            laplacian_matrix = compute_laplacian_matrix(mesh, hodge_type, mesh_level, regular)
            
            # Compute eigenvalues
            eigenvalues = compute_eigenvalues(laplacian_matrix)
            
            # Store results
            eigenvalues_dict[hodge_type] = eigenvalues
            
            println("  Found $(length(eigenvalues)) eigenvalues")
            println("  Range: |λ| ∈ [$(minimum(abs.(eigenvalues))), $(maximum(abs.(eigenvalues)))]")
            
            # Find the eigenvalue with maximum modulus
            max_mod_eigenvalue = eigenvalues[argmax(abs.(eigenvalues))]
            println("  Max modulus eigenvalue: λ = $(real(max_mod_eigenvalue)) + $(imag(max_mod_eigenvalue))im (|λ| = $(abs(max_mod_eigenvalue)))")
            
            println("  Condition number: $(maximum(abs.(eigenvalues))/minimum(abs.(eigenvalues)))")
        end
        
        # Get mesh size for titles
        mesh_size = maximum(mesh.edges.lengthDual)
        
        # Create plots
        fig1 = plot_eigenvalue_distribution(eigenvalues_dict, mesh_type, mesh_level, mesh_size)
        fig2 = plot_complex_eigenvalue_comparison(eigenvalues_dict, mesh_type, mesh_level, mesh_size)
        
        # Save plots
        filename1 = "results/laplacian_eigenvalues_mesh$(mesh_level)_$(lowercase(mesh_type)).pdf"
        filename2 = "results/laplacian_complex_eigenvalues_mesh$(mesh_level)_$(lowercase(mesh_type)).pdf"
        CairoMakie.save(filename1, fig1)
        CairoMakie.save(filename2, fig2)
        println("Plots saved to $filename1 and $filename2")
        
        # Print summary statistics
        println("\nSummary for $mesh_type mesh (h = $(round(mesh_size, digits=4))):")
        println("="^60)
        for (hodge_type, eigenvalues) in eigenvalues_dict
            println("$hodge_type:")
            println("  Total eigenvalues: $(length(eigenvalues))")
            println("  Min magnitude: $(minimum(abs.(eigenvalues)))")
            println("  Max magnitude: $(maximum(abs.(eigenvalues)))")
            
            # Find the eigenvalue with maximum modulus
            max_mod_eigenvalue = eigenvalues[argmax(abs.(eigenvalues))]
            println("  Max modulus eigenvalue: λ = $(real(max_mod_eigenvalue)) + $(imag(max_mod_eigenvalue))im (|λ| = $(abs(max_mod_eigenvalue)))")
            
            println("  Condition number: $(maximum(abs.(eigenvalues))/minimum(abs.(eigenvalues)))")
            println()
        end
    end
end

"""
    compare_eigenvalue_spectra(mesh_level::Int=4)

Compare eigenvalue spectra between regular and irregular meshes.
"""
function compare_eigenvalue_spectra(mesh_level::Int=4)
    println("Comparing eigenvalue spectra for mesh level $mesh_level")
    
    # Load both meshes
    mesh_regular = VoronoiMesh("meshes/mesh$(mesh_level)_regular.nc")
    mesh_irregular = VoronoiMesh("meshes/mesh$(mesh_level).nc")
    
    # Get mesh sizes for titles
    mesh_size_regular = maximum(mesh_regular.edges.lengthDual)
    mesh_size_irregular = maximum(mesh_irregular.edges.lengthDual)
    
    # Dictionary to store results
    results = Dict{String, Dict{String, Vector{ComplexF64}}}()
    
    # Compute eigenvalues for both meshes and all Hodge star operators
    for (mesh_type, mesh, regular) in [("Regular", mesh_regular, true), ("Irregular", mesh_irregular, false)]
        results[mesh_type] = Dict{String, Vector{ComplexF64}}()
        
        for hodge_type in ["diagonal", "whitney", "lsq1", "lsq2"]
            laplacian_matrix = compute_laplacian_matrix(mesh, hodge_type, mesh_level, regular)
            eigenvalues = compute_eigenvalues(laplacian_matrix)
            results[mesh_type][hodge_type] = eigenvalues
        end
    end
    
    # Create comparison plots
    fig = Figure(size=(1600, 1200))
    
    # Create subplots for each Hodge star operator
    hodge_types = ["diagonal", "whitney", "lsq1", "lsq2"]
    colors = [:red, :blue]
    
    for (i, hodge_type) in enumerate(hodge_types)
        ax = Axis(fig[div(i-1,2)+1, mod(i-1,2)+1],
                  xlabel="Re(λ)",
                  ylabel="Im(λ)",
                  title="$hodge_type Hodge Star",
                  aspect=1,
                  titlesize=18,
                  xlabelsize=16,
                  ylabelsize=16,
                  xticklabelsize=14,
                  yticklabelsize=14)
        
        # Plot both regular and irregular
        for (j, mesh_type) in enumerate(["Regular", "Irregular"])
            eigenvalues = results[mesh_type][hodge_type]
            mesh_size = mesh_type == "Regular" ? mesh_size_regular : mesh_size_irregular
            
            scatter!(ax, real.(eigenvalues), imag.(eigenvalues),
                     color=colors[j],
                     marker=:circle,
                     label="$mesh_type (h = $(round(mesh_size, digits=4)))",
                     markersize=8)
        end
        
        # Add reference lines
        hlines!(ax, [0], color=:gray, linestyle=:dash, alpha=0.5)
        vlines!(ax, [0], color=:gray, linestyle=:dash, alpha=0.5)
        
        axislegend(ax, position=:lt)
    end
    
    # Save comparison plot
    filename = "laplacian_eigenvalues/laplacian_eigenvalues_comparison_mesh$(mesh_level).pdf"
    CairoMakie.save(filename, fig)
    println("Comparison plot saved to $filename")
    
    return results
end

# Run the analysis
println("Starting Laplacian eigenvalue analysis...")

# Create results directory if it doesn't exist
mkpath("laplacian_eigenvalues")

# Function to save mesh size information
function save_mesh_sizes()
    mesh_sizes = Dict{Int, Dict{String, Float64}}()
    
    for mesh_level in [2, 3, 4]
        mesh_sizes[mesh_level] = Dict{String, Float64}()
        
        # Regular mesh
        try
            mesh_regular = VoronoiMesh("meshes/mesh$(mesh_level)_regular.nc")
            mesh_sizes[mesh_level]["regular"] = maximum(mesh_regular.edges.lengthDual)
        catch
            mesh_sizes[mesh_level]["regular"] = NaN
        end
        
        # Irregular mesh
        try
            mesh_irregular = VoronoiMesh("meshes/mesh$(mesh_level).nc")
            mesh_sizes[mesh_level]["irregular"] = maximum(mesh_irregular.edges.lengthDual)
        catch
            mesh_sizes[mesh_level]["irregular"] = NaN
        end
    end
    
    # Save to file
    open("laplacian_eigenvalues/mesh_sizes.txt", "w") do io
        println(io, "Mesh Size Information (h = maximum edge length)")
        println(io, "="^50)
        for mesh_level in sort(collect(keys(mesh_sizes)))
            println(io, "\nMesh Level $mesh_level:")
            println(io, "  Regular:   h = $(round(mesh_sizes[mesh_level]["regular"], digits=6))")
            println(io, "  Irregular: h = $(round(mesh_sizes[mesh_level]["irregular"], digits=6))")
        end
    end
    
    println("Mesh sizes saved to laplacian_eigenvalues/mesh_sizes.txt")
    return mesh_sizes
end

"""
    analyze_imaginary_convergence()

Analyze the convergence of imaginary parts of eigenvalues as mesh is refined.
"""
function analyze_imaginary_convergence()
    println("\n" * "="^60)
    println("ANALYZING IMAGINARY PART CONVERGENCE")
    println("="^60)
    
    # Dictionary to store results for convergence analysis
    convergence_data = Dict{String, Dict{String, Vector{Tuple{Float64, Float64}}}}()
    
    # Initialize for each Hodge type
    for hodge_type in ["diagonal", "whitney", "lsq1", "lsq2"]
        convergence_data[hodge_type] = Dict{String, Vector{Tuple{Float64, Float64}}}()
        convergence_data[hodge_type]["regular"] = []
        convergence_data[hodge_type]["irregular"] = []
    end
    
    # Collect data for each mesh level
    for mesh_level in [2, 3, 4]
        println("\nMesh Level $mesh_level:")
        
        for (mesh_type, regular) in [("Regular", true), ("Irregular", false)]
            try
                # Load mesh
                mesh_filename = regular ? "meshes/mesh$(mesh_level)_regular.nc" : "meshes/mesh$(mesh_level).nc"
                mesh = VoronoiMesh(mesh_filename)
                mesh_size = maximum(mesh.edges.lengthDual)
                
                println("  $mesh_type mesh (h = $(round(mesh_size, digits=6))):")
                
                for hodge_type in ["diagonal", "whitney", "lsq1", "lsq2"]
                    # Compute eigenvalues
                    laplacian_matrix = compute_laplacian_matrix(mesh, hodge_type, mesh_level, regular)
                    eigenvalues = compute_eigenvalues(laplacian_matrix)
                    
                    # Analyze imaginary parts
                    imag_parts = imag.(eigenvalues)
                    max_imag = maximum(abs.(imag_parts))
                    mean_imag = mean(abs.(imag_parts))
                    
                    # Store for convergence analysis
                    mesh_type_key = lowercase(mesh_type)
                    push!(convergence_data[hodge_type][mesh_type_key], (mesh_size, max_imag))
                    
                    println("    $hodge_type: max|Im(λ)| = $(round(max_imag, digits=10)), mean|Im(λ)| = $(round(mean_imag, digits=10))")
                end
                
            catch e
                println("  Error processing $mesh_type mesh level $mesh_level: $e")
            end
        end
    end
    
    # Analyze convergence rates
    println("\n" * "="^60)
    println("CONVERGENCE ANALYSIS")
    println("="^60)
    
    for hodge_type in ["diagonal", "whitney", "lsq1", "lsq2"]
        println("\n$hodge_type Hodge Star:")
        
        for mesh_type in ["regular", "irregular"]
            data = convergence_data[hodge_type][mesh_type]
            if length(data) >= 2
                # Sort by mesh size (descending)
                sort!(data, by=x->x[1], rev=true)
                
                println("  $mesh_type mesh:")
                for i in 1:length(data)
                    h, max_imag = data[i]
                    println("    h = $(round(h, digits=6)): max|Im(λ)| = $(round(max_imag, digits=10))")
                end
                
                # Compute convergence rate if we have at least 2 points
                if length(data) >= 2
                    # Use the first two points to estimate convergence rate
                    h1, max_imag1 = data[1]
                    h2, max_imag2 = data[2]
                    
                    if max_imag1 > 0 && max_imag2 > 0
                        # Estimate convergence rate: max_imag ∝ h^p
                        # log(max_imag2/max_imag1) = p * log(h2/h1)
                        # p = log(max_imag2/max_imag1) / log(h2/h1)
                        convergence_rate = log(max_imag2/max_imag1) / log(h2/h1)
                        println("    Estimated convergence rate: p ≈ $(round(convergence_rate, digits=3))")
                        println("    (max|Im(λ)| ∝ h^$(round(convergence_rate, digits=3)))")
                    end
                end
            end
        end
    end
    
    # Save convergence data to CSV for further analysis
    save_convergence_data(convergence_data)
    
    return convergence_data
end

"""
    save_convergence_data(convergence_data)

Save convergence data to CSV files for plotting and analysis.
"""
function save_convergence_data(convergence_data)
    # Create directory if it doesn't exist
    mkpath("laplacian_eigenvalues/convergence")
    
    for hodge_type in ["diagonal", "whitney", "lsq1", "lsq2"]
        for mesh_type in ["regular", "irregular"]
            data = convergence_data[hodge_type][mesh_type]
            if !isempty(data)
                # Sort by mesh size
                sort!(data, by=x->x[1])
                
                filename = "laplacian_eigenvalues/convergence/$(hodge_type)_$(mesh_type)_imaginary_convergence.csv"
                open(filename, "w") do io
                    println(io, "mesh_size,max_imaginary")
                    for (h, max_imag) in data
                        println(io, "$(h),$(max_imag)")
                    end
                end
                println("Convergence data saved to $filename")
            end
        end
    end
end

# Save mesh size information
mesh_sizes = save_mesh_sizes()

# Analyze eigenvalues for meshes 4, 5, and 6
for mesh_level in [2, 3, 4]
    println("\n" * "="^60)
    println("ANALYZING MESH LEVEL $mesh_level")
    println("="^60)
    
    # Analyze eigenvalues for current mesh level
    analyze_laplacian_eigenvalues(mesh_level)
    
    # Compare regular vs irregular for current mesh level
    results = compare_eigenvalue_spectra(mesh_level)
end

println("\nAnalysis complete for all mesh levels!")

# Run convergence analysis for imaginary parts
println("\n" * "="^60)
println("STARTING CONVERGENCE ANALYSIS")
println("="^60)
convergence_results = analyze_imaginary_convergence()