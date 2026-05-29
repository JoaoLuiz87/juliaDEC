using VoronoiMeshes,TensorsLite, TensorsLiteGeometry, ImmutableVectors, LinearAlgebra, NCDatasets, MPASTools
using StaticArrays, SparseArrays
using CairoMakie, Printf, DataFrames, CSV, JLD2    
include("juliaDEC.jl")

# Parameters to control test type
const USE_REGULAR_MESH = false  # Set to true for regular mesh, false for irregular mesh
const USE_CONSTANT_FIELD = true  # Set to true for constant field, false for rotating field

function rk4_step(f, u, t, dt)
    k1 = f(u, t)
    k2 = f(u + 0.5*dt*k1, t + 0.5*dt)
    k3 = f(u + 0.5*dt*k2, t + 0.5*dt)
    k4 = f(u + dt*k3, t + dt)
    return u + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
end

struct AdvectionProblem
    mesh::VoronoiMesh
    velocity::Function
    initial_condition::Function
    dt::Float64
    T::Float64
end

function initial_condition(x)
    xp = 1.0
    yp = 1.0
    center = [0.5, 0.5]
    sigma = 0.05

    x_vec = TensorsLite.Vec(x = x[1], y = x[2])
    center_vec = TensorsLite.Vec(x = center[1], y = center[2])
    x_raw = closest(center_vec, x_vec, xp, yp)
    y = [x_raw.x, x_raw.y]
    return exp(-(norm(y - center))^2/(2*sigma^2))
end

function solution(x)
    xp = 1.0
    yp = 1.0
    center = [0.5 + 50*0.00033050301170003726, 0.5]
    sigma = 0.05

    x_vec = TensorsLite.Vec(x = x[1], y = x[2])   
    center_vec = TensorsLite.Vec(x = center[1], y = center[2])
    x_raw = closest(center_vec, x_vec, xp, yp)
    y = [x_raw.x, x_raw.y]
    return exp(-(norm(y - center))^2/(2*sigma^2))
end

@load "lsq_matrices.jld2" matrices_regular matrices_irregular

# Function to run simulation with different Hodge operators
function run_simulation(mesh::VoronoiMesh, velocity, dt, T1, hodge_type, mesh_level, regular::Bool=true)
    dt = 0.00033050301170003726
    problem = AdvectionProblem(mesh, velocity, initial_condition, dt, T1)
    steps = Int(floor(problem.T / problem.dt))
    d1 = d1_matrix_dual(problem.mesh)
    v = discretize_vector_field_dual(problem.mesh, problem.velocity)
    u = Cochain(initial_condition.(mesh.cells.centroid) .* mesh.cells.area, 2, false)
    
    wedge10 = wedge_product_10_matrix(problem.mesh, v)
    
    if regular
    # Choose Hodge star operator based on type
        if hodge_type == "diagonal"
            star1 = star(problem.mesh, 1, true)
        elseif hodge_type == "whitney"
            star1 = whitney_hodge_star_matrix(problem.mesh)
        elseif hodge_type == "lsq_linear"
            star1 = matrices_regular[mesh_level]["lsq_linear"]  # Use pre-computed matrix
        elseif hodge_type == "lsq_quadratic"
            star1 = matrices_regular[mesh_level]["lsq_quadratic"]
        end
    else
        if hodge_type == "diagonal"
            star1 = star(problem.mesh, 1, true)
        elseif hodge_type == "whitney"
            star1 = whitney_hodge_star_matrix(problem.mesh)
        elseif hodge_type == "lsq_linear"
            star1 = matrices_irregular[mesh_level]["lsq_linear"]  # Use pre-computed matrix
        elseif hodge_type == "lsq_quadratic"
            star1 = matrices_irregular[mesh_level]["lsq_quadratic"]
        end
    end
    
    star2 = star(problem.mesh, 2, false)
    
    function rhs(u_values, t)
        return - d1 * star1 * wedge10 * star2 * u_values
    end
    
    t = 0.0
    for n in 1:steps
        u = Cochain(rk4_step(rhs, u.values, t, problem.dt), 2, false)
        t += problem.dt
    end
    
    # Relative error
    error_Linf = maximum(abs.(u.values - solution.(mesh.cells.position) .* mesh.cells.area))
    error_L2 = norm(u.values - solution.(mesh.cells.position) .* mesh.cells.area)
    return dt, error_L2, error_Linf
end

# Function to compute convergence order
function compute_order(errors, dxs)
    if length(errors) < 2
        return 0.0
    end
    return log(errors[end]/errors[end-1])/log(dxs[end]/dxs[end-1])
end

# Function to run reversible flow simulation
function run_reversible_simulation(mesh::VoronoiMesh, velocity, reverse_velocity, dt, T1, hodge_type, mesh_level, regular::Bool=true)
    
    # Initial condition
    u_initial = Cochain(initial_condition.(mesh.cells.centroid) .* mesh.cells.area, 2, false)
    
    # Forward flow for T/2
    problem_forward = AdvectionProblem(mesh, velocity, initial_condition, dt, T1/2)
    steps_forward = Int(floor(problem_forward.T / problem_forward.dt))
    
    d1 = d1_matrix_dual(problem_forward.mesh)
    v_forward = discretize_vector_field_dual(problem_forward.mesh, problem_forward.velocity)
    u = Cochain(initial_condition.(mesh.cells.centroid) .* mesh.cells.area, 2, false)
    
    wedge10_forward = wedge_product_10_matrix(problem_forward.mesh, v_forward)
    
    # Choose Hodge star operator based on type
    if regular
        if hodge_type == "diagonal"
            star1 = star(problem_forward.mesh, 1, true)
        elseif hodge_type == "whitney"
            star1 = whitney_hodge_star_matrix(problem_forward.mesh)
        elseif hodge_type == "lsq_linear"
            star1 = matrices_regular[mesh_level]["lsq_linear"]
        elseif hodge_type == "lsq_quadratic"
            star1 = matrices_regular[mesh_level]["lsq_quadratic"]
        end
    else
        if hodge_type == "diagonal"
            star1 = star(problem_forward.mesh, 1, true)
        elseif hodge_type == "whitney"
            star1 = whitney_hodge_star_matrix(problem_forward.mesh)
        elseif hodge_type == "lsq_linear"
            star1 = matrices_irregular[mesh_level]["lsq_linear"]
        elseif hodge_type == "lsq_quadratic"
            star1 = matrices_irregular[mesh_level]["lsq_quadratic"]
        end
    end
    
    star2 = star(problem_forward.mesh, 2, false)
    
    # Forward flow
    function rhs_forward(u_values, t)
        return - d1 * star1 * wedge10_forward * star2 * u_values
    end
    
    t = 0.0
    for n in 1:steps_forward
        u = Cochain(rk4_step(rhs_forward, u.values, t, problem_forward.dt), 2, false)
        t += problem_forward.dt
    end
    
    # Store solution at T/2
    u_half = deepcopy(u)
    
    # Reverse flow for T/2
    problem_reverse = AdvectionProblem(mesh, reverse_velocity, x -> u.values[findcell(mesh, x)]*mesh.cells.area[findcell(mesh, x)], dt, T1/2)
    steps_reverse = Int(floor(problem_reverse.T / problem_reverse.dt))
    
    v_reverse = discretize_vector_field_dual(problem_reverse.mesh, problem_reverse.velocity)
    wedge10_reverse = wedge_product_10_matrix(problem_reverse.mesh, v_reverse)
    
    # Reverse flow
    function rhs_reverse(u_values, t)
        return - d1 * star1 * wedge10_reverse * star2 * u_values
    end
    
    t = 0.0
    for n in 1:steps_reverse
        u = Cochain(rk4_step(rhs_reverse, u.values, t, problem_reverse.dt), 2, false)
        t += problem_reverse.dt
    end
    
    # Relative error
    error_Linf = maximum(abs.(u.values - initial_condition.(mesh.cells.centroid) .* mesh.cells.area))
    error_L2 = norm(u.values - initial_condition.(mesh.cells.centroid) .* mesh.cells.area)
    
    return dt, error_L2, error_Linf, u_initial, u_half, u
end

# Function to plot solution
function plot_solution(mesh::VoronoiMesh, u::Cochain, title::String="Solution")
    fig = Figure(size=(800, 800))
    ax = Axis(fig[1,1], 
             aspect=1,
             title=title)
    
    # Plot the solution
    plot_cochain(fig, ax, u, mesh, :viridis)
    
    # Add colorbar
    Colorbar(fig[1,2], limits=(minimum(u.values), maximum(u.values)))
    
    return fig
end

# TEST WITH CONSTANT FIELD FOR CONVERGENCE
T1 = 50*0.00033050301170003726
velocity = x -> [1.0, 0.0]

# Initialize data structures for results
mesh_sizes = []
results = Dict(
    "diagonal" => Dict("dt" => [], "L2" => [], "Linf" => []),
    "whitney" => Dict("dt" => [], "L2" => [], "Linf" => []),
    "lsq_linear" => Dict("dt" => [], "L2" => [], "Linf" => []),
    "lsq_quadratic" => Dict("dt" => [], "L2" => [], "Linf" => [])
)

if USE_CONSTANT_FIELD
    # Run simulations for each mesh size and Hodge operator
    for i in 1:7
        println("i = $i ------------------------------------------------------------")
        mesh = USE_REGULAR_MESH ? VoronoiMesh("meshes/mesh$(i)_regular.nc") : VoronoiMesh("meshes/mesh$(i).nc")
        println("Mesh loaded: You are using a $(USE_REGULAR_MESH ? "regular" : "irregular") mesh")
        push!(mesh_sizes, maximum(mesh.edges.lengthDual))
        dt =  0.1*minimum(mesh.edges.lengthDual)

        for hodge_type in ["diagonal", "whitney", "lsq_linear", "lsq_quadratic"]
            dt, error_L2, error_Linf = run_simulation(mesh, velocity, dt, T1, hodge_type, i, USE_REGULAR_MESH)
            push!(results[hodge_type]["dt"], dt)
            push!(results[hodge_type]["L2"], error_L2)
            push!(results[hodge_type]["Linf"], error_Linf)
        end
    end

    # Create results directory if it doesn't exist
    mesh_type = USE_REGULAR_MESH ? "regular" : "irregular"
    mkpath("results/advection/constant_field")

    # Prepare data for DataFrame
    for (method, data) in results
        # Create DataFrame for each method
        df = DataFrame(
            dx = mesh_sizes,
            dt = data["dt"],
            L2_Error = data["L2"],
            Linf_Error = data["Linf"],
            L2_Order = [i > 1 ? compute_order(data["L2"][1:i], mesh_sizes[1:i]) : 0.0 for i in 1:length(mesh_sizes)],
            Linf_Order = [i > 1 ? compute_order(data["Linf"][1:i], mesh_sizes[1:i]) : 0.0 for i in 1:length(mesh_sizes)]
        )
        
        # Print to console
        for i in 1:length(mesh_sizes)
            @printf("Method: %s, Mesh %d: dx = %.2e, dt = %.2e, L2 = %.2e, Linf = %.2e, L2 Order = %.2f, Linf Order = %.2f\n",
                    method, i, mesh_sizes[i], data["dt"][i], data["L2"][i], data["Linf"][i], 
                    df.L2_Order[i], df.Linf_Order[i])
        end
        
        # Save to CSV
        CSV.write("results/advection/constant_field/adv_2form_cte_$(method)_$(mesh_type).csv", df)
    end

    println("Creating figure for Linf error")
    fig = Figure()
        mesh_type_str = USE_REGULAR_MESH ? "Regular" : "Irregular"
        ax = Axis(fig[1,1], xlabel = L"h", ylabel = L"Error $||⋅||_{\infty}$", 
                  title = L"Advection Error $||⋅||_{\infty}$ " * mesh_type_str * " Mesh for constant field.", 
                  yscale = log10, xscale = log2)
    scatterlines!(ax, mesh_sizes, results["diagonal"]["Linf"], label="Diagonal", color = :red, marker = :circle)
    scatterlines!(ax, mesh_sizes, results["whitney"]["Linf"], label="Whitney", color = :blue, marker = :rect)    
    scatterlines!(ax, mesh_sizes, results["lsq_linear"]["Linf"], label="LSQ1", color = :green, marker = :star5)
    scatterlines!(ax, mesh_sizes, results["lsq_quadratic"]["Linf"], label="LSQ2", color = :orange, marker = :star5)
    scatterlines!(ax, [2^(-4), 2^(-5)], [results["lsq_quadratic"]["Linf"][end], results["lsq_quadratic"]["Linf"][end]/4], label="Second order", color = :black)
    axislegend(ax, position = :rb)
    CairoMakie.save("results/advection/constant_field/adv_2form_cte_Linf_$(mesh_type).pdf", fig)
else
    println("TEST WITH REVERSIBLE FIELD")
    # Initialize data structures for results
    mesh_sizes = []
    results = Dict(
        "diagonal" => Dict("dt" => [], "L2" => [], "Linf" => []),
        "whitney" => Dict("dt" => [], "L2" => [], "Linf" => []),
        "lsq_linear" => Dict("dt" => [], "L2" => [], "Linf" => []),
        "lsq_quadratic" => Dict("dt" => [], "L2" => [], "Linf" => [])
    )

    # Define velocity fields
    velocity2 = x -> [sin(2pi*x[1])*cos(2pi*x[2]), -sin(2pi*x[2])*cos(2pi*x[1])]
    reverse_velocity2 = x -> [-sin(2pi*x[1])*cos(2pi*x[2]), sin(2pi*x[2])*cos(2pi*x[1])]
    T1 = 1.0

    # Run simulations for mesh
    for i in 1:7
        println("i = $i ------------------------------------------------------------")
        mesh = USE_REGULAR_MESH ? VoronoiMesh("meshes/mesh$(i)_regular.nc") : VoronoiMesh("meshes/mesh$(i).nc")
        push!(mesh_sizes, maximum(mesh.edges.lengthDual))
        
        for hodge_type in ["diagonal", "whitney", "lsq_linear", "lsq_quadratic"]
            println("Testing $hodge_type")
            dt, error_L2, error_Linf, u_initial, u_half, u_final = run_reversible_simulation(mesh, velocity2, reverse_velocity2, T1, hodge_type, i, USE_REGULAR_MESH)
            push!(results[hodge_type]["dt"], dt)
            push!(results[hodge_type]["L2"], error_L2)
            push!(results[hodge_type]["Linf"], error_Linf)
        end
    end

    # Create results directory
    mesh_type = USE_REGULAR_MESH ? "regular" : "irregular"
    mkpath("results/advection/rotating_field")

    # Prepare data for DataFrame
    for (method, data) in results
        # Create DataFrame for each method
        df = DataFrame(
            dx = mesh_sizes,
            dt = data["dt"],
            L2_Error = data["L2"],
            Linf_Error = data["Linf"],
            L2_Order = [i > 1 ? compute_order(data["L2"][1:i], mesh_sizes[1:i]) : 0.0 for i in 1:length(mesh_sizes)],
            Linf_Order = [i > 1 ? compute_order(data["Linf"][1:i], mesh_sizes[1:i]) : 0.0 for i in 1:length(mesh_sizes)]
        )
        
        # Print to console
        for i in 1:length(mesh_sizes)
            @printf("Method: %s, Mesh %d: dx = %.2e, dt = %.2e, L2 = %.2e, Linf = %.2e, L2 Order = %.2f, Linf Order = %.2f\n",
                    method, i, mesh_sizes[i], data["dt"][i], data["L2"][i], data["Linf"][i], 
                    df.L2_Order[i], df.Linf_Order[i])
        end
        
        # Save to CSV
        CSV.write("results/advection/rotating_field/adv_2form_rotating_$(method)_$(mesh_type).csv", df)
    end

    println("Creating figure for Linf error")
    fig = Figure()
        mesh_type_str = USE_REGULAR_MESH ? "Regular" : "Irregular"
        ax = Axis(fig[1,1], xlabel = L"h", ylabel = L"Error $||⋅||_{\infty}$", 
                  title = L"Advection Error $||⋅||_{\infty}$ " * mesh_type_str * " Mesh for rotating field.", 
                  yscale = log10, xscale = log2)
    scatterlines!(ax, mesh_sizes, results["diagonal"]["Linf"], label="Diagonal", color = :red, marker = :circle)
    scatterlines!(ax, mesh_sizes, results["whitney"]["Linf"], label="Whitney", color = :blue, marker = :rect)    
    scatterlines!(ax, mesh_sizes, results["lsq_linear"]["Linf"], label="LSQ1", color = :green, marker = :star5)
    scatterlines!(ax, mesh_sizes, results["lsq_quadratic"]["Linf"], label="LSQ2", color = :orange, marker = :star5)
    scatterlines!(ax, [2^(-4), 2^(-5)], [results["lsq_quadratic"]["Linf"][end], results["lsq_quadratic"]["Linf"][end]/4], label="Second order", color = :black)
    axislegend(ax, position = :rb)
    CairoMakie.save("results/advection/rotating_field/adv_2form_rotating_Linf_$(mesh_type).pdf", fig)
end

