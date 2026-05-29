using VoronoiMeshes, TensorsLite, TensorsLiteGeometry, ImmutableVectors, LinearAlgebra, NCDatasets, MPASTools
using StaticArrays, SparseArrays
using CairoMakie, Printf, DataFrames, CSV, JLD2
include("juliaDEC.jl")

# Load pre-computed matrices
@load "lsq_matrices.jld2" matrices_regular matrices_irregular

struct AdvectionProblem
    mesh::VoronoiMesh
    velocity::Function
    initial_condition::Function
    dt::Float64
    T::Float64
end

function rk4_step(f, u, t, dt)
    k1 = f(u, t)
    k2 = f(u + 0.5*dt*k1, t + 0.5*dt)
    k3 = f(u + 0.5*dt*k2, t + 0.5*dt)
    k4 = f(u + dt*k3, t + dt)
    return u + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
end

function initial_condition(x)
    xp = 1.0
    yp = 1.0
    center = [0.4, 0.5]
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
    center = [0.4, 0.5]
    sigma = 0.05

    x_vec = TensorsLite.Vec(x = x[1], y = x[2])
    center_vec = TensorsLite.Vec(x = center[1], y = center[2])
    x_raw = closest(center_vec, x_vec, xp, yp)
    y = [x_raw.x, x_raw.y]
    return exp(-(norm(y - center))^2/(2*sigma^2))
end

function run_simulation(mesh::VoronoiMesh, velocity, dt, T1, hodge_type, mesh_level)
    problem = AdvectionProblem(mesh, velocity, initial_condition, dt, T1)
    steps = Int(floor(problem.T / problem.dt))
    d1 = d1_matrix_dual(problem.mesh)
    v = discretize_vector_field_dual(problem.mesh, problem.velocity)
    u = Cochain(initial_condition.(mesh.cells.centroid) .* mesh.cells.area, 2, false)
    
    wedge10 = wedge_product_10_matrix(problem.mesh, v)
    
    # Choose Hodge star operator based on type
    if hodge_type == "diagonal"
        star1 = star(problem.mesh, 1, true)
    elseif hodge_type == "whitney"
        star1 = whitney_hodge_star_matrix(problem.mesh)
    elseif hodge_type == "lsq_linear"
        star1 = matrices_irregular[mesh_level]["lsq_linear"]
    elseif hodge_type == "lsq_quadratic"
        star1 = matrices_irregular[mesh_level]["lsq_quadratic"]
    end
    
    star2 = star(problem.mesh, 2, false)
    
    function rhs(u_values, t)
        return - d1 * star1 * wedge10 * star2 * u_values
    end
    
    t = 0.0
    for n in 1:steps
        println("Step $n of $steps, t = $t")
        u = Cochain(rk4_step(rhs, u.values, t, problem.dt), 2, false)
        t += problem.dt
    end
    
    error_Linf = maximum(abs.(u.values - solution.(mesh.cells.position) .* mesh.cells.area))
    error_L2 = norm(u.values - solution.(mesh.cells.position) .* mesh.cells.area)
    
    return error_L2, error_Linf
end

# Function to compute convergence order
function compute_order(errors, dts)
    if length(errors) < 2
        return 0.0
    end
    return log(errors[end-1]/errors[end])/log(dts[end-1]/dts[end])
end

# Main simulation parameters
T1 = 1.0
velocity = x -> [1.0, 1.0]
mesh_level = 6
mesh = VoronoiMesh("meshes/mesh$(mesh_level).nc")
base_dt = 0.5*minimum(mesh.edges.lengthDual)/sqrt(2)

# Define different dt values to test
dt_factors = [1.0, 0.5, 0.25, 0.125]
dts = base_dt .* dt_factors

# Initialize results dictionary
results = Dict(
    "diagonal" => Dict("dt" => [], "L2" => [], "Linf" => []),
    "whitney" => Dict("dt" => [], "L2" => [], "Linf" => []),
    "lsq_linear" => Dict("dt" => [], "L2" => [], "Linf" => []),
    "lsq_quadratic" => Dict("dt" => [], "L2" => [], "Linf" => [])
)

# Run simulations for each dt and Hodge operator
for dt in dts
    println("Testing dt = $dt")
    for hodge_type in ["diagonal", "whitney", "lsq_linear", "lsq_quadratic"]
        error_L2, error_Linf = run_simulation(mesh, velocity, dt, T1, hodge_type, mesh_level)
        push!(results[hodge_type]["dt"], dt)
        push!(results[hodge_type]["L2"], error_L2)
        push!(results[hodge_type]["Linf"], error_Linf)
    end
end

# Create results directory
mkpath("results/advection/dt_convergence")

# Prepare data for DataFrame and plotting
for (method, data) in results
    # Create DataFrame for each method
    df = DataFrame(
        dt = data["dt"],
        L2_Error = data["L2"],
        Linf_Error = data["Linf"],
        L2_Order = [i > 1 ? compute_order(data["L2"][1:i], data["dt"][1:i]) : 0.0 for i in 1:length(data["dt"])],
        Linf_Order = [i > 1 ? compute_order(data["Linf"][1:i], data["dt"][1:i]) : 0.0 for i in 1:length(data["dt"])]
    )
    
    # Print to console
    for i in 1:length(data["dt"])
        @printf("Method: %s, dt = %.2e: L2 = %.2e, Linf = %.2e, L2 Order = %.2f, Linf Order = %.2f\n",
                method, data["dt"][i], data["L2"][i], data["Linf"][i], 
                df.L2_Order[i], df.Linf_Order[i])
    end
    
    # Save to CSV
    CSV.write("results/advection/dt_convergence/adv_2form_dt_$(method).csv", df)
end

# Create convergence plot
fig = Figure(size=(800, 1200))

# L∞ error plot
ax1 = Axis(fig[1,1],
          xlabel="Time Step (dt)",
          ylabel="L∞ Error",
          xscale=log2,
          yscale=log10,
          title="Time Step Convergence of Advection Operator (Mesh Level 6) - L∞ Error")

# Plot L∞ errors for each Hodge star type
colors = [:red, :blue, :green, :orange]
for (i, hodge_type) in enumerate(["diagonal", "whitney", "lsq_linear", "lsq_quadratic"])
    scatterlines!(ax1, results[hodge_type]["dt"], results[hodge_type]["Linf"], 
           color=colors[i], 
           label=hodge_type,
           marker=:circle)
end

# Add reference line for fourth-order convergence
ref_dt = [dts[1], dts[end]]
ref_error = [results["lsq_quadratic"]["Linf"][1], results["lsq_quadratic"]["Linf"][1]/16]
scatterlines!(ax1, ref_dt, ref_error, 
       color=:black, 
       label="Fourth order",
       linestyle=:dash)

axislegend(ax1, position = :rb)

# L2 error plot
ax2 = Axis(fig[2,1],
          xlabel="Time Step (dt)",
          ylabel="L2 Error",
          xscale=log2,
          yscale=log10,
          title="Time Step Convergence of Advection Operator (Mesh Level 6) - L2 Error")

# Plot L2 errors for each Hodge star type
for (i, hodge_type) in enumerate(["diagonal", "whitney", "lsq_linear", "lsq_quadratic"])
    scatterlines!(ax2, results[hodge_type]["dt"], results[hodge_type]["L2"], 
           color=colors[i], 
           label=hodge_type,
           marker=:circle)
end

# Add reference line for fourth-order convergence
ref_error_L2 = [results["lsq_quadratic"]["L2"][1], results["lsq_quadratic"]["L2"][1]/16]
scatterlines!(ax2, ref_dt, ref_error_L2, 
       color=:black, 
       label="Fourth order",
       linestyle=:dash)

axislegend(ax2, position = :rb)

# Adjust layout
rowgap!(fig.layout, 50)

# Save plot
CairoMakie.save("results/advection/dt_convergence/adv_2form_dt_convergence.pdf", fig) 