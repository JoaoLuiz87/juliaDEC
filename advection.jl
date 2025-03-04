using VoronoiMeshes,TensorsLite, TensorsLiteGeometry, ImmutableVectors, LinearAlgebra, NCDatasets, MPASTools
using StaticArrays, SparseArrays
using CairoMakie
include("juliaDEC.jl")

"""
    rk4_step(f, u, t, dt)

Perform one step of RK4 integration.
f: function that computes the right-hand side
u: current solution
t: current time
dt: time step
"""
function rk4_step(f, u, t, dt)
    k1 = f(u, t)
    k2 = f(u + 0.5*dt*k1, t + 0.5*dt)
    k3 = f(u + 0.5*dt*k2, t + 0.5*dt)
    k4 = f(u + dt*k3, t + dt)
    return u + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
end


"""
    AdvectionProblem

Struct to store the problem data for the advection problem.
"""
struct AdvectionProblem #we solve du/dt + iᵥ(du) = 0
    mesh::VoronoiMesh
    velocity::Function  # Vector field function
    rho::Float64
    initial_condition::Function
    dt::Float64
    T::Float64
end

"""
    solve_advection(problem::AdvectionProblem, spatial_method::String = "FDM", temporal_method::String = "Implicit")

Solve the advection problem using the specified spatial and temporal methods.
"""
function solve_advection(problem::AdvectionProblem, spatial_method::String = "FDM", temporal_method::String = "Implicit")

    if spatial_method == "FDM" && temporal_method == "Implicit"

        u = Cochain(problem.initial_condition.(problem.mesh.cells.position), 0, true)
        v = discretize_vector_field_dual(problem.mesh, problem.velocity)
        d1 = d1_matrix_dual(problem.mesh)
        star1 = star(problem.mesh, 1, true) 
        star0 = star(problem.mesh, 0, true)
        wedge10 = wedge_product_10_matrix(problem.mesh, v)
        M = (problem.rho) * star0 + problem.dt * d1 * star1 * wedge10 
        for t in 0:problem.dt:problem.T
            println("t = $t and solving system")
            b = (problem.rho)* (star0 * u.values)
            u_new = M \ b
            u = Cochain(u_new, 0, true)
            println("Maximum value of u: ", maximum(abs.(u.values)), "   Minimum value of u: ", minimum(abs.(u.values)))
        end

        # Plot final solution
        fig_final = plot_cochain(u, problem.mesh)
        CairoMakie.save("advection_final.png", fig_final)

        return u

    elseif spatial_method == "FDM" && temporal_method == "RK4"
        # Initialize solution
        u = Cochain(problem.initial_condition.(problem.mesh.cells.position), 0, true)
        
        # Plot initial condition
        fig_init = plot_cochain(u, problem.mesh)
        CairoMakie.save("advection_initial.png", fig_init)

        # Precompute operators
        v = discretize_vector_field_dual(problem.mesh, problem.velocity)
        d1 = d1_matrix_dual(problem.mesh)
        star1 = star(problem.mesh, 1, true) 
        star2 = star(problem.mesh, 2, false)
        wedge10 = wedge_product_10_matrix(problem.mesh, v)
        
        # Define the right-hand side function for RK4
        function rhs(u_values, t)
            # Compute -iᵥ(du)
            return - star2 * d1 * star1 * wedge10 * u_values
        end
        t = 0.0
        while t < problem.T
            dt = min(problem.dt, problem.T - t)
            u = Cochain(rk4_step(rhs, u.values, t, dt), 0, true)
            t += dt
            println("t = $t, max value = $(maximum(abs.(u.values))),   min value = $(minimum(abs.(u.values)))")
        end

        # Plot final solution
        fig_final = plot_cochain(u, problem.mesh)
        CairoMakie.save("advection_final.png", fig_final)

        return u

    elseif spatial_method == "FVM" && temporal_method == "Explicit"
        
        
        u = Cochain(problem.initial_condition.(problem.mesh.cells.position) .* problem.mesh.cells.area, 2, false)
        u0 = Cochain(u.values, 2, false)
        v = discretize_vector_field_dual(problem.mesh, problem.velocity)
        
        d1 = d1_matrix_dual(problem.mesh)
        #star1 = star(problem.mesh, 1, true) 
        star1 = hodge_star_lsq_matrix(problem.mesh, true)
        star2 = star(problem.mesh, 2, false)
        wedge10 = wedge_product_10_matrix(problem.mesh, v)
        
        t = 0.0
        k = 0
        while t <= problem.T
            k % 50 == 0 ? println("t = $t, k = $k, Maximum value = ", maximum(u.values), "   Minimum value = ", minimum(u.values)) : nothing
            u_new = u.values - problem.dt * d1 * star1 * wedge10 * star2 * u.values
            u = Cochain(u_new, 2, false)
            t += problem.dt
            k += 1
        end
        return u, u0

    elseif spatial_method == "FVM" && temporal_method == "Implicit"
        
        
        u = Cochain(problem.initial_condition.(problem.mesh.cells.centroid) .* problem.mesh.cells.area, 2, false)
        u0 = Cochain(u.values, 2, false)
        v = discretize_vector_field_dual(problem.mesh, problem.velocity)
        
        d1 = d1_matrix_dual(problem.mesh)
        #tar1 = star(problem.mesh, 1, true) 
        star1 = whitney_hodge_star_matrix(problem.mesh)
        star2 = star(problem.mesh, 2, false)
        wedge10 = wedge_product_10_matrix(problem.mesh, v)
        id = Diagonal(ones(problem.mesh.cells.n))
        M = id + (problem.dt * d1 * star1 * wedge10 * star2)
        t = 0.0
        k=0
        for t in 0:problem.dt:problem.T
            k % 10 == 0 ? println("t = $t, k = $k, Maximum value = ", maximum(u.values), "   Minimum value = ", minimum(u.values)) : nothing
            b =  u.values
            u_new = M \ b
            u = Cochain(u_new, 2, false)
            k += 1
        end
        return u, u0

    elseif spatial_method == "DEC" && temporal_method == "Implicit"

        u = Cochain(problem.initial_condition.(problem.mesh.vertices.position), 0, false)

        # Plot initial condition
        fig_init = plot_cochain(u, problem.mesh)
        CairoMakie.save("advection_initial.png", fig_init)

        v = discretize_vector_field_dual(problem.mesh, problem.velocity)
        
        d0 = d0_matrix_dual(problem.mesh)
        star0 = star(problem.mesh, 0, false)
        star1 = star(problem.mesh, 1, false)
        wedge11 = wedge_product_11_matrix(problem.mesh, v)
        M = (problem.rho) * star0 - problem.dt * wedge11 * star1 * d0
        for t in 0:problem.dt:problem.T
            tprintln("t = $t and solving system")
            b = (problem.rho) * (star0 * u.values)
            u_new = M \ b
            u = Cochain(u_new, 0, false)
            println("Maximum value of u: ", maximum(abs.(u.values)), "   Minimum value of u: ", minimum(abs.(u.values)))
        end

        # Plot final solution
        fig_final = plot_cochain(u, problem.mesh)
        CairoMakie.save("advection_final.png", fig_final)

        return u
    
    elseif spatial_method == "DEC" && temporal_method == "RK4"
        u = Cochain(problem.initial_condition.(problem.mesh.vertices.position), 0, false)
        v = discretize_vector_field_dual(problem.mesh, problem.velocity)
        d0 = d0_matrix_dual(problem.mesh)
        star2 = star(problem.mesh, 2, true)
        star1 = star(problem.mesh, 1, false)
        wedge11 = wedge_product_11_matrix(problem.mesh, v)
        
        # iᵥ(du)
        f(u_values, t) = star2 * wedge11 * star1 * d0 * u_values

        t = 0.0
        while t < problem.T
            dt = min(problem.dt, problem.T - t)
            u = Cochain(rk4_step(f, u.values, t, dt), 0, false)
            t += dt
            println("t = $t, max value = $(maximum(abs.(u.values))),   min value = $(minimum(abs.(u.values)))")
        end

        # Plot final solution
        fig_final = plot_cochain(u, problem.mesh)
        CairoMakie.save("advection_final.png", fig_final)

        return u
    else
        error("Method not supported")
    end

end

"""
    Example usage and test case
"""
function test_advection(spatial_method::String = "FDM", temporal_method::String = "RK4")
    mesh = VoronoiMesh("meshes/mesh3.nc")
    velocity = x -> [1.0, 0.0]  # Rotational field
    rho = 1.0   
    T = 1.0

    # Initial condition (Gaussian)
    center = [0.5, 0.5]
    sigma = 0.1
    initial_condition = x -> exp(-((x[1]-center[1])^2 + (x[2]-center[2])^2)/(2*sigma^2)) 

    #function initial_condition(x)
    #    if 0.4 <= x[1] <= 0.6 && 0.4 <= x[2] <= 0.6
    #        return 1.0
    #    else
    #        return 0.0
    #    end
    #end

    error_list = []
    length_list = []
    for i = 1:6
        mesh = VoronoiMesh("meshes/mesh" * string(i) * "_regular.nc")
        dx = maximum(mesh.edges.lengthDual)
        dt = 0.001
        
        problem = AdvectionProblem(mesh, velocity, rho, initial_condition, dt, T)
        u, u0 = solve_advection(problem, spatial_method, temporal_method)
        push!(error_list, maximum(abs.(u.values - u0.values) ./ mesh.cells.area))
        push!(length_list, dx)

        if i==5
            fig_init = plot_cochain(u0, problem.mesh)
            CairoMakie.save("advection_initial.png", fig_init)

            fig_final = plot_cochain(u, problem.mesh)
            CairoMakie.save("advection_final.png", fig_final)
        end
    end


    for i = 2:length(error_list)
        order = log(error_list[i]/error_list[i-1]) / log(length_list[i]/length_list[i-1])
        println("Order Linf between meshes $(i-1) and $i: $order")
    end
end


test_advection("FVM", "Implicit")