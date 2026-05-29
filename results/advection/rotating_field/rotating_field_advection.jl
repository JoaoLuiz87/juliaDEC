using VoronoiMeshes,TensorsLite, TensorsLiteGeometry, ImmutableVectors, LinearAlgebra, NCDatasets, MPASTools
using StaticArrays, SparseArrays
using CairoMakie, CSV, JLD2
include("../../../juliaDEC.jl")
@load "../../../lsq_matrices.jld2" matrices_regular matrices_irregular

mesh = VoronoiMesh("../../../meshes/mesh7.nc")

function rk4_step(f, u, t, dt)
    k1 = f(u, t)
    k2 = f(u + 0.5*dt*k1, t + 0.5*dt)
    k3 = f(u + 0.5*dt*k2, t + 0.5*dt)
    k4 = f(u + dt*k3, t + dt)
    return u + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
end

struct AdvectionProblem #we solve du/dt + iᵥ(du) = 0
    mesh::VoronoiMesh
    velocity::Function  # Vector field function
    rho::Float64
    initial_condition::Function
    dt::Float64
    T::Float64
end

function initial_condition1(x)
    xp = mesh.x_period
    yp = mesh.y_period
    center = [0.5, 0.5]
    sigma = 0.1

    x_vec = TensorsLite.Vec(x = x[1], y = x[2])
    center_vec = TensorsLite.Vec(x = center[1], y = center[2])
    x_raw = closest(center_vec, x_vec, xp, yp)
    y = [x_raw.x, x_raw.y]
    return exp(-(norm(y - center))^2/(2*sigma^2))
end

function solution(x)
    xp = mesh.x_period
    yp = mesh.y_period
    center = [0.5, 0.5]
    sigma = 0.05

    x_vec = TensorsLite.Vec(x = x[1], y = x[2])
    center_vec = TensorsLite.Vec(x = center[1], y = center[2])
    x_raw = closest(center_vec, x_vec, xp, yp)
    y = [x_raw.x, x_raw.y]
    return exp(-(norm(y - center))^2/(2*sigma^2))
end

velocity2 = x -> [sin(2pi*x[1])*cos(2pi*x[2]), -sin(2pi*x[2])*cos(2pi*x[1])] # Non-constant field
reverse_velocity2 = x -> [-sin(2pi*x[1])*cos(2pi*x[2]), sin(2pi*x[2])*cos(2pi*x[1])]

dx = minimum(mesh.edges.lengthDual)
dt2 = (0.1/pi)*dx
rho = 1.0
T2 = 0.4

p2 = AdvectionProblem(mesh, velocity2, rho, initial_condition1, dt2, T2)
p2_rev = AdvectionProblem(mesh, reverse_velocity2, rho, initial_condition1, dt2, T2)


u_initial_dual_2form = Cochain(initial_condition1.(mesh.cells.position) .* mesh.cells.area, 2, false);

println("length of mesh = ", maximum(mesh.edges.lengthDual))

# Define problem
problem = p2

# Initialize solution
global u = Cochain(u_initial_dual_2form.values, 2, false)
global t = 0.0

# Precompute operators for diagonal hodge
v = discretize_vector_field_dual(problem.mesh, problem.velocity)
d1 = d1_matrix_dual(problem.mesh)
star1 = star(problem.mesh, 1, true) 
star2 = star(problem.mesh, 2, false)
wedge10 = wedge_product_10_matrix(problem.mesh, v)
steps = Int(floor(problem.T / problem.dt))

# Define the right-hand side function for RK4
function rhs(u_values, t)
    # Compute -iᵥ(du) = (-1)^(k(2-k)) *(*du ∧ v)
    return -d1 * star1 * wedge10 * star2 * u_values
end

# Run simulation with diagonal hodge
for n in 1:steps
    global u, t
    u = Cochain(rk4_step(rhs, u.values, t, problem.dt), 2, false)
    t += problem.dt
    if n % 100 == 0
        println("t = $t, max value = $(maximum(u.values)),   min value = $(minimum(u.values))")
    end
end

# Plot and save results for diagonal hodge
fig2 = Figure(size=(800, 600))
ax2 = Axis(fig2[1, 1], title = "Solution after $(problem.T) seconds (Diagonal Hodge)")
fig2,ax2 = plot_cochain(fig2, ax2, u, problem.mesh, :rainbow)
CairoMakie.save("../../../results/advection/rotating_field/rotating_field_advection_diagonal.pdf", fig2)

# Reset for lsq2 hodge
global u = Cochain(u_initial_dual_2form.values, 2, false)
global t = 0.0

# Precompute operators for lsq2 hodge
star1 = matrices_irregular[7]["lsq_quadratic"]

# Run simulation with lsq2 hodge
for n in 1:steps
    global u, t
    u = Cochain(rk4_step(rhs, u.values, t, problem.dt), 2, false)
    t += problem.dt
    if n % 100 == 0
        println("t = $t, max value = $(maximum(u.values)),   min value = $(minimum(u.values))")
    end
end

# Plot and save results for lsq2 hodge
fig3 = Figure(size=(800, 600))
ax3 = Axis(fig3[1, 1], title = "Solution after $(problem.T) seconds (LSQ2 Hodge)")
fig3,ax3 = plot_cochain(fig3, ax3, u, problem.mesh, :rainbow)
CairoMakie.save("../../../results/advection/rotating_field/rotating_field_advection_lsq2.pdf", fig3)




