using VoronoiMeshes,TensorsLite, TensorsLiteGeometry, ImmutableVectors, LinearAlgebra, NCDatasets, MPASTools
using StaticArrays, SparseArrays
using CairoMakie 

include("juliaDEC.jl")
println("Use distorted mesh? (y/n)")
use_distorted_mesh = readline()

if use_distorted_mesh == "y"
    n_meshes = 5
else
    n_meshes = 7
end

println("Choose test case:")
println("1: diagonal hodge star operator")
println("2: whitney hodge star operator")
println("3: least squares hodge star operator")
test_case = parse(Int, readline())


if test_case == 1
    error_list_Linf = []
    edge_length_list = []
    for i = 1:n_meshes-1

        if use_distorted_mesh == "y"
            mesh = VoronoiMesh("meshes/mesh" * string(i) * ".nc")
        else
            mesh = VoronoiMesh("meshes/mesh" * string(i) * "_regular.nc")
        end
        println("Testing diagonal hodge star operator")
        xp = mesh.x_period
        yp = mesh.y_period
        A = star(mesh, 1, true)

        field = x-> [cos(2*pi*x[1]/xp) * sin(2*pi*x[2]/yp), sin(2*pi*x[1]/xp) * cos(2*pi*x[2]/yp)]
        hodge_field = x -> [-sin(2*pi*x[1]/xp) * cos(2*pi*x[2]/yp), cos(2*pi*x[1]/xp) * sin(2*pi*x[2]/yp)]
        v = discretize_vector_field_dual(mesh, field)
        v_hodge = discretize_vector_field_primal(mesh, hodge_field)

        #push!(error_list_Linf, maximum(abs.(A * v.values - v_hodge.values)))
        #push!(error_list_Linf, norm((A * v.values - v_hodge.values)))
        #L2 norm
        aux = (A * v.values - v_hodge.values) .^2
        aux = aux .* (mesh.edges.lengthDual/mesh.edges.length)
        push!(error_list_Linf, sqrt(sum(aux)))
        
        push!(edge_length_list, maximum(sqrt.(mesh.vertices.area)))
        println("number of triangles: ", mesh.vertices.n, " number of edges: ", mesh.edges.n, " number of vertices: ", mesh.cells.n)
        println("The hodge matrix is symmetric: ", issymmetric(A))
        println("The hodge matrix is positive definite: ", isposdef(A))
        println("---------")
    end
    println(error_list_Linf)
    println(edge_length_list)

    for i = 2:length(error_list_Linf)
        order = log(error_list_Linf[i]/error_list_Linf[i-1]) / log(edge_length_list[i]/edge_length_list[i-1])
        println("Order Linf between meshes $(i-1) and $i: $order")
    end
end

if test_case == 2
#order of convergence of whitney hodge star operator

    error_list_Linf = []
    edge_length_list = []

    for i = 1:n_meshes
        if use_distorted_mesh == "y"
            mesh = VoronoiMesh("meshes/mesh" * string(i) * ".nc")
        else
            mesh = VoronoiMesh("meshes/mesh" * string(i) * "_regular.nc")
        end
        xp = mesh.x_period
        yp = mesh.y_period
        A = whitney_hodge_star_matrix(mesh)
        
        field = x-> [cos(2*pi*x[1]/xp) * sin(2*pi*x[2]/yp), sin(2*pi*x[1]/xp) * cos(2*pi*x[2]/yp)]
        hodge_field = x -> [-sin(2*pi*x[1]/xp) * cos(2*pi*x[2]/yp), cos(2*pi*x[1]/xp) * sin(2*pi*x[2]/yp)]
        
        v = discretize_vector_field_dual(mesh, field)
        v_hodge = discretize_vector_field_primal(mesh, hodge_field)

        push!(error_list_Linf, maximum(abs.(A * v.values - v_hodge.values)))
        push!(edge_length_list, maximum(mesh.edges.lengthDual))
        println("number of triangles: ", mesh.cells.n, " number of edges: ", mesh.edges.n, " number of vertices: ", mesh.vertices.n)
        println("The hodge matrix is symmetric: ", issymmetric(A))
        println("The hodge matrix is positive definite: ", isposdef(A))
        println("---------")
    end
    println(error_list_Linf)
    println(edge_length_list)

    for i = 2:length(error_list_Linf)
        order = log(error_list_Linf[i]/error_list_Linf[i-1]) / log(edge_length_list[i]/edge_length_list[i-1])
        println("Order Linf between meshes $(i-1) and $i: $order")
    end
end

if test_case == 3
    println("Checking for convergence of the lsq hodge star operator")
    error_list_Linf = []
    edge_length_list = []

    for i = 1:n_meshes-1
        if use_distorted_mesh == "y"
            mesh = VoronoiMesh("meshes/mesh" * string(i) * ".nc")
        else
            mesh = VoronoiMesh("meshes/mesh" * string(i) * "_regular.nc")
        end
        xp = mesh.x_period
        yp = mesh.y_period
        A = hodge_star_lsq_matrix(mesh, true, true)

        field = x-> [cos(2*pi*x[1]/xp) * sin(2*pi*x[2]/yp), sin(2*pi*x[1]/xp) * cos(2*pi*x[2]/yp)]
        hodge_field = x -> [-sin(2*pi*x[1]/xp) * cos(2*pi*x[2]/yp), cos(2*pi*x[1]/xp) * sin(2*pi*x[2]/yp)]
        v = discretize_vector_field_dual(mesh, field)
        v_hodge = discretize_vector_field_primal(mesh, hodge_field)
        #push!(error_list_Linf, maximum(abs.(A * v.values - v_hodge.values)))
        
        #L2 norm
        aux = (A * v.values - v_hodge.values) .^2
        aux = aux .* (mesh.edges.lengthDual/mesh.edges.length)
        push!(error_list_Linf, sqrt(sum(aux)))

        push!(edge_length_list, maximum(sqrt.(mesh.vertices.area)))
        println("number of triangles: ", mesh.vertices.n, " number of edges: ", mesh.edges.n, " number of vertices: ", mesh.cells.n)
        println("The hodge matrix is symmetric: ", issymmetric(A))
        println("The hodge matrix is positive definite: ", isposdef(A))
        println("---------")
    end
    println(error_list_Linf)
    println(edge_length_list)

    for i = 2:length(error_list_Linf)
        order = log(error_list_Linf[i]/error_list_Linf[i-1]) / log(edge_length_list[i]/edge_length_list[i-1])
        println("Order Linf between meshes $(i-1) and $i: $order")
    end
end