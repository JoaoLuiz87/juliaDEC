using VoronoiMeshes,TensorsLite, TensorsLiteGeometry, ImmutableVectors, LinearAlgebra, NCDatasets, MPASTools
using StaticArrays, SparseArrays
using CairoMakie 

include("juliaDEC.jl")

error_Linf_diagonal = []
error_Linf_whitney = []
error_Linf_lsq1= []
error_Linf_lsq2= []
error_l2_diagonal = []
error_l2_whitney = []
error_l2_lsq1 = []
error_l2_lsq2 = []
order_diagonal_Linf = []
order_whitney_Linf = []
order_lsq1_Linf = []
order_lsq2_Linf = []
order_diagonal_l2 = []
order_whitney_l2 = []
order_lsq1_l2 = []
order_lsq2_l2 = []
edge_length_list = []


for i = 1:7
    mesh = VoronoiMesh("meshes/mesh" * string(i) * "_regular.nc")
    xp = mesh.x_period
    yp = mesh.y_period
    println("Mesh ", i,"     vertices: ", mesh.cells.n,"    edges: ", mesh.edges.n, "   cells: ", mesh.vertices.n)
    A_diagonal = star(mesh, 1, true)
    A_whitney = whitney_hodge_star_matrix(mesh)
    A_lsq_linear = hodge_star_lsq_matrix(mesh, true, true)
    A_lsq_quadratic = hodge_star_lsq_matrix(mesh, true, false)

    field = x-> [sin(8*pi*x[1]/xp) * cos(8*pi*x[2]/yp), cos(8*pi*x[1]/xp) * sin(8*pi*x[2]/yp)]
    hodge_field = x -> [-cos(8*pi*x[1]/xp) * sin(8*pi*x[2]/yp), sin(8*pi*x[1]/xp) * cos(8*pi*x[2]/yp)]
    v = discretize_vector_field_dual(mesh, field)
    v_hodge = discretize_vector_field_primal(mesh, hodge_field)

    v_diagonal = A_diagonal * v.values
    v_whitney = A_whitney * v.values
    v_lsq_linear = A_lsq_linear * v.values
    v_lsq_quadratic = A_lsq_quadratic * v.values

    push!(error_Linf_diagonal, maximum(abs.(v_diagonal - v_hodge.values)))
    push!(error_Linf_whitney, maximum(abs.(v_whitney - v_hodge.values)))
    push!(error_Linf_lsq1, maximum(abs.(v_lsq_linear - v_hodge.values)))
    push!(error_Linf_lsq2, maximum(abs.(v_lsq_quadratic - v_hodge.values)))
    push!(error_l2_diagonal, L2_errorOnEdges(mesh, v_diagonal, v_hodge.values)) 
    push!(error_l2_whitney, L2_errorOnEdges(mesh, v_whitney, v_hodge.values))
    push!(error_l2_lsq1, L2_errorOnEdges(mesh, v_lsq_linear, v_hodge.values))
    push!(error_l2_lsq2, L2_errorOnEdges(mesh, v_lsq_quadratic, v_hodge.values))
    push!(edge_length_list, maximum(sqrt.(mesh.vertices.area)))
end

for i = 2:7
    push!(order_diagonal_Linf, log(error_Linf_diagonal[i]/error_Linf_diagonal[i-1]) / log(edge_length_list[i]/edge_length_list[i-1]))
    push!(order_whitney_Linf, log(error_Linf_whitney[i]/error_Linf_whitney[i-1]) / log(edge_length_list[i]/edge_length_list[i-1]))
    push!(order_lsq1_Linf, log(error_Linf_lsq1[i]/error_Linf_lsq1[i-1]) / log(edge_length_list[i]/edge_length_list[i-1]))
    push!(order_lsq2_Linf, log(error_Linf_lsq2[i]/error_Linf_lsq2[i-1]) / log(edge_length_list[i]/edge_length_list[i-1]))
    push!(order_diagonal_l2, log(error_l2_diagonal[i]/error_l2_diagonal[i-1]) / log(edge_length_list[i]/edge_length_list[i-1]))
    push!(order_whitney_l2, log(error_l2_whitney[i]/error_l2_whitney[i-1]) / log(edge_length_list[i]/edge_length_list[i-1]))
    push!(order_lsq1_l2, log(error_l2_lsq1[i]/error_l2_lsq1[i-1]) / log(edge_length_list[i]/edge_length_list[i-1]))
    push!(order_lsq2_l2, log(error_l2_lsq2[i]/error_l2_lsq2[i-1]) / log(edge_length_list[i]/edge_length_list[i-1]))
end

fig = Figure()
ax = Axis(fig[1,1], xlabel = L"h", ylabel = L"Error $||⋅||_{\infty}$", title = L"Hodge Star Error $||⋅||_{\infty}$ Regular Mesh", yscale = log10, xscale = log2)
scatterlines!(ax, edge_length_list, error_Linf_diagonal, label="Diagonal", color = :red, marker = :circle)
scatterlines!(ax, edge_length_list, error_Linf_whitney, label="Whitney", color = :blue, marker = :hexagon)
scatterlines!(ax, edge_length_list, error_Linf_lsq1, label="LSQ1", color = :green, marker = :diamond)
scatterlines!(ax, edge_length_list, error_Linf_lsq2, label="LSQ2", color = :orange, marker = :star5)
axislegend(ax, position = :rb)
CairoMakie.save("results/hodges/hodge_star_error_Linf_regular.pdf", fig)    
fig = Figure()
ax = Axis(fig[1,1], xlabel = L"h", ylabel = L"Error $||⋅||_2$", title = L"Hodge Star Error $||⋅||_2$ Regular Mesh", yscale = log10, xscale = log2)
scatterlines!(ax, edge_length_list, error_l2_diagonal, label="Diagonal", color = :red, marker = :circle)
scatterlines!(ax, edge_length_list, error_l2_whitney, label="Whitney", color = :blue, marker = :hexagon)
scatterlines!(ax, edge_length_list, error_l2_lsq1, label="LSQ1", color = :green, marker = :diamond)
scatterlines!(ax, edge_length_list, error_l2_lsq2, label="LSQ2", color = :orange, marker = :star5)
axislegend(ax, position = :rb)
CairoMakie.save("results/hodges/hodge_star_error_L2_regular.pdf", fig)

println("TESTE COM MALHA REGULAR")
println("Order of Linf error for diagonal Hodge star: ", order_diagonal_Linf[end])
println("Order of Linf error for whitney Hodge star: ", order_whitney_Linf[end])
println("Order of Linf error for lsq1 Hodge star: ", order_lsq1_Linf[end])
println("Order of Linf error for lsq2 Hodge star: ", order_lsq2_Linf[end])
println("Order of l2 error for diagonal Hodge star: ", order_diagonal_l2[end])
println("Order of l2 error for whitney Hodge star: ", order_whitney_l2[end])
println("Order of l2 error for lsq1 Hodge star: ", order_lsq1_l2[end])
println("Order of l2 error for lsq2 Hodge star: ", order_lsq2_l2[end])

error_Linf_diagonal = []
error_Linf_whitney = []
error_Linf_lsq1= []
error_Linf_lsq2= []
error_l2_diagonal = []
error_l2_whitney = []
error_l2_lsq1 = []
error_l2_lsq2 = []
order_diagonal_Linf = []
order_whitney_Linf = []
order_lsq1_Linf = []
order_lsq2_Linf = []
order_diagonal_l2 = []
order_whitney_l2 = []
order_lsq1_l2 = []
order_lsq2_l2 = []
edge_length_list = []


for i = 1:7
    mesh = VoronoiMesh("meshes/mesh" * string(i) * ".nc")
    xp = mesh.x_period
    yp = mesh.y_period
    println("Mesh ", i,"     vertices: ", mesh.cells.n,"    edges: ", mesh.edges.n, "   cells: ", mesh.vertices.n)
    A_diagonal = star(mesh, 1, true)
    A_whitney = whitney_hodge_star_matrix(mesh)
    A_lsq_linear = hodge_star_lsq_matrix(mesh, true, true)
    A_lsq_quadratic = hodge_star_lsq_matrix(mesh, true, false)

    field = x-> [sin(8*pi*x[1]/xp) * cos(8*pi*x[2]/yp), cos(8*pi*x[1]/xp) * sin(8*pi*x[2]/yp)]
    hodge_field = x -> [-cos(8*pi*x[1]/xp) * sin(8*pi*x[2]/yp), sin(8*pi*x[1]/xp) * cos(8*pi*x[2]/yp)]
    v = discretize_vector_field_dual(mesh, field)
    v_hodge = discretize_vector_field_primal(mesh, hodge_field)

    v_diagonal = A_diagonal * v.values
    v_whitney = A_whitney * v.values
    v_lsq_linear = A_lsq_linear * v.values
    v_lsq_quadratic = A_lsq_quadratic * v.values

    push!(error_Linf_diagonal, maximum(abs.(v_diagonal - v_hodge.values)))
    push!(error_Linf_whitney, maximum(abs.(v_whitney - v_hodge.values)))
    push!(error_Linf_lsq1, maximum(abs.(v_lsq_linear - v_hodge.values)))
    push!(error_Linf_lsq2, maximum(abs.(v_lsq_quadratic - v_hodge.values)))
    push!(error_l2_diagonal, L2_errorOnEdges(mesh, v_diagonal, v_hodge.values)) 
    push!(error_l2_whitney, L2_errorOnEdges(mesh, v_whitney, v_hodge.values))
    push!(error_l2_lsq1, L2_errorOnEdges(mesh, v_lsq_linear, v_hodge.values))
    push!(error_l2_lsq2, L2_errorOnEdges(mesh, v_lsq_quadratic, v_hodge.values))
    push!(edge_length_list, maximum(sqrt.(mesh.vertices.area)))
end
for i = 2:7
    push!(order_diagonal_Linf, log(error_Linf_diagonal[i]/error_Linf_diagonal[i-1]) / log(edge_length_list[i]/edge_length_list[i-1]))
    push!(order_whitney_Linf, log(error_Linf_whitney[i]/error_Linf_whitney[i-1]) / log(edge_length_list[i]/edge_length_list[i-1]))
    push!(order_lsq1_Linf, log(error_Linf_lsq1[i]/error_Linf_lsq1[i-1]) / log(edge_length_list[i]/edge_length_list[i-1]))
    push!(order_lsq2_Linf, log(error_Linf_lsq2[i]/error_Linf_lsq2[i-1]) / log(edge_length_list[i]/edge_length_list[i-1]))
    push!(order_diagonal_l2, log(error_l2_diagonal[i]/error_l2_diagonal[i-1]) / log(edge_length_list[i]/edge_length_list[i-1]))
    push!(order_whitney_l2, log(error_l2_whitney[i]/error_l2_whitney[i-1]) / log(edge_length_list[i]/edge_length_list[i-1]))
    push!(order_lsq1_l2, log(error_l2_lsq1[i]/error_l2_lsq1[i-1]) / log(edge_length_list[i]/edge_length_list[i-1]))
    push!(order_lsq2_l2, log(error_l2_lsq2[i]/error_l2_lsq2[i-1]) / log(edge_length_list[i]/edge_length_list[i-1]))
end
fig = Figure()
ax = Axis(fig[1,1], xlabel = L"h", ylabel = L"Error $||⋅||_{\infty}$", title = L"Hodge Star Error $||⋅||_{\infty}$ Irregular Mesh", yscale = log10, xscale = log2)
scatterlines!(ax, edge_length_list, error_Linf_diagonal, label="Diagonal", color = :red, marker = :circle)
scatterlines!(ax, edge_length_list, error_Linf_whitney, label="Whitney", color = :blue, marker = :hexagon)
scatterlines!(ax, edge_length_list, error_Linf_lsq1, label="LSQ1", color = :green, marker = :diamond)
scatterlines!(ax, edge_length_list, error_Linf_lsq2, label="LSQ2", color = :orange, marker = :star5)
axislegend(ax, position = :rb)
CairoMakie.save("results/hodges/hodge_star_error_Linf_irregular.pdf", fig)
fig = Figure()
ax = Axis(fig[1,1], xlabel = L"h", ylabel = L"Error $||⋅||_2$", title = L"Hodge Star Error $||⋅||_2$ Irregular Mesh", yscale = log10, xscale = log2)
scatterlines!(ax, edge_length_list, error_l2_diagonal, label="Diagonal", color = :red, marker = :circle)
scatterlines!(ax, edge_length_list, error_l2_whitney, label="Whitney", color = :blue, marker = :hexagon)
scatterlines!(ax, edge_length_list, error_l2_lsq1, label="LSQ1", color = :green, marker = :diamond)
scatterlines!(ax, edge_length_list, error_l2_lsq2, label="LSQ2", color = :orange, marker = :star5)
axislegend(ax, position = :rb)
CairoMakie.save("results/hodges/hodge_star_error_L2_irregular.pdf", fig)
println("TESTE COM MALHA IRREGULAR")
println("Order of Linf error for diagonal Hodge star: ", order_diagonal_Linf[end])
println("Order of Linf error for whitney Hodge star: ", order_whitney_Linf[end])
println("Order of Linf error for lsq1 Hodge star: ", order_lsq1_Linf[end])
println("Order of Linf error for lsq2 Hodge star: ", order_lsq2_Linf[end])
println("Order of l2 error for diagonal Hodge star: ", order_diagonal_l2[end])
println("Order of l2 error for whitney Hodge star: ", order_whitney_l2[end])
println("Order of l2 error for lsq1 Hodge star: ", order_lsq1_l2[end])
println("Order of l2 error for lsq2 Hodge star: ", order_lsq2_l2[end])



