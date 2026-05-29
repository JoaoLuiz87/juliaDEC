using VoronoiMeshes,TensorsLite, TensorsLiteGeometry, ImmutableVectors, LinearAlgebra, NCDatasets, MPASTools
using StaticArrays, SparseArrays
using CairoMakie, LowRankApprox
include("juliaDEC.jl");
include("mesh_refinement_map.jl");

error_list = []
h_list = []
vector_field(x) = [sin(2*pi*x[1]) * cos(2*pi*x[2]), -cos(2*pi*x[1]) * sin(2*pi*x[2])]

for i = 1:4
    println("Mesh $i")
    coarse_mesh = VoronoiMesh("meshes/mesh$(i)_regular.nc")
    fine_mesh = VoronoiMesh(vcat(coarse_mesh.cells.position, coarse_mesh.edges.position),
                            coarse_mesh.x_period, coarse_mesh.y_period; max_iter=0)
    n_tris = coarse_mesh.vertices.n
    results = zeros(n_tris)
    # Discretize the field once on the fine mesh
    fine_v = discretize_vector_field_dual(fine_mesh, vector_field)

    for coarse_tri in 1:n_tris
        A, rows_edges, cols_key = build_whitney_matrix_for_coarse_triangle(coarse_mesh, fine_mesh, coarse_tri; degree=2, k=4, nq=3)
        edge_ids = first.(rows_edges)
        edge_orients = last.(rows_edges)

        # Right-hand side and apply rank-revealing solve
        rhs = fine_v.values[edge_ids].*edge_orients
        _, col_keep, A_ind = independent_columns(A; rtol=1e-6)

        # Solve
        coeffs = A_ind \ rhs
        b = A_ind * coeffs

        # Reconstruct the field on the coarse triangle
        approximation = combined_whitney_field(coarse_mesh, coarse_tri, coeffs, cols_key[col_keep], 2)
        p = coarse_mesh.vertices.position[coarse_tri]
        error = maximum(abs.(approximation(p) - vector_field(p)))
        results[coarse_tri] = error
    end
    push!(error_list, maximum(results))
    push!(h_list, maximum(coarse_mesh.edges.lengthDual))
end
println("Error list: ", error_list)
println("H list: ", h_list)
