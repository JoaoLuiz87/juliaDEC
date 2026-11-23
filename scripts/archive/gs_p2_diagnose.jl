# ============================================================================
#  gs_p2_diagnose.jl -- why the degree-2 star's coefficients grow under refinement
#
#      julia scripts/archive/gs_p2_diagnose.jl          (writes output/gs_data/p2_diagnose.txt)
#
#  (U) for [P2]^2: min over pairs of triangles sharing an edge of σ12/σ1 of the rows of P_T on S_T ∩ S_T'.
#  Regularisation sweep: degree 2, rho = 0.01, reg ∈ {1e-13 ... 1e-4} (Tikhonov on the multiplier system).
# ============================================================================
using VoronoiMeshes, TensorsLite, TensorsLiteGeometry, SmallCollections, LinearAlgebra, NCDatasets
using StaticArrays, SparseArrays, Printf
Base.similar(bc::Base.Broadcast.Broadcasted, ::Type{Union{}}) = Array{Union{}}(undef, length.(axes(bc)))
const ROOT = joinpath(@__DIR__, "..", ".."); cd(ROOT)
include(joinpath(ROOT, "src", "juliaDEC.jl")); include(joinpath(ROOT, "src", "hodge_stars.jl")); include(joinpath(ROOT, "src", "galerkin_star.jl"))
const OUT = joinpath(@__DIR__, "..", "..", "output", "gs_data"); mkpath(OUT)
lines = String[]
logp(s) = (println(s); flush(stdout); push!(lines, s))
u1(x) = [sin(2pi * x[1]) * cos(2pi * x[2]), cos(2pi * x[1]) * sin(2pi * x[2])]
u2(x) = [sin(2pi * x[1]) * cos(2pi * x[2]), -cos(2pi * x[1]) * sin(2pi * x[2])]

for mn in ("mesh2", "mesh3", "mesh4", "mesh5")
    mesh = VoronoiMesh("meshes/$mn.nc"); E = mesh.edges.n
    s = sum(edge_length(mesh, e, true) for e in 1:E) / E
    cells = [gs_cell(mesh, t, s; lin = false) for t in 1:mesh.vertices.n]
    conds = Float64[]
    for e in 1:E
        t1, t2 = Int.(collect(mesh.edges.vertices[e]))
        C = cells[t1]; common = intersect(C.S, cells[t2].S)
        sv = svdvals(C.P[[findfirst(==(f), C.S) for f in common], :]); push!(conds, sv[12] / sv[1])
    end
    sort!(conds)
    logp(@sprintf("%s: (U) for [P2]^2: smallest σ12/σ1 over edges = %.2e, %.2e, %.2e (1st, 10th, 100th smallest); fraction below 1e-3: %.3f",
                  mn, conds[1], conds[min(10, end)], conds[min(100, end)], count(<(1e-3), conds) / length(conds)))
end

logp("\nregularisation sweep, degree 2, rho = 0.01")
logp(@sprintf("%-6s %8s %8s %8s %6s %7s %7s %9s %9s %9s %9s", "mesh", "reg", "rowP1", "rowP2", "corr", "C_e", "μ", "u1 max", "u1 rms", "u2 max", "u2 rms"))
for mn in ("mesh3", "mesh4"), reg in (1e-13, 1e-10, 1e-8, 1e-6, 1e-4)
    mesh = VoronoiMesh("meshes/$mn.nc"); Hd = spdiagm(edge_lengths(mesh, false) ./ edge_lengths(mesh, true))
    H, info = galerkin_star_matrix(mesh; degree = 2, rho = 0.01, reg = reg, verbose = false)
    a1 = gs_hodge_error(mesh, H, u1); a2 = gs_hodge_error(mesh, H, u2)
    logp(@sprintf("%-6s %8.0e %8.1e %8.2e %6.2f %7.2f %7.4f %9.3e %9.3e %9.3e %9.3e", mn, reg, gs_row_exactness(mesh, H),
                  gs_row_exactness(mesh, H; degree = 2), info.dev, maximum(gs_row_constants(mesh, H)),
                  gs_smallest_eigenvalue(H; D = Hd), a1..., a2...))
end
write(joinpath(OUT, "p2_diagnose.txt"), join(lines, "\n") * "\n")
