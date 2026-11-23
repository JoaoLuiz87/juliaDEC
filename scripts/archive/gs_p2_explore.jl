# ============================================================================
#  gs_p2_explore.jl -- the degree-2 Galerkin star: [P2]^2 reconstructions, linear moments exact, quadratic moments
#                      matched in the least-squares sense with weight 1/rho.  Compared with DEC, the lift and the
#                      degree-1 star on one mesh.
#
#      julia scripts/archive/gs_p2_explore.jl mesh3      (writes output/gs_data/p2_explore_mesh3.txt)
#
#  Columns: rowP1 / rowP2 = max row error on the linear / quadratic test fields (per |ẽ|; rowP2 is the constant of the
#  leading O(h²) error), quad = relative quadratic-moment residual of the reconstruction functions, corr = max
#  correction, C_e = max row constant, μ = λmin(Hd^-1/2 H Hd^-1/2), u1 / u2 = max Hodge error per |ẽ|.
# ============================================================================
using VoronoiMeshes, TensorsLite, TensorsLiteGeometry, SmallCollections, LinearAlgebra, NCDatasets
using StaticArrays, SparseArrays, Printf
Base.similar(bc::Base.Broadcast.Broadcasted, ::Type{Union{}}) = Array{Union{}}(undef, length.(axes(bc)))
const ROOT = joinpath(@__DIR__, "..", ".."); cd(ROOT)
include(joinpath(ROOT, "src", "juliaDEC.jl")); include(joinpath(ROOT, "src", "hodge_stars.jl")); include(joinpath(ROOT, "src", "galerkin_star.jl"))
const OUT = joinpath(@__DIR__, "..", "..", "output", "gs_data"); mkpath(OUT)
mn = isempty(ARGS) ? "mesh3" : ARGS[1]
lines = String[]
logp(s) = (println(s); flush(stdout); push!(lines, s))
u1(x) = [sin(2pi * x[1]) * cos(2pi * x[2]), cos(2pi * x[1]) * sin(2pi * x[2])]
u2(x) = [sin(2pi * x[1]) * cos(2pi * x[2]), -cos(2pi * x[1]) * sin(2pi * x[2])]

mesh = VoronoiMesh("meshes/$mn.nc"); E = mesh.edges.n
s = sum(edge_length(mesh, e, true) for e in 1:E) / E
Hd = spdiagm(edge_lengths(mesh, false) ./ edge_lengths(mesh, true))
c2 = [gs_cell(mesh, t, s; lin = false) for t in 1:mesh.vertices.n]
logp(@sprintf("%s: [P2]^2 cells: min σ12/σ1 of P_T = %.3f, stencil sizes %d..%d", mn,
              minimum(begin sv = svdvals(C.P); sv[12] / sv[1] end for C in c2), minimum(length(C.S) for C in c2), maximum(length(C.S) for C in c2)))
logp(@sprintf("%-22s %8s %8s %9s %6s %6s %7s %9s %9s %9s %9s %6s %6s", "star", "rowP1", "rowP2", "quad", "corr", "C_e", "μ",
              "u1 max", "u1 rms", "u2 max", "u2 rms", "nnz", "time"))
function report(lbl, H, info)
    H = sparse(H)
    rp1 = gs_row_exactness(mesh, H); rp2 = gs_row_exactness(mesh, H; degree = 2)
    a1 = gs_hodge_error(mesh, H, u1); a2 = gs_hodge_error(mesh, H, u2)
    C = maximum(gs_row_constants(mesh, H))
    μ = norm(H - H', 1) / norm(H, 1) < 1e-12 ? gs_smallest_eigenvalue(H; D = Hd) : NaN
    q = info === nothing ? NaN : info.quad_residual; d = info === nothing ? NaN : info.dev
    t = info === nothing ? NaN : sum(info.time)
    logp(@sprintf("%-22s %8.1e %8.2e %9.2e %6.2f %6.2f %7.4f %9.3e %9.3e %9.3e %9.3e %6.1f %6.1f", lbl, rp1, rp2, q, d, C, μ, a1..., a2...,
                  nnz(H) / E, t))
end
report("DEC", Hd, nothing)
report("lift (not symmetric)", hodge_star_lift_matrix(mesh; only_linear = true), nothing)
H1, i1 = galerkin_star_matrix(mesh; verbose = false); report("degree 1", H1, i1)
H2, i2 = galerkin_star_matrix(mesh; degree = 2, verbose = false); report("degree 2, rho = Inf", H2, i2)
for ρ in (10.0, 1.0, 0.1, 0.01, 1e-3, 1e-4)
    H, info = galerkin_star_matrix(mesh; degree = 2, rho = ρ, verbose = false)
    report(@sprintf("degree 2, rho = %g", ρ), H, info)
    ρ == 10.0 && logp(@sprintf("   (least-squares quadratic residual %.2e; obstruction %.1e; linear moment residual %.1e)",
                               info.quad_residual_lsq, info.obstruction, info.moment_residual))
end
write(joinpath(OUT, "p2_explore_$mn.txt"), join(lines, "\n") * "\n")
