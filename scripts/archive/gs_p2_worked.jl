# ============================================================================
#  gs_p2_worked.jl -- worked example for the degree-1 and degree-2 stars: the 12 moments (constant, linear,
#                     quadratic test fields) of one reconstruction function Φ_e against those of its Voronoi edge ẽ
#
#      julia scripts/archive/gs_p2_worked.jl          (writes output/gs_data/p2_worked.txt)
#
#  Same mesh and edge as gs_figures.jl (SCVT mesh 3, edge 545).  Also the size of the degree-2 obstruction
#  V = Σ_e ∫_ẽ∫_e |x - y|² = Σ_e |e||ẽ| ((|e|² + |ẽ|²)/12 + δ_e²).
# ============================================================================
using VoronoiMeshes, TensorsLite, TensorsLiteGeometry, SmallCollections, LinearAlgebra, NCDatasets
using StaticArrays, SparseArrays, Printf
Base.similar(bc::Base.Broadcast.Broadcasted, ::Type{Union{}}) = Array{Union{}}(undef, length.(axes(bc)))
const ROOT = joinpath(@__DIR__, "..", ".."); cd(ROOT)
include(joinpath(ROOT, "src", "juliaDEC.jl")); include(joinpath(ROOT, "src", "hodge_stars.jl")); include(joinpath(ROOT, "src", "galerkin_star.jl"))
const OUT = joinpath(@__DIR__, "..", "..", "output", "gs_data"); mkpath(OUT)
lines = String[]
logp(s) = (println(s); flush(stdout); push!(lines, s))

mesh = VoronoiMesh("meshes/mesh3.nc"); E = mesh.edges.n; Ωarea = mesh.x_period * mesh.y_period
s = sum(edge_length(mesh, e, true) for e in 1:E) / E; e0 = 545
β = gs_edge(mesh, e0, s; nt = 12).β

"the 12 moments Σ_{T: e0 ∈ S_T} ∫_T φ̃_{T,e0}·ψ_i of the reconstruction function of e0, for reconstructions Rs on cells of degree `lin`"
function moments(Rs, lin)
    m = zeros(12)
    for t in 1:mesh.vertices.n
        C = gs_cell(mesh, t, s; lin = lin, nt = 12); q = findfirst(==(e0), C.S)
        q === nothing && continue
        m .+= C.Nm[q]' * Rs[t][:, q]
    end
    m
end
_, i1 = galerkin_star_matrix(mesh; keep = true, verbose = false)
_, i2 = galerkin_star_matrix(mesh; degree = 2, keep = true, verbose = false)
_, i3 = galerkin_star_matrix(mesh; degree = 2, rho = 0.01, reg = 1e-8, keep = true, verbose = false)
cols = (("least squares, P1", moments([C.R0 for C in i1.data.cells], true)),
        ("degree 1", moments(i1.data.R, true)),
        ("least squares, P2", moments([gs_cell(mesh, t, s; lin = false).R0 for t in 1:mesh.vertices.n], false)),
        ("degree 2, rho = Inf", moments(i2.data.R, false)),
        ("degree 2, rho = 0.01", moments(i3.data.R, false)))
names = ["(1,0)", "(0,1)", "(X,0)", "(Y,0)", "(0,X)", "(0,Y)", "(X²,0)", "(XY,0)", "(Y²,0)", "(0,X²)", "(0,XY)", "(0,Y²)"]
logp(@sprintf("SCVT mesh 3, edge %d (|e| = %.5f, |ẽ| = %.5f), s = %.5f; moments ∫Φ_e·ψ against ∫_ẽ ψ·t_e", e0,
              edge_length(mesh, e0, true), edge_length(mesh, e0, false), s))
logp(@sprintf("%-8s %11s %s", "ψ", "dual edge", join([@sprintf("%21s", c[1]) for c in cols])))
for i in 1:12
    logp(@sprintf("%-8s %+11.6f %s", names[i], β[i], join([@sprintf("%21s", @sprintf("%+.6f", c[2][i])) for c in cols])))
end
for (lbl, v) in cols
    logp(@sprintf("   %-22s relative mismatch: constant+linear %.2e, quadratic %.2e", lbl, norm(v[1:6] - β[1:6]) / norm(β[1:6]),
                  norm(v[7:12] - β[7:12]) / norm(β[7:12])))
end
# the degree-2 obstruction
V = 0.0; wsum = 0.0
for e in 1:E
    d = gs_edge(mesh, e, s); t = (d.p2 - d.p1) / d.le; n = [-t[2], t[1]]
    δ = dot((d.v1 + d.v2) / 2 - d.m, n)
    global V += d.le * d.lde * ((d.le^2 + d.lde^2) / 12 + δ^2); global wsum += d.le * d.lde
end
h = sqrt(Ωarea / mesh.cells.n)
logp(@sprintf("degree-2 obstruction: Σ_e ∫_ẽ∫_e |x-y|² = %.4e (required: 0); per unit area %.4e = %.3f h²; Σ|e||ẽ| = %.6f (2|Ω| = %.6f)",
              V, V / Ωarea, V / Ωarea / h^2, wsum, 2Ωarea))
write(joinpath(OUT, "p2_worked.txt"), join(lines, "\n") * "\n")
