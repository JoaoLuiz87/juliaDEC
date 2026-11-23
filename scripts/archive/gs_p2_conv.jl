# ============================================================================
#  gs_p2_conv.jl -- convergence of DEC, the degree-1 star and the degree-2 star on the SCVT meshes 2-5
#
#      julia scripts/archive/gs_p2_conv.jl          (writes output/gs_data/p2_conv.txt and p2_conv_table.txt)
#
#  degree 2 = [P2]^2 reconstructions, linear moments exact up to the Tikhonov term reg = 1e-8, quadratic moments in
#  least squares with rho = 0.01 (see gs_p2_explore.jl and gs_p2_diagnose.jl for the choice).
# ============================================================================
using VoronoiMeshes, TensorsLite, TensorsLiteGeometry, SmallCollections, LinearAlgebra, NCDatasets
using StaticArrays, SparseArrays, Printf
Base.similar(bc::Base.Broadcast.Broadcasted, ::Type{Union{}}) = Array{Union{}}(undef, length.(axes(bc)))
const ROOT = joinpath(@__DIR__, "..", ".."); cd(ROOT)
include(joinpath(ROOT, "src", "juliaDEC.jl")); include(joinpath(ROOT, "src", "hodge_stars.jl")); include(joinpath(ROOT, "src", "galerkin_star.jl"))
const OUT = joinpath(@__DIR__, "..", "..", "output", "gs_data"); mkpath(OUT)
lines = String[]; table = String[]
logp(s) = (println(s); flush(stdout); push!(lines, s))
u1(x) = [sin(2pi * x[1]) * cos(2pi * x[2]), cos(2pi * x[1]) * sin(2pi * x[2])]
u2(x) = [sin(2pi * x[1]) * cos(2pi * x[2]), -cos(2pi * x[1]) * sin(2pi * x[2])]
pex(x) = cos(2pi * x[1]) * cos(2pi * x[2]); gpex(x) = -2pi .* u1(x); fsrc(x) = 8pi^2 * pex(x)

function poisson_errors(mesh, H)
    ph = solve_poisson_primal(mesh, H, fsrc)
    pI = [pex(_pt(x)) for x in mesh.cells.position]; pI .-= sum(pI) / length(pI)
    d0 = sparse(Float64.(d0_matrix_primal(mesh)))
    bg = reduce_dual_edges(mesh, star_field(gpex)).values; ld = edge_lengths(mesh, false)
    maximum(abs.(ph - pI)) / maximum(abs.(pI)), maximum(abs.(H * d0 * ph - bg) ./ ld) / maximum(abs.(bg) ./ ld)
end

logp(@sprintf("%-6s %-9s %8s %8s %6s %6s %7s %9s %9s %9s %9s %9s %9s %6s %6s", "mesh", "star", "rowP1", "rowP2", "corr", "C_e", "μ",
              "u1 max", "u1 rms", "u2 max", "u2 rms", "p nodal", "flux", "nnz", "time"))
for mn in ("mesh2", "mesh3", "mesh4", "mesh5")
    mesh = VoronoiMesh("meshes/$mn.nc"); E = mesh.edges.n; h = sqrt(mesh.x_period * mesh.y_period / mesh.cells.n)
    Hd = spdiagm(edge_lengths(mesh, false) ./ edge_lengths(mesh, true))
    for lbl in ("DEC", "degree1", "degree2")
        t0 = time()
        H, info = lbl == "DEC" ? (Hd, nothing) :
                  lbl == "degree1" ? galerkin_star_matrix(mesh; verbose = false) :
                  galerkin_star_matrix(mesh; degree = 2, rho = 0.01, reg = 1e-8, verbose = false)
        tb = time() - t0
        rp1 = gs_row_exactness(mesh, H); rp2 = gs_row_exactness(mesh, H; degree = 2)
        a1 = gs_hodge_error(mesh, H, u1); a2 = gs_hodge_error(mesh, H, u2)
        C = maximum(gs_row_constants(mesh, H)); μ = gs_smallest_eigenvalue(H; D = Hd)
        pe = poisson_errors(mesh, H); d = info === nothing ? 0.0 : info.dev
        logp(@sprintf("%-6s %-9s %8.1e %8.2e %6.2f %6.2f %7.4f %9.3e %9.3e %9.3e %9.3e %9.3e %9.3e %6.1f %6.1f", mn, lbl, rp1, rp2, d, C, μ,
                      a1..., a2..., pe..., nnz(H) / E, tb))
        push!(table, @sprintf("%s %.6f %s %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e", mn, h, lbl, rp1, rp2, d, C, μ,
                              a1..., a2..., pe...))
        H = nothing; GC.gc()
    end
end
logp("\nobserved orders (h halves): u1 max, u1 rms, u2 max, u2 rms, p nodal, flux")
T = [split(l) for l in table]
for lbl in ("DEC", "degree1", "degree2")
    rs = [parse.(Float64, r[4:end]) for r in T if r[3] == lbl]
    rate(k) = join([@sprintf("%5.2f", log2(rs[i][k] / rs[i+1][k])) for i in 1:length(rs)-1], " ")
    logp(@sprintf("  %-8s | %s | %s | %s | %s | %s | %s", lbl, rate(6), rate(7), rate(8), rate(9), rate(10), rate(11)))
end
write(joinpath(OUT, "p2_conv.txt"), join(lines, "\n") * "\n")
write(joinpath(OUT, "p2_conv_table.txt"), "# mesh h star rowP1 rowP2 corr Cmax mu u1max u1rms u2max u2rms pnodal flux\n" * join(table, "\n") * "\n")
