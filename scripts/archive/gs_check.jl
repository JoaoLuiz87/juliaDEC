# ============================================================================
#  gs_check.jl -- verification and convergence study of the Galerkin star (galerkin_star.jl) on the juliaDEC
#                 SCVT meshes, against DEC, the sandwich and the lift of hodge_stars.jl
#
#      julia scripts/archive/gs_check.jl          (writes output/gs_data/check.txt and check_table.txt)
#
#  For every mesh and star:
#    asym      ‖H - Hᵀ‖₁ / ‖H‖₁
#    rowP1     max_e max_ψ |Σ_f H_ef α_f(ψ) - β_e(ψ)| / |ẽ|, ψ the 6 linear basis fields about m_e (exactness)
#    u1, u2    Hodge error per unit dual length |(H α(u) - β(u))_e| / |ẽ|, max and rms, with juliaDEC's reductions
#              u1 = (sin 2πx cos 2πy, cos 2πx sin 2πy) = -∇(cos 2πx cos 2πy)/2π,  u2 = (sin 2πx cos 2πy, -cos 2πx sin 2πy)
#    C_e       row constants Σ_f |H_ef||f| / |ẽ_e| (max)
#    λmin, μ   smallest eigenvalue of H and of Hd^{-1/2} H Hd^{-1/2} (symmetric stars)
#    Poisson   solve_poisson_primal(mesh, H, f), p = cos 2πx cos 2πy, f = 8π² p:
#              nodal max error (relative), gradient error ‖d0 p_h - α(∇p)‖ in the |ẽ|/|e|-weighted norm (relative),
#              flux error max_e |(H d0 p_h - β(∇p))_e| / |ẽ| relative to max_e |β_e(∇p)|/|ẽ|
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
pex(x) = cos(2pi * x[1]) * cos(2pi * x[2])
gpex(x) = -2pi .* u1(x)
fsrc(x) = 8pi^2 * pex(x)

function poisson_errors(mesh, H)
    ph = solve_poisson_primal(mesh, H, fsrc)
    pI = [pex(_pt(x)) for x in mesh.cells.position]; pI .-= sum(pI) / length(pI)
    d0 = sparse(Float64.(d0_matrix_primal(mesh)))
    ag = reduce_primal_edges(mesh, gpex).values; bg = reduce_dual_edges(mesh, star_field(gpex)).values
    lp = edge_lengths(mesh, true); ld = edge_lengths(mesh, false); w = ld ./ lp
    eg = sqrt(sum(w .* (d0 * ph - ag) .^ 2) / sum(w .* ag .^ 2))
    ef = maximum(abs.(H * d0 * ph - bg) ./ ld) / maximum(abs.(bg) ./ ld)
    maximum(abs.(ph - pI)) / maximum(abs.(pI)), eg, ef
end

logp(@sprintf("%-6s %6s %-9s %8s %8s %9s %9s %9s %9s %6s %7s %7s %9s %9s %9s", "mesh", "E", "star", "asym", "rowP1",
              "u1 max", "u1 rms", "u2 max", "u2 rms", "C_e", "λmin", "μ", "p nodal", "grad", "flux"))
for mn in ("mesh2", "mesh3", "mesh4", "mesh5")
    mesh = VoronoiMesh("meshes/$mn.nc")
    E = mesh.edges.n; h = sqrt(mesh.x_period * mesh.y_period / mesh.cells.n)
    Hd = spdiagm(edge_lengths(mesh, false) ./ edge_lengths(mesh, true))
    t0 = time(); HG, info = galerkin_star_matrix(mesh; verbose = false); tG = time() - t0
    logp(@sprintf("%s: Galerkin build %.1f s (cells %.1f, system %.1f, assembly %.1f); stencil %.1f edges, %.1f nnz/row; moment residual %.1e (least squares %.1e); obstruction %.1e; max correction %.2f; dual-orientation errors %d",
                  mn, tG, info.time.cells, info.time.system, info.time.assembly, info.stencil_mean, info.nnz_per_row,
                  info.moment_residual, info.moment_residual_lsq, info.obstruction, info.dev, info.dual_orientation_errors))
    stars = (("DEC", Hd), ("sandwich", hodge_star_sandwich_matrix(mesh; only_linear = true)),
             ("lift", hodge_star_lift_matrix(mesh; only_linear = true)), ("Galerkin", HG))
    for (lbl, H) in stars
        H = sparse(H)
        asym = norm(H - H', 1) / norm(H, 1)
        rp = gs_row_exactness(mesh, H)
        a1 = gs_hodge_error(mesh, H, u1); a2 = gs_hodge_error(mesh, H, u2)
        C = maximum(gs_row_constants(mesh, H))
        λm, μ = asym < 1e-12 ? (gs_smallest_eigenvalue(H), gs_smallest_eigenvalue(H; D = Hd)) : (NaN, NaN)
        pe = poisson_errors(mesh, H)
        s = @sprintf("%-6s %6d %-9s %8.1e %8.1e %9.2e %9.2e %9.2e %9.2e %6.2f %7.4f %7.4f %9.2e %9.2e %9.2e", mn, E, lbl, asym, rp,
                     a1..., a2..., C, λm, μ, pe...)
        logp(s); push!(table, @sprintf("%s %.6f %s %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e", mn, h, lbl, asym, rp, a1..., a2..., C, λm, μ, pe...))
    end
end
logp("\nobserved orders between consecutive meshes (h halves): u1 max, u1 rms, u2 max, u2 rms, p nodal, grad, flux")
T = [split(l) for l in table]
for lbl in ("DEC", "sandwich", "lift", "Galerkin")
    rs = [parse.(Float64, r[4:end]) for r in T if r[3] == lbl]
    rate(k) = join([@sprintf("%5.2f", log2(rs[i][k] / rs[i+1][k])) for i in 1:length(rs)-1], " ")
    logp(@sprintf("  %-9s | %s | %s | %s | %s | %s | %s | %s", lbl, rate(3), rate(4), rate(5), rate(6), rate(10), rate(11), rate(12)))
end
write(joinpath(OUT, "check.txt"), join(lines, "\n") * "\n")
write(joinpath(OUT, "check_table.txt"), "# mesh h star asym rowP1 u1max u1rms u2max u2rms Cmax lmin mu pnodal grad flux\n" * join(table, "\n") * "\n")
