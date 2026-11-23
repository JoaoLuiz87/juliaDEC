# ============================================================================
#  es_recipe_check.jl -- every number of report/energy_star_recipe.tex: the hand example, the checks of each step on the
#  SCVT meshes 2-5, the comparison of the two penalties, and the agreement with galerkin_star_matrix.
#
#      julia scripts/archive/es_recipe_check.jl          (writes output/gs_data/recipe.txt)
# ============================================================================
using VoronoiMeshes, TensorsLite, TensorsLiteGeometry, SmallCollections, LinearAlgebra, NCDatasets
using StaticArrays, SparseArrays, Printf
Base.similar(bc::Base.Broadcast.Broadcasted, ::Type{Union{}}) = Array{Union{}}(undef, length.(axes(bc)))
const ROOT = joinpath(@__DIR__, "..", ".."); cd(ROOT)
include(joinpath(ROOT, "src", "juliaDEC.jl")); include(joinpath(ROOT, "src", "hodge_stars.jl"))
include(joinpath(ROOT, "src", "galerkin_star.jl")); include(joinpath(ROOT, "energy_star_recipe.jl"))
const OUT = joinpath(@__DIR__, "..", "..", "output", "gs_data"); mkpath(OUT)
lines = String[]
logp(s) = (println(s); flush(stdout); push!(lines, s))
u1(x) = [sin(2pi * x[1]) * cos(2pi * x[2]), cos(2pi * x[1]) * sin(2pi * x[2])]

# ---------------------------------------------------------------- the hand example
logp("== hand example: T1 = (0,0),(4,0),(1,3), T2 = (0,0),(4,0),(2,-3), e = (0,0) -> (4,0), s = 1")
pts = [[0.0, 0.0], [4.0, 0.0], [1.0, 3.0]]; A = _tri_area(pts...); xb = sum(pts) / 3
J = (A / 12) * sum((p - xb) * (p - xb)' for p in pts)
T1 = (pts = pts, A = A, xb = xb, J = J)
logp(@sprintf("|T1| = %g, x̄ = (%g, %g), J = [%g %g; %g %g]", A, xb..., J...))
mids3 = [(pts[1] + pts[2]) / 2, (pts[2] + pts[3]) / 2, (pts[3] + pts[1]) / 2]
Jq = (A / 3) * sum((mq - xb) * (mq - xb)' for mq in mids3)
logp(@sprintf("J by the 3-point edge-midpoint rule: [%g %g; %g %g]", Jq...))
circ(a, b, c) = begin                                                 # circumcentre
    d = 2 * (a[1] * (b[2] - c[2]) + b[1] * (c[2] - a[2]) + c[1] * (a[2] - b[2]))
    [(norm(a)^2 * (b[2] - c[2]) + norm(b)^2 * (c[2] - a[2]) + norm(c)^2 * (a[2] - b[2])) / d,
     (norm(a)^2 * (c[1] - b[1]) + norm(b)^2 * (a[1] - c[1]) + norm(c)^2 * (b[1] - a[1])) / d]
end
c1 = circ(pts...); c2 = circ([0.0, 0.0], [4.0, 0.0], [2.0, -3.0])
logp(@sprintf("circumcentres c(T1) = (%g, %g), c(T2) = (%g, %g)", c1..., c2...))
me = [2.0, 0.0]; t = [1.0, 0.0]; n = _ROT90 * t
ed = (t = t, n = n, lde = norm(c1 - c2), δ = dot((c1 + c2) / 2 - me, n))
logp(@sprintf("|ẽ| = %g, δ = %g, target τ_e = %s", ed.lde, ed.δ, string(round.(es_target(ed, 1.0), digits = 6))))
L = es_moments(T1, me, 1.0)
logp("L_{T1,e} ="); for i in 1:6; logp("   " * join([@sprintf("%8.4f", L[i, j]) for j in 1:6], " ")); end
X, Y = me - xb
logp("row of P_T1 for e: " * string(round.(4 .* [t[1], t[2], t[1] * X, t[1] * Y, t[2] * X, t[2] * Y], digits = 6)))
# check L against a degree-5 quadrature for a random piece
b, w = tri_rule(5); r = [0.3, -1.2, 0.7, 0.4, -0.5, 1.1]; mq = zeros(6)
for (bb, ww) in zip(b, w)
    x = bb[1] * pts[1] + bb[2] * pts[2] + bb[3] * pts[3]
    F = basis_at(x, xb, true, 1.0) * r; ψ = basis_at(x, me, true, 1.0)
    mq .+= (ww * A) .* (ψ' * F)
end
logp(@sprintf("L r vs quadrature for a random r: max difference %.1e", maximum(abs.(L * r - mq))))
M = es_mass(T1, 1.0); logp("M_T1 diagonal blocks: |T| = $(M[1,1]), J/s² = $(M[3:4,3:4])")

# ---------------------------------------------------------------- meshes
logp("\n== geometry and identities")
logp(@sprintf("%-6s %6s %8s %10s %10s %10s %10s", "mesh", "E", "orient", "perp", "(A) dev", "(B) rel", "σ6/σ1 min"))
for mn in ("mesh2", "mesh3", "mesh4", "mesh5")
    mesh = VoronoiMesh("meshes/$mn.nc"); E = mesh.edges.n; Ω = mesh.x_period * mesh.y_period
    eds = [es_edge(mesh, e) for e in 1:E]
    Aten = sum(ed.le * ed.lde * ed.t * ed.t' for ed in eds) / Ω
    Bten = zeros(2, 2, 2)
    for ed in eds, i in 1:2, j in 1:2, k in 1:2
        Bten[i, j, k] += ed.le * ed.lde * ed.δ * ed.t[i] * ed.t[j] * ed.n[k]
    end
    Bscale = sum(ed.le * ed.lde * abs(ed.δ) for ed in eds)
    sm = sum(ed.le for ed in eds) / E
    sc = minimum(es_split(es_circulations(mesh, es_triangle(mesh, t), es_stencil(mesh, t), sm)[1])[3] for t in 1:mesh.vertices.n)
    logp(@sprintf("%-6s %6d %8d %10.1e %10.1e %10.1e %10.3f", mn, E, count(ed -> !ed.orient, eds), maximum(ed.perp for ed in eds),
                  opnorm(Aten - I), maximum(abs, Bten) / Bscale, sc))
end

logp("\n== agreement with galerkin_star.jl (mesh3): closed forms vs quadrature, and the assembled star")
mesh = VoronoiMesh("meshes/mesh3.nc"); E = mesh.edges.n
s = sum(edge_length(mesh, e, true) for e in 1:E) / E
dP = 0.0; dM = 0.0; dL = 0.0
for t in 1:mesh.vertices.n
    C = gs_cell(mesh, t, s); T = es_triangle(mesh, t); S = es_stencil(mesh, t)
    P, mids, _ = es_circulations(mesh, T, S, s)
    global dP = max(dP, maximum(abs.(P - C.P)) / maximum(abs.(C.P)))
    global dM = max(dM, maximum(abs.(es_mass(T, s) - C.M)) / maximum(abs.(C.M)))
    global dL = max(dL, maximum(maximum(abs.(es_moments(T, mids[q], s) - C.Nm[q]')) / maximum(abs.(C.Nm[q])) for q in eachindex(S)))
end
logp(@sprintf("max relative difference: P_T %.1e, M_T %.1e, L_{T,e} vs N_{T,e}ᵀ %.1e", dP, dM, dL))
Hg, _ = galerkin_star_matrix(mesh; verbose = false); Hm, _ = energy_star(mesh; penalty = :misfit)
logp(@sprintf("‖H(recipe, misfit penalty) - H(galerkin_star_matrix)‖/‖H‖ = %.1e", norm(nonzeros(Hm - Hg)) / norm(nonzeros(Hg))))
s2 = energy_star(mesh; s = 2s)[1]; s1 = energy_star(mesh)[1]
logp(@sprintf("basis scale: ‖H(2s) - H(s)‖/‖H(s)‖ = %.1e", norm(nonzeros(s2 - s1)) / norm(nonzeros(s1))))

logp("\n== the kernel of G (mesh2): eigenvalues of G / max eigenvalue")
mesh = VoronoiMesh("meshes/mesh2.nc")
_, inf = energy_star(mesh; keep = true); ev = eigvals(Symmetric(Matrix(inf.data.G)))
ev = ev / maximum(ev)
logp(@sprintf("smallest 14: %s", join([@sprintf("%.1e", x) for x in ev[1:14]], " ")))
logp(@sprintf("number below 1e-10: %d; r along the kernel: %.1e", count(<(1e-10), ev),
              norm(eigvecs(Symmetric(Matrix(inf.data.G)))[:, 1:12]' * inf.data.r) / norm(inf.data.r)))

logp("\n== the star on meshes 2-5: simple penalty σ W Wᵀ vs the misfit penalty of galerkin_star.jl")
logp(@sprintf("%-6s %-7s %9s %9s %6s %9s %8s %6s %10s %10s %7s", "mesh", "penalty", "lsq res", "moment", "corr", "asym",
              "rowP1", "C_e", "u1 max", "u1 rms", "λmin"))
errs = Dict(:DEC => Float64[], :simple => Float64[], :misfit => Float64[])
for mn in ("mesh2", "mesh3", "mesh4", "mesh5")
    mesh = VoronoiMesh("meshes/$mn.nc")
    Hd = spdiagm(edge_lengths(mesh, false) ./ edge_lengths(mesh, true))
    a = gs_hodge_error(mesh, Hd, u1); push!(errs[:DEC], a[1])
    logp(@sprintf("%-6s %-7s %9s %9s %6s %9s %8.1e %6.2f %10.3e %10.3e %7.4f", mn, "DEC", "", "", "", "", gs_row_exactness(mesh, Hd),
                  1.0, a..., 1.0))
    for pen in (:simple, :misfit)
        t0 = time(); H, inf = energy_star(mesh; penalty = pen); tb = time() - t0
        a = gs_hodge_error(mesh, H, u1); push!(errs[pen], a[1])
        logp(@sprintf("%-6s %-7s %9.2e %9.1e %6.2f %9.1e %8.1e %6.2f %10.3e %10.3e %7.4f   (%.1f s)", mn, pen, inf.lsq_residual,
                      inf.moment_residual, inf.correction, norm(nonzeros(H - H')), gs_row_exactness(mesh, H),
                      maximum(gs_row_constants(mesh, H)), a..., gs_smallest_eigenvalue(H; D = Hd), tb))
        H = nothing; GC.gc()
    end
end
logp("\norders of the u1 max error")
for k in (:DEC, :simple, :misfit)
    v = errs[k]; logp(@sprintf("  %-7s %s", k, join([@sprintf("%6.2f", log2(v[i] / v[i+1])) for i in 1:3], " ")))
end
write(joinpath(OUT, "recipe.txt"), join(lines, "\n") * "\n")
