# ============================================================================
#  p2_stencil_rank.jl -- how many [P_k]^2 fields can a stencil tell apart?  (Section 2.2 of three_stars_report.tex)
#
#      julia scripts/stars/p2_stencil_rank.jl      (writes output/ts_data/stencil_rank.txt)
#
#  Row e can be made exact on [P_k]^2 for every right-hand side only if A_e (circulations of the basis on the stencil)
#  has full column rank N = 6 (k = 1) or 12 (k = 2).  Counting edges is not enough: the circulation of a gradient
#  field grad(phi) along an edge is the difference of phi at its ends, so the stencil sees gradients only through the
#  values of phi at its V vertices (V - 1 numbers modulo constants), and the rest of the field only through the
#  circulations around its E - V + 1 independent cycles.  [P_k]^2 = grad P_{k+1} (+) a complement of dimension
#  dim P_{k-1} (the curl is in P_{k-1}), so
#        rank A_e <= min(V - 1, dim P_{k+1} - 1) + min(E - V + 1, dim P_{k-1}).
#  For k = 2: dim P_3 - 1 = 9 and dim P_1 = 3, so a stencil needs V >= 10 vertices; the 13-edge stencil has 8.
# ============================================================================
include(joinpath(@__DIR__, "three_stars.jl"))         # tri_layers, ring(mesh, e, :t2 | :t3), STENCILS, SNAME
empty!(lines)

dimP(k) = k < 0 ? 0 : (k + 1) * (k + 2) ÷ 2
logp("stencil rank: E edges, V Delaunay vertices, cycles E - V + 1, rank of A_e (tol 1e-10 relative) against the bound")
for mn in ("mesh3", "mesh5")
    mesh = VoronoiMesh("meshes/$mn.nc"); n = mesh.edges.n; lp = edge_lengths(mesh, true)
    logp("$mn  ($n edges)")
    for st in STENCILS
        Es, Vs, rk = Int[], Int[], Dict(1 => Int[], 2 => Int[]); smin = Dict(1 => Inf, 2 => Inf)
        for e in 1:n
            S = ring(mesh, e, st); x0 = _pt(mesh.edges.position[e])
            V = length(unique(vcat([collect(mesh.edges.cells[j]) for j in S]...)))
            push!(Es, length(S)); push!(Vs, V)
            for k in (1, 2)
                A = reduce(hcat, [(q = edge_endpoints(mesh, j, true; about = mesh.edges.position[e]); seg_moments(q[1], q[2], x0, lp[e], k))
                                  for j in S])'
                sv = svdvals(A); push!(rk[k], count(>(1e-10 * sv[1]), sv)); smin[k] = min(smin[k], sv[end] / sv[1])
            end
        end
        bound(k) = minimum(min(V - 1, dimP(k + 1) - 1) + min(E - V + 1, dimP(k - 1)) for (E, V) in zip(Es, Vs))
        logp(@sprintf("   %-28s E %2d..%2d  V %2d..%2d  cycles %2d..%2d  | [P1]^2: rank %d..%d of 6 (bound %d), min σ ratio %.1e | [P2]^2: rank %2d..%2d of 12 (bound %2d), min σ ratio %.1e",
                      SNAME[st], extrema(Es)..., extrema(Vs)..., extrema(Es .- Vs .+ 1)..., extrema(rk[1])..., bound(1), smin[1],
                      extrema(rk[2])..., bound(2), smin[2]))
    end
end
write(joinpath(TSDIR, "stencil_rank.txt"), join(lines, "\n") * "\n")
