# ============================================================================
#  homogeneous_p2_check.jl -- on a replicated (homogeneous) family the skew is O(h^3), so the barrier no longer forbids a
#  bounded symmetric star exact on [P2]^2.  Does the smallest symmetric correction stay bounded, and is it third order?
#
#      julia -t 4 scripts/barrier/homogeneous_p2_check.jl      (writes output/ts_data/homogeneous_p2.txt and .jls)
#
#  Family: M_n = n x n replication of SCVT mesh 2, scaled by 1/n (h = 1/8, 1/16, 1/32, 1/64).  Stars: DEC, H_P1 (symmetric
#  [P1]^2, vertex stencil), the symmetric [P2]^2 correction on the 4-ring, and the row-wise [P2]^2 correction (H_P2).
#  For comparison the same symmetric [P2]^2 construction on the SCVT meshes 2-5 (from p2_symmetric_check.jl).
# ============================================================================
include(joinpath(@__DIR__, "..", "stars", "three_stars.jl"))
empty!(lines)

function lambda_min(mesh, H)
    Dm = Diagonal(1 ./ sqrt.(diag(hdiag(mesh)))); S = sparse(Symmetric(Dm * ((H + H') / 2) * Dm))
    spd(t) = issuccess(cholesky(S - t * I; check = false))
    lo = -1.0; while !spd(lo) && lo > -1e7; lo *= 4; end
    hi = 1.5
    while hi - lo > 1e-3 * max(1.0, abs(lo)); t = (lo + hi) / 2; spd(t) ? (lo = t) : (hi = t); end
    return lo
end

M1 = VoronoiMesh("meshes/mesh2.nc")
xs = [p.x for p in M1.cells.position]; ys = [p.y for p in M1.cells.position]
R = Dict{Any,Any}()
logp("replicated SCVT mesh 2: symmetric and row-wise stars exact on [P2]^2, against DEC and H_P1")
logp("   Λ, λmin (symmetric part, relative to DEC), r_2(d) and r_inf(d)")
stars = (("DEC", m -> hdiag(m)), ("H_P1", m -> symmetric_correction(m, 1, 3)[1]),
         ("sym P2 (4-ring)", m -> symmetric_correction(m, 2, 4)[1]), ("row P2 (H_P2)", m -> rowwise_correction(m, hdiag(m), 2, 3)[1]))
for n in (1, 2, 4, 8)
    M = n == 1 ? M1 : delaunay_mesh(vcat([(xs .+ i) ./ n for i in 0:n-1, j in 0:n-1]...), vcat([(ys .+ j) ./ n for i in 0:n-1, j in 0:n-1]...))
    h = meshsize(M)
    logp(@sprintf("n = %d (h = %.5f, %d cells)", n, h, M.cells.n))
    for (name, build) in stars
        t0 = time(); H = build(M)
        hd = hodge_error(M, H, HFIELDS.d...); hu = hodge_error(M, H, HFIELDS.u2...)
        Λ = bigLambda(M, H); μ = lambda_min(M, H)
        R[(n, name)] = (h = h, Λ = Λ, μ = μ, d2 = hd.l2, dinf = hd.inf, u2 = hu.l2)
        logp(@sprintf("   %-16s Λ %6.2f  λmin %+7.3f  r_2(d) %.3e  r_inf(d) %.3e  r_2(u2) %.3e  (%.0f s)", name, Λ, μ, hd.l2, hd.inf, hu.l2, time() - t0))
    end
end
logp("\norders between consecutive n (r_2(d) / r_inf(d)):")
for (name, _) in stars, (a, b) in ((1, 2), (2, 4), (4, 8))
    o(x, y) = log2(x / y)
    A, B = R[(a, name)], R[(b, name)]
    logp(@sprintf("   %-16s n %d->%d:  %.2f / %.2f   (Λ %.2f -> %.2f)", name, a, b, o(A.d2, B.d2), o(A.dinf, B.dinf), A.Λ, B.Λ))
end
serialize(joinpath(TSDIR, "homogeneous_p2.jls"), R)
write(joinpath(TSDIR, "homogeneous_p2.txt"), join(lines, "\n") * "\n")
