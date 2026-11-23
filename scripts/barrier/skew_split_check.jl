# ============================================================================
#  skew_split_check.jl -- where does the skew of the primal-dual pairing live?  (barrier_paper.tex)
#
#      julia scripts/barrier/skew_split_check.jl      (writes output/ts_data/skew_split.txt)
#
#  s_h(u,v) = <R_h u, R~*v> - <R_h v, R~*u> for smooth periodic u, v.  A symmetric star pays for it with its error
#  (Theorem ceiling).  Split the fields into gradients grad p and rotated gradients J grad p (Helmholtz on the torus,
#  constants aside) and measure the three kinds of pairs:
#      grad-grad :  s_h(grad p, grad q)  = sum_v [ q(x_v) ∫_v~ Δp - p(x_v) ∫_v~ Δq ]        (exact, by Stokes)
#      curl-curl :  s_h(J grad p, J grad q)
#      grad-curl :  s_h(grad p, J grad q)
#  on the SCVT meshes 2-6, the lattices 2-6 and the perturbed lattices pert1-6.  Also checks the per-triangle identity
#      3 |T| (xbar_T - c_T) = -1/4 sum_{E in T} |E|^3 nu_E      (nu_E outward unit normal)
#  that makes the [P1]^2 compatibility condition a sum of terms that cancel edge by edge.
# ============================================================================
include(joinpath(@__DIR__, "..", "stars", "three_stars.jl"))
empty!(lines)

const P(x) = sin(TP * x[1]) * sin(TP * x[2]) + 0.5cos(2TP * x[1])
const GP(x) = [TP * cos(TP * x[1]) * sin(TP * x[2]) - TP * sin(2TP * x[1]), TP * sin(TP * x[1]) * cos(TP * x[2])]
const Q(x) = cos(TP * x[1]) * cos(2TP * x[2]) + sin(TP * (x[1] + x[2]))
const GQ(x) = [-TP * sin(TP * x[1]) * cos(2TP * x[2]) + TP * cos(TP * (x[1] + x[2])),
               -2TP * cos(TP * x[1]) * sin(2TP * x[2]) + TP * cos(TP * (x[1] + x[2]))]
rot(f) = x -> (g = f(x); [-g[2], g[1]])                                      # J f

function skew_of(mesh, u, v)
    α(w) = reduce_primal_edges(mesh, w).values; β(w) = reduce_dual_edges(mesh, star_field(w)).values
    return dot(α(u), β(v)) - dot(α(v), β(u))
end

"max_T |3|T| d_T + 1/4 Σ_E |E|^3 ν_E| / (|T| h), and max_T |d_T| / h for scale"
function centroid_identity(mesh)
    tris, _, _ = triangles_ccw(mesh); h = meshsize(mesh); worst = 0.0; dmax = 0.0
    for T in tris
        a, b, c = T.pts; xbar = (a .+ b .+ c) ./ 3; d = xbar .- T.cc
        s = sum(begin τ = q .- p; norm(τ)^2 .* [τ[2], -τ[1]] end for (p, q) in ((a, b), (b, c), (c, a)))   # Σ |E|^3 ν_E (CCW)
        worst = max(worst, norm(3 * T.area .* d .+ s ./ 4) / (T.area * h)); dmax = max(dmax, norm(d) / h)
    end
    return worst, dmax
end

logp("skew of the primal-dual pairing split into gradient and rotated-gradient pairs; s/h^2 in brackets")
logp("   p = sin2πx sin2πy + ½cos4πx,  q = cos2πx cos4πy + sin2π(x+y);  u2, d as in three_stars_report.tex")
R = Dict{Any,Any}()
for (fam, names) in (("SCVT", ["mesh$k" for k in 2:6]), ("lattice", ["mesh$(k)_regular" for k in 2:6]), ("perturbed", ["pert$k" for k in 1:6]))
    logp("\n$fam")
    for mn in names
        mesh = VoronoiMesh("meshes/$mn.nc"); h = meshsize(mesh)                 # relative path: NetCDF fails on the non-ASCII home path
        gg = skew_of(mesh, GP, GQ); cc = skew_of(mesh, rot(GP), rot(GQ)); gc = skew_of(mesh, GP, rot(GQ)); cg = skew_of(mesh, rot(GP), GQ)
        ud = skew_of(mesh, U2, fields.d)
        ci, dm = centroid_identity(mesh)
        R[(fam, mn)] = (h = h, gg = gg, cc = cc, gc = gc, cg = cg, ud = ud)
        logp(@sprintf("   %-15s h = %.5f   grad-grad %+.2e (%+.2e)   curl-curl %+.2e (%+.3f)   grad-curl %+.2e (%+.3f)   curl-grad %+.2e (%+.3f)   (u2,d) %+.2e (%+.3f)   | 3|T|d+¼Σ|E|³ν: %.1e, max|d_T|/h %.2f",
                      mn, h, gg, gg / h^2, cc, cc / h^2, gc, gc / h^2, cg, cg / h^2, ud, ud / h^2, ci, dm))
    end
    ks = [k for k in keys(R) if k[1] == fam]; sort!(ks, by = k -> -R[k].h)
    for i in 2:length(ks)
        a, b = R[ks[i-1]], R[ks[i]]; o(x, y) = log(abs(x) / abs(y)) / log(a.h / b.h)
        logp(@sprintf("   order %-14s grad-grad %5.2f   curl-curl %5.2f   grad-curl %5.2f   curl-grad %5.2f   (u2,d) %5.2f",
                      ks[i][2], o(a.gg, b.gg), o(a.cc, b.cc), o(a.gc, b.gc), o(a.cg, b.cg), o(a.ud, b.ud)))
    end
end
serialize(joinpath(TSDIR, "skew_split.jls"), R)
write(joinpath(TSDIR, "skew_split.txt"), join(lines, "\n") * "\n")
