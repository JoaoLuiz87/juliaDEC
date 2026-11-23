# ============================================================================
#  skew_fourier_check.jl -- an exact formula for the skew of plane waves, and what Hypothesis S_2 means for the mesh
#
#      julia -t 4 scripts/barrier/skew_fourier_check.jl        ->  output/ts_data/skew_fourier.{txt,jls}
#
#  For u = a e^{ik·x}, v = b e^{ik'·x} (bilinear extension of s_h) on ANY periodic Delaunay-Voronoi mesh, with m_e the
#  midpoint of e, m̃_e = m_e + δ_e n_e that of ẽ, S_e(k) = sinc(|e| k·t_e/2), S̃_e(k) = sinc(|ẽ| k·n_e/2):
#
#     s_h(u,v) = Σ_e |e||ẽ| (a·t_e)(b·t_e) e^{i(k+k')·m_e} [ S_e(k) S̃_e(k') e^{iδ_e k'·n_e} − S_e(k') S̃_e(k) e^{iδ_e k·n_e} ]   (exact)
#
#  Expansion, κ = k + k':   s_h = M(κ) + N(κ) + O(h^3), with
#     M = i Σ_e |e||ẽ| δ_e (a·t)(b·t) ((k'−k)·n) e^{iκ·m_e}                                      "offset" term (Fourier coefficient of the
#                                                                                              measure Σ |e||ẽ|δ n⊗t⊗t δ_{m_e})
#     N = −Σ_e |e||ẽ| (a·t)(b·t) e^{iκ·m_e} { [|e|²((k·t)²−(k'·t)²) + |ẽ|²((k'·n)²−(k·n)²)]/24 + δ²((k'·n)²−(k·n)²)/2 }   "shape" term
#  For κ = 0 (same wave vector, u = a cos k·x, v = b sin k·x) the phases drop out and s_h = Σ |e||ẽ|(a·t)(b·t) S S̃ sin(δ k·n) = O(h^3)
#  on every mesh.  So the h^2 skew lives in pairs with κ ≠ 0 and is the Fourier coefficient at κ of two measures built from the
#  mesh alone: Hypothesis S_2 is a statement about the large-scale inhomogeneity of the mesh, not about the fields.
#
#  Part 1: the exact formula against direct quadrature of the cochains (real pairs u = a cos k·x, v = b sin k'·x).
#  Part 2: s_h/h^2 and its two parts M/h^2, N/h^2, and the remainder (s_h − M − N)/h^3, on the SCVT meshes 2-7.
#  Part 3: the same on the 2x2 ... replications of mesh3 (a homogeneous family, also centroidal): both parts vanish.
# ============================================================================
isdefined(Main, :cached_stars) || include(joinpath(@__DIR__, "..", "stars", "stars_cache.jl"))
using Printf, LinearAlgebra, Serialization
const OUT = String[]
logp(s) = (println(s); flush(stdout); push!(OUT, s))
usinc(x) = x == 0 ? 1.0 : sin(x) / x
const AV = [1.0, 0.5]; const BV = [0.3, 1.0]
const PAIRS = (("k=(1,0) k'=(0,1)", TP .* [1, 0], TP .* [0, 1]), ("k=(1,0) k'=(1,1)", TP .* [1, 0], TP .* [1, 1]),
               ("k=(1,0) k'=(2,0)", TP .* [1, 0], TP .* [2, 0]), ("k=(1,1) k'=(1,-1)", TP .* [1, 1], TP .* [1, -1]),
               ("k=(1,0) k'=(1,0)", TP .* [1, 0], TP .* [1, 0]))

"Edge geometry: midpoint m, tangent t, normal n (direction of ẽ), |e|, |ẽ|, offset δ."
function edge_geometry(mesh)
    n = mesh.edges.n; G = Vector{NamedTuple}(undef, n)
    for e in 1:n
        p1, p2 = edge_endpoints(mesh, e, true); q1, q2 = edge_endpoints(mesh, e, false)
        le = norm(p2 - p1); ld = norm(q2 - q1); t = (p2 - p1) / le; nn = (q2 - q1) / ld; m = (p1 + p2) / 2; mt = (q1 + q2) / 2
        G[e] = (m = m, t = t, n = nn, le = le, ld = ld, δ = dot(mt - m, nn), tang = dot(mt - m, t))
    end
    return G
end
"(exact, M, N) for the complex pair a e^{ik·x}, b e^{ik'·x}."
function skew_plane(G, a, b, k, kp)
    ex = 0.0im; M = 0.0im; N = 0.0im; κ = k .+ kp
    for g in G
        w = g.le * g.ld * dot(a, g.t) * dot(b, g.t) * cis(dot(κ, g.m))
        S(q) = usinc(g.le * dot(q, g.t) / 2); St(q) = usinc(g.ld * dot(q, g.n) / 2)
        ex += w * (S(k) * St(kp) * cis(g.δ * dot(kp, g.n)) - S(kp) * St(k) * cis(g.δ * dot(k, g.n)))
        M += im * w * g.δ * dot(kp .- k, g.n)
        N -= w * ((g.le^2 * (dot(k, g.t)^2 - dot(kp, g.t)^2) + g.ld^2 * (dot(kp, g.n)^2 - dot(k, g.n)^2)) / 24 + g.δ^2 * (dot(kp, g.n)^2 - dot(k, g.n)^2) / 2)
    end
    return ex, M, N
end
"The real pair u = a cos k·x, v = b sin k'·x from the four complex pairs:  (1/4i)[s(k,k') − s(k,−k') + s(−k,k') − s(−k,−k')]."
function skew_real(G, a, b, k, kp)
    c = (skew_plane(G, a, b, k, kp) .- skew_plane(G, a, b, k, -kp) .+ skew_plane(G, a, b, -k, kp) .- skew_plane(G, a, b, -k, -kp)) ./ 4im
    return real.(c)
end
"Direct evaluation with the reduction maps (edge quadrature)."
function skew_direct(mesh, u, v)
    Ru = reduce_primal_edges(mesh, u).values; Rv = reduce_primal_edges(mesh, v).values
    Su = reduce_dual_edges(mesh, star_field(u)).values; Sv = reduce_dual_edges(mesh, star_field(v)).values
    return dot(Ru, Sv) - dot(Rv, Su)
end

R = Dict{Any,Any}()
logp("Part 1/2: SCVT meshes.  u = a cos(k·x), v = b sin(k'·x), a = (1, 0.5), b = (0.3, 1); k in units of 2π")
for mn in ("mesh2", "mesh3", "mesh4", "mesh5", "mesh6", "mesh7")
    mesh = VoronoiMesh("meshes/$mn.nc"); G = edge_geometry(mesh); h = sqrt(1 / mesh.cells.n)
    logp(@sprintf("\n%s (h = 1/%d);  max tangential offset of ẽ from m_e (should be 0): %.1e", mn, round(Int, 1 / h), maximum(abs(g.tang) for g in G)))
    for (name, k, kp) in PAIRS
        ex, M, N = skew_real(G, AV, BV, k, kp)
        s = @sprintf("   %-18s s_h/h² %+.4f   offset part M/h² %+.4f   shape part N/h² %+.4f   (s_h−M−N)/h³ %+.3f", name, ex / h^2, M / h^2, N / h^2, (ex - M - N) / h^3)
        if mn in ("mesh2", "mesh3", "mesh4", "mesh5")
            d = skew_direct(mesh, x -> AV .* cos(dot(k, x)), x -> BV .* sin(dot(kp, x)))
            s *= @sprintf("   | direct quadrature: %+.6e, formula %+.6e, difference %.1e", d, ex, abs(d - ex))
        end
        k == kp && (s *= @sprintf("   [same k: s_h/h³ = %+.4f]", ex / h^3))
        R[(mn, name)] = (s = ex, M = M, N = N, h = h); logp(s)
    end
end
logp("\nPart 3: replications of mesh3 (homogeneous family; each is again a centroidal Voronoi mesh)")
let base = VoronoiMesh("meshes/mesh3.nc"), G1 = edge_geometry(base), h1 = sqrt(1 / base.cells.n)
    for nrep in (1, 2, 4, 8)
        G = [(m = (g.m .+ [i, j]) ./ nrep, t = g.t, n = g.n, le = g.le / nrep, ld = g.ld / nrep, δ = g.δ / nrep, tang = 0.0) for g in G1 for i in 0:nrep-1 for j in 0:nrep-1]
        h = h1 / nrep
        for (name, k, kp) in PAIRS[[1, 2, 5]]
            ex, M, N = skew_real(G, AV, BV, k, kp)
            logp(@sprintf("   n = %d (h = 1/%d)  %-18s s_h/h² %+.2e   M/h² %+.2e   N/h² %+.2e   s_h/h³ %+.4f", nrep, round(Int, 1 / h), name, ex / h^2, M / h^2, N / h^2, ex / h^3))
            R[(:rep, nrep, name)] = (s = ex, M = M, N = N, h = h)
        end
    end
end
serialize(joinpath(@__DIR__, "..", "..", "output", "ts_data", "skew_fourier.jls"), R)
write(joinpath(@__DIR__, "..", "..", "output", "ts_data", "skew_fourier.txt"), join(OUT, "\n") * "\n")
