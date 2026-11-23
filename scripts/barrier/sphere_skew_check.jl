# ============================================================================
#  sphere_skew_check.jl -- the skew of the primal--dual pairing on spherical Delaunay--Voronoi meshes
#
#      julia -t 4 scripts/barrier/sphere_skew_check.jl      ->  output/ts_data/sphere_skew.{txt,jls}
#
#  Meshes of the unit sphere (no package needed):
#    * "ico"  : the icosahedral bisection grid (level L: N = 10·4^L + 2 generators), Delaunay by construction;
#    * "scvt" : the same generators after Lloyd iterations with exact spherical centroids (SCVT, as in MPAS),
#               with Lawson edge flips that keep the triangulation Delaunay (= the convex hull).
#  For a Delaunay edge e from x_a to x_b (great-circle arc) and its Voronoi edge ẽ (arc between the circumcentres of
#  the two triangles of e, on the bisector great circle {y·(x_b − x_a) = 0}):
#      (R u)_e  = ∫_e u·t ds,          (R̃⋆v)_e = ∫_ẽ v·N ds,   N = (x_b − x_a)/|x_b − x_a|  (tangent along ẽ, normal to it)
#      s_h(u,v) = Σ_e (R u)_e (R̃⋆v)_e − (R v)_e (R̃⋆u)_e
#  Fields: surface gradients ∇p, ∇q and rotated gradients J∇p = y × ∇p, J∇q of p = y1 y2 + y3, q = y1 + y2 y3 − ½y3²,
#  and the solid-body rotations ω_z = e_z × y, ω_x = e_x × y.  h = sqrt(4π/N).
#  Also reported: the offsets δ_e (arc between the midpoints of e and ẽ) and the Lloyd residual max|x − centroid|/h.
# ============================================================================
using LinearAlgebra, Printf, Serialization
const SDIR = joinpath(@__DIR__, "..", "..", "output", "ts_data"); mkpath(SDIR)
const V3 = Vector{Float64}

# ---------------------------------------------------------------------------- meshes
function icosahedron()
    φ = (1 + sqrt(5)) / 2
    P = [(-1, φ, 0), (1, φ, 0), (-1, -φ, 0), (1, -φ, 0), (0, -1, φ), (0, 1, φ), (0, -1, -φ), (0, 1, -φ), (φ, 0, -1), (φ, 0, 1), (-φ, 0, -1), (-φ, 0, 1)]
    X = [normalize(Float64[p...]) for p in P]
    F = [(1, 12, 6), (1, 6, 2), (1, 2, 8), (1, 8, 11), (1, 11, 12), (2, 6, 10), (6, 12, 5), (12, 11, 3), (11, 8, 7), (8, 2, 9),
         (4, 10, 5), (4, 5, 3), (4, 3, 7), (4, 7, 9), (4, 9, 10), (5, 10, 6), (3, 5, 12), (7, 3, 11), (9, 7, 8), (10, 9, 2)]
    F = [dot(cross(X[b] - X[a], X[c] - X[a]), X[a] + X[b] + X[c]) > 0 ? (a, b, c) : (a, c, b) for (a, b, c) in F]   # CCW seen from outside
    return X, F
end
function subdivide(X0, F)
    X = copy(X0); mid = Dict{Tuple{Int,Int},Int}()
    function getmid(i, j)
        k = minmax(i, j)
        haskey(mid, k) && return mid[k]
        push!(X, normalize(X[i] + X[j])); mid[k] = length(X); return length(X)
    end
    G = NTuple{3,Int}[]
    for (a, b, c) in F
        ab = getmid(a, b); bc = getmid(b, c); ca = getmid(c, a)
        push!(G, (a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca))
    end
    return X, G
end
function icogrid(L)
    X, F = icosahedron()
    for _ in 1:L; X, F = subdivide(X, F); end
    return X, F
end

"Undirected edges (a<b) with the triangle on the left of a→b (seen from outside) and the one on the right."
function edge_table(F)
    D = Dict{Tuple{Int,Int},Vector{Int}}()
    for (t, (a, b, c)) in enumerate(F), (i, j) in ((a, b), (b, c), (c, a))
        v = get!(D, minmax(i, j), [0, 0]); v[i < j ? 1 : 2] = t
    end
    E = sort!(collect(keys(D)))
    return E, [D[e][1] for e in E], [D[e][2] for e in E]
end
other(T, i, j) = T[1] != i && T[1] != j ? T[1] : T[2] != i && T[2] != j ? T[2] : T[3]

"Lawson flips until every edge is locally Delaunay (on the sphere: the opposite vertex is not above the plane of the triangle)."
function make_delaunay!(X, F; tol = 1e-14)
    nflip = 0
    while true
        E, Lt, Rt = edge_table(F); used = falses(length(F)); flipped = 0
        for (k, (i, j)) in enumerate(E)
            tl, tr = Lt[k], Rt[k]; (used[tl] || used[tr]) && continue
            kk = other(F[tl], i, j); l = other(F[tr], i, j)                    # left (i,j,kk) CCW, right (j,i,l) CCW
            n = cross(X[j] - X[i], X[kk] - X[i])
            if dot(X[l] - X[i], n) > tol * norm(n) * norm(X[j] - X[i])
                F[tl] = (kk, i, l); F[tr] = (l, j, kk); used[tl] = used[tr] = true; flipped += 1
            end
        end
        nflip += flipped; flipped == 0 && return nflip
    end
end

circumcentre(X, T) = normalize(cross(X[T[2]] - X[T[1]], X[T[3]] - X[T[1]]))
"Incident triangles of every generator, in counter-clockwise order (seen from outside)."
function vertex_fans(X, F, C)
    inc = [Int[] for _ in eachindex(X)]
    for (t, T) in enumerate(F), v in T; push!(inc[v], t); end
    for v in eachindex(X)
        x = X[v]; e1 = normalize(cross(abs(x[3]) < 0.9 ? [0.0, 0, 1] : [1.0, 0, 0], x)); e2 = cross(x, e1)
        sort!(inc[v]; by = t -> atan(dot(C[t], e2), dot(C[t], e1)))
    end
    return inc
end
"Exact centroid direction of the spherical polygon with CCW vertices P: ∫x dA = ½ Σ θ_k N_k."
function sph_centroid(P)
    s = zeros(3)
    for k in eachindex(P)
        a = P[k]; b = P[mod1(k + 1, length(P))]; c = cross(a, b); nc = norm(c)
        s .+= atan(nc, dot(a, b)) .* (c ./ nc)
    end
    return normalize(s)
end
function lloyd!(X, F; tol = 1e-9, maxit = 5000, report = 0)
    h = sqrt(4π / length(X)); res = Inf; it = 0; nflip = 0
    C = [circumcentre(X, T) for T in F]; fans = vertex_fans(X, F, C); Y = similar(X)
    while it < maxit
        it += 1
        Threads.@threads for t in eachindex(F); C[t] = circumcentre(X, F[t]); end
        Threads.@threads for v in eachindex(X); Y[v] = sph_centroid(view(C, fans[v])); end
        res = maximum(norm(Y[v] - X[v]) for v in eachindex(X)) / h
        X .= Y
        if it % 25 == 0 || res < tol                       # the topology rarely changes: check it now and then
            nf = make_delaunay!(X, F); nflip += nf
            nf > 0 && (C = [circumcentre(X, T) for T in F]; fans = vertex_fans(X, F, C); res = Inf)
        end
        report > 0 && it % report == 0 && @printf("      it %5d  residual %.2e\n", it, res)
        res < tol && break
    end
    return it, res, nflip
end

# ---------------------------------------------------------------------------- integrals and the skew
const GL6 = ((-0.9324695142031521, 0.1713244923791704), (-0.6612093864662645, 0.3607615730479277), (-0.2386191860831969, 0.4679139345726910),
             (0.2386191860831969, 0.4679139345726910), (0.6612093864662645, 0.3607615730479277), (0.9324695142031521, 0.1713244923791704))
"∫ over the great-circle arc from P to Q of f(y, τ) ds (τ the unit tangent)."
function arc_integral(f, P, Q)
    Θ = atan(norm(cross(P, Q)), dot(P, Q)); w = normalize(Q - dot(P, Q) * P); s = 0.0
    for (ξ, g) in GL6
        θ = Θ * (ξ + 1) / 2; y = cos(θ) * P + sin(θ) * w; τ = -sin(θ) * P + cos(θ) * w
        s += g * f(y, τ)
    end
    return s * Θ / 2
end
sgrad(gf, y) = (g = gf(y); g - dot(g, y) * y)
const ∇p = y -> [y[2], y[1], 1.0]
const ∇q = y -> [1.0, y[3], y[2] - y[3]]
#  P = exp(y1 + y2/2), Q = cos(2y3 + y1): all spherical-harmonic degrees (the low-degree p, q cannot see the mesh, see below)
const ∇P = y -> exp(y[1] + y[2] / 2) .* [1.0, 0.5, 0.0]
const ∇Q = y -> -sin(2y[3] + y[1]) .* [1.0, 0.0, 2.0]
const FIELDS = Dict(
    "grad p" => y -> sgrad(∇p, y), "grad q" => y -> sgrad(∇q, y),
    "J grad p" => y -> cross(y, sgrad(∇p, y)), "J grad q" => y -> cross(y, sgrad(∇q, y)),
    "grad P" => y -> sgrad(∇P, y), "grad Q" => y -> sgrad(∇Q, y),
    "J grad P" => y -> cross(y, sgrad(∇P, y)), "J grad Q" => y -> cross(y, sgrad(∇Q, y)),
    "rot z" => y -> cross([0.0, 0, 1], y), "rot x" => y -> cross([1.0, 0, 0], y))
const PAIRS = (("grad P", "grad Q"), ("J grad P", "J grad Q"), ("grad P", "J grad Q"), ("J grad P", "grad Q"),
               ("grad p", "J grad q"), ("rot z", "rot x"))

function mesh_geometry(X, F)
    E, Lt, Rt = edge_table(F); C = [circumcentre(X, T) for T in F]
    return E, Lt, Rt, C
end
function skews(X, F)
    E, Lt, Rt, C = mesh_geometry(X, F); nE = length(E)
    circ = Dict(k => zeros(nE) for k in keys(FIELDS)); flux = Dict(k => zeros(nE) for k in keys(FIELDS))
    δ = zeros(nE); le = zeros(nE); ld = zeros(nE)
    Threads.@threads for k in 1:nE
        a, b = E[k]; xa, xb = X[a], X[b]; N = normalize(xb - xa); cR, cL = C[Rt[k]], C[Lt[k]]
        for (name, u) in FIELDS
            circ[name][k] = arc_integral((y, τ) -> dot(u(y), τ), xa, xb)
            flux[name][k] = arc_integral((y, τ) -> dot(u(y), N), cR, cL)
        end
        me = normalize(xa + xb); md = normalize(cR + cL)
        δ[k] = atan(norm(cross(me, md)), dot(me, md)); le[k] = atan(norm(cross(xa, xb)), dot(xa, xb)); ld[k] = atan(norm(cross(cR, cL)), dot(cR, cL))
        dot(cross(cR, cL), N) > 0 && (ld[k] = -ld[k])      # a non-Delaunay edge would give a negative dual length
    end
    s = Dict(p => sum(circ[p[1]] .* flux[p[2]] .- circ[p[2]] .* flux[p[1]]) for p in PAIRS)
    return s, (δ = δ, le = le, ld = ld, nE = nE)
end

# ---------------------------------------------------------------------------- run
OUT = String[]; say(s) = (println(s); flush(stdout); push!(OUT, s))
R = Dict{Any,Any}()
LEVELS = 2:parse(Int, get(ENV, "SPH_LMAX", "7"))
say("Skew of the primal-dual pairing on the unit sphere; s_h/h^2 for each pair; h = sqrt(4π/N)")
say("pairs: " * join(["($(p[1]), $(p[2]))" for p in PAIRS], "  "))
PREV = nothing                                  # SCVT of the previous level: bisected to start the next one (as in MPAS)
for kind in ("ico", "scvt"), L in LEVELS
    global PREV
    X, F = icogrid(L); N = length(X); h = sqrt(4π / N)
    it, res, nflip = 0, NaN, 0
    if kind == "scvt"
        if PREV !== nothing && PREV[1] == L - 1
            X, F = subdivide(PREV[2], PREV[3]); @assert length(X) == N
        end
        t = @elapsed (it, res, nflip) = lloyd!(X, F; tol = 1e-9, maxit = 40000)
        say(@sprintf("  [lloyd L=%d: %d iterations, residual %.1e h, %d flips, %.0f s]", L, it, res, nflip, t))
        PREV = (L, copy(X), copy(F))
    end
    s, g = skews(X, F)
    R[(kind, L)] = (N = N, h = h, s = s, it = it, res = res, nflip = nflip, δmax = maximum(abs, g.δ) / h, δmean = sum(abs, g.δ) / g.nE / h,
                    ldmin = minimum(g.ld) / h, sumlld = sum(g.le .* g.ld))
    say(@sprintf("%-4s L=%d N=%6d h=%.4f  max|δ|/h=%.2e mean|δ|/h=%.2e  min|ẽ|/h=%.3f  Σ|e||ẽ|/8π=%.6f  ", kind, L, N, h,
                 R[(kind, L)].δmax, R[(kind, L)].δmean, R[(kind, L)].ldmin, R[(kind, L)].sumlld / (8π)) *
        join([@sprintf("%+.4e", s[p] / h^2) for p in PAIRS], " "))
    serialize(joinpath(SDIR, "sphere_skew.jls"), R)
end
#  Control: the low-degree pairs vanish on the symmetric meshes. Break the icosahedral symmetry by a small random displacement
#  (0.02h, then Delaunay flips, no Lloyd) of the level-4 SCVT and recompute.
let L = 4, X = nothing, F = nothing
    X, F = subdivide(icogrid(3)...); lloyd!(X, F; tol = 1e-9, maxit = 40000); h = sqrt(4π / length(X))
    s0, _ = skews(X, F)
    rng_state = 12345; rnd() = (rng_state = (1103515245 * rng_state + 12345) % 2^31; rng_state / 2^31 - 0.5)   # fixed, portable
    for v in eachindex(X); X[v] = normalize(X[v] + 0.04h .* [rnd(), rnd(), rnd()]); end
    make_delaunay!(X, F); s1, _ = skews(X, F)
    for p in PAIRS
        say(@sprintf("control L=4 %-22s symmetric %+.3e   perturbed 0.02h %+.3e   (s_h/h^2)", "($(p[1]), $(p[2]))", s0[p] / h^2, s1[p] / h^2))
    end
    R[:control] = (s0 = s0, s1 = s1, h = h)
end
serialize(joinpath(SDIR, "sphere_skew.jls"), R)
write(joinpath(SDIR, "sphere_skew.txt"), join(OUT, "\n") * "\n")
println("done")
