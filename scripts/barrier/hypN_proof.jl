# ============================================================================
#  hypN_proof.jl -- a computer-assisted proof of Hypothesis N of barrier_hodge.tex (locality theorem)
#
#      julia scripts/barrier/hypN_proof.jl      ->  output/ts_data/hypN_proof.txt
#
#  Hypothesis N: there are a mesh satisfying (A1) and a generator v such that, for every nonzero traceless symmetric Z,
#      S_Z(x_v) = Σ_e |e|^3 |ẽ| n_eᵀ Z n_e      (over the edges whose geometry depends on x_v)
#  has a nonzero derivative with respect to x_v, the connectivity being kept fixed.
#  Since S_Z is linear in Z = α D + β S (D = diag(1,-1), S = [0 1; 1 0]), this holds iff the 2×2 matrix
#      M = [∇S_D ; ∇S_S]
#  is nonsingular.  |e|^3|ẽ| nᵀZn = |e||ẽ| wᵀZw with w = J(x_b − x_a), so S_Z involves only rational functions of the
#  coordinates and the square roots |e|, |ẽ|.
#
#  The mesh: the centred rectangular lattice x_ij = (i/8 + (j mod 2)/16, j/8), i, j = 0..7, on the unit torus, with the
#  generator v = (4,4) moved to (1/2 + 1/100, 1/2 + 3/200).  Everything is exact:
#    * coordinates are rationals; orientation, area and in-circle tests use exact rational arithmetic;
#    * the derivative is computed with forward-mode dual numbers over intervals with rational endpoints; every square
#      root is enclosed by integer square roots, so the computed intervals contain the true values.
# ============================================================================
using Printf
const Q = Rational{BigInt}

# ---------------------------------------------------------------------------- rational intervals
struct RI
    lo::Q
    hi::Q
    RI(lo, hi) = (lo <= hi || error("bad interval"); new(Q(lo), Q(hi)))
end
RI(x) = RI(x, x)
Base.:+(a::RI, b::RI) = RI(a.lo + b.lo, a.hi + b.hi)
Base.:-(a::RI, b::RI) = RI(a.lo - b.hi, a.hi - b.lo)
Base.:-(a::RI) = RI(-a.hi, -a.lo)
function Base.:*(a::RI, b::RI)
    p = (a.lo * b.lo, a.lo * b.hi, a.hi * b.lo, a.hi * b.hi); RI(minimum(p), maximum(p))
end
function Base.inv(b::RI)
    (b.lo > 0 || b.hi < 0) || error("division by an interval containing 0"); RI(1 / b.hi, 1 / b.lo)
end
Base.:/(a::RI, b::RI) = a * inv(b)
const KS = big(10)^40
"Rational enclosure of sqrt on an interval with nonnegative endpoints."
function Base.sqrt(a::RI)
    a.lo >= 0 || error("sqrt of a negative interval")
    lo = Q(isqrt(floor(BigInt, a.lo * KS^2)), KS)                     # isqrt(⌊qK²⌋)/K ≤ √q
    hi = Q(isqrt(ceil(BigInt, a.hi * KS^2)) + 1, KS)                  # (isqrt(⌈qK²⌉)+1)/K ≥ √q
    RI(lo, hi)
end
contains0(a::RI) = a.lo <= 0 <= a.hi
width(a::RI) = a.hi - a.lo
mid(a::RI) = (a.lo + a.hi) / 2

# ---------------------------------------------------------------------------- dual numbers (value, ∂/∂x, ∂/∂y)
struct DN
    v::RI
    g1::RI
    g2::RI
end
DN(c) = DN(RI(c), RI(0), RI(0))
Base.:+(a::DN, b::DN) = DN(a.v + b.v, a.g1 + b.g1, a.g2 + b.g2)
Base.:-(a::DN, b::DN) = DN(a.v - b.v, a.g1 - b.g1, a.g2 - b.g2)
Base.:*(a::DN, b::DN) = DN(a.v * b.v, a.g1 * b.v + a.v * b.g1, a.g2 * b.v + a.v * b.g2)
Base.:/(a::DN, b::DN) = (ib = inv(b.v); DN(a.v * ib, (a.g1 * b.v - a.v * b.g1) * ib * ib, (a.g2 * b.v - a.v * b.g2) * ib * ib))
Base.sqrt(a::DN) = (s = sqrt(a.v); t = inv(RI(2) * s); DN(s, a.g1 * t, a.g2 * t))

# ---------------------------------------------------------------------------- the mesh (exact)
const n = 8
pos(i, j) = (Q(mod(i, n), n) + Q(mod(j, 2), 2n), Q(mod(j, n), n))
const V0 = (4, 4)
const DISP = (Q(1, 100), Q(3, 200))
function point(i, j)
    p = pos(i, j)
    (mod(i, n), mod(j, n)) == V0 ? (p[1] + DISP[1], p[2] + DISP[2]) : p
end
"Triangles of the centred rectangular lattice, as lattice index triples (counter-clockwise), with periodic wrapping."
function lattice_triangles()
    T = NTuple{3,Tuple{Int,Int}}[]
    for j in 0:n-1, i in 0:n-1
        if iseven(j)
            push!(T, ((i, j), (i + 1, j), (i, j + 1)), ((i, j + 1), (i + 1, j), (i + 1, j + 1)))
        else
            push!(T, ((i, j), (i + 1, j), (i + 1, j + 1)), ((i, j), (i + 1, j + 1), (i, j + 1)))
        end
    end
    return T
end
"Coordinates of lattice point (i,j) lifted near the reference r (nearest periodic image)."
function lifted(i, j, r)
    p = point(i, j)
    (p[1] + round(BigInt, r[1] - p[1]), p[2] + round(BigInt, r[2] - p[2]))
end
wrap(k) = (mod(k[1], n), mod(k[2], n))
orient(a, b, c) = (b[1] - a[1]) * (c[2] - a[2]) - (b[2] - a[2]) * (c[1] - a[1])
"In-circle determinant (> 0 iff d is inside the circumcircle of the CCW triangle abc)."
function incircle(a, b, c, d)
    m = [(p[1] - d[1], p[2] - d[2], (p[1] - d[1])^2 + (p[2] - d[2])^2) for p in (a, b, c)]
    m[1][1] * (m[2][2] * m[3][3] - m[2][3] * m[3][2]) - m[1][2] * (m[2][1] * m[3][3] - m[2][3] * m[3][1]) +
        m[1][3] * (m[2][1] * m[3][2] - m[2][2] * m[3][1])
end

OUT = String[]; say(s) = (println(s); push!(OUT, s))
T = lattice_triangles()
# (a) orientation and area, lifting each triangle about its first vertex
area = Q(0)
for t in T
    r = point(t[1]...); A = lifted(t[1]..., r); B = lifted(t[2]..., r); C = lifted(t[3]..., r)
    o = orient(A, B, C); o > 0 || error("triangle not counter-clockwise: $t")
    global area += o / 2
end
say("triangles: $(length(T)), all counter-clockwise; total area = $(area) (must be 1)")
area == 1 || error("the triangles do not tile the torus")
# (b) every edge in exactly two triangles, with opposite orientations
E = Dict{Any,Vector{Tuple{Int,Int}}}()          # undirected edge -> [(triangle, local index of the opposite vertex)]
for (k, t) in enumerate(T), l in 1:3
    a = wrap(t[l]); b = wrap(t[mod1(l + 1, 3)]); key = a < b ? (a, b) : (b, a)
    push!(get!(E, key, Tuple{Int,Int}[]), (k, mod1(l + 2, 3)))
end
all(length(v) == 2 for v in values(E)) || error("an edge is not shared by exactly two triangles")
say("edges: $(length(E)), each shared by two triangles (Euler: V - E + T = $(n^2 - length(E) + length(T)), torus = 0)")
# (c) strict local Delaunay property of every edge: the opposite vertex is strictly outside the other circumcircle
global minmargin = Inf
for (key, v) in E
    (t1, o1), (t2, o2) = v; r = point(T[t1][1]...)
    A, B, C = (lifted(T[t1][l]..., r) for l in 1:3); D = lifted(T[t2][o2]..., r)
    ic = incircle(A, B, C, D); ic < 0 || error("edge $key is not strictly Delaunay")
    global minmargin = min(minmargin, Float64(-ic))
end
say(@sprintf("every edge is strictly locally Delaunay (smallest margin of the in-circle determinant %.3e): the mesh is the unique Delaunay triangulation, so (A1) holds", minmargin))

# ---------------------------------------------------------------------------- S_Z and its derivative at x_v
xv = point(V0...)
"Dual-number coordinates of a lattice point near x_v (x_v itself carries the derivative)."
function dpt(k)
    p = lifted(k..., xv)
    wrap(k) == V0 ? (DN(RI(p[1]), RI(1), RI(0)), DN(RI(p[2]), RI(0), RI(1))) : (DN(p[1]), DN(p[2]))
end
function circumcentre(a, b, c)
    d = DN(2) * (a[1] * (b[2] - c[2]) + b[1] * (c[2] - a[2]) + c[1] * (a[2] - b[2]))
    s(p) = p[1] * p[1] + p[2] * p[2]
    ((s(a) * (b[2] - c[2]) + s(b) * (c[2] - a[2]) + s(c) * (a[2] - b[2])) / d,
     (s(a) * (c[1] - b[1]) + s(b) * (a[1] - c[1]) + s(c) * (b[1] - a[1])) / d)
end
fan = [k for (k, t) in enumerate(T) if any(wrap(p) == V0 for p in t)]
edges_v = Set{Any}()
for k in fan, l in 1:3
    a = wrap(T[k][l]); b = wrap(T[k][mod1(l + 1, 3)]); push!(edges_v, a < b ? (a, b) : (b, a))
end
say("generator v = $(V0) at $(Float64.(xv)); its fan has $(length(fan)) triangles and $(length(edges_v)) edges")
function SZ(which)
    acc = DN(0)
    for key in edges_v
        (t1, _), (t2, _) = E[key]
        a = dpt(key[1]); b = dpt(key[2])
        c1 = circumcentre((dpt(p) for p in T[t1])...); c2 = circumcentre((dpt(p) for p in T[t2])...)
        le = sqrt((b[1] - a[1]) * (b[1] - a[1]) + (b[2] - a[2]) * (b[2] - a[2]))
        ld = sqrt((c2[1] - c1[1]) * (c2[1] - c1[1]) + (c2[2] - c1[2]) * (c2[2] - c1[2]))
        w1 = DN(0) - (b[2] - a[2]); w2 = b[1] - a[1]                       # w = J(b − a)
        q = which == :D ? w1 * w1 - w2 * w2 : DN(2) * w1 * w2
        acc = acc + le * ld * q
    end
    return acc
end
SD = SZ(:D); SS = SZ(:S)
f(x) = @sprintf("[%.12e, %.12e]", Float64(x.lo), Float64(x.hi))
say("∂S_D/∂x1 ∈ " * f(SD.g1) * "   ∂S_D/∂x2 ∈ " * f(SD.g2))
say("∂S_S/∂x1 ∈ " * f(SS.g1) * "   ∂S_S/∂x2 ∈ " * f(SS.g2))
detM = SD.g1 * SS.g2 - SD.g2 * SS.g1
say("det M ∈ " * f(detM) * @sprintf("   (interval width %.1e)", Float64(width(detM))))
contains0(detM) && error("not proved: the determinant interval contains 0")
say("PROVED: det M ≠ 0, so ∇S_Z ≠ 0 for every nonzero traceless symmetric Z (Hypothesis N holds for this mesh and generator).")

# ---------------------------------------------------------------------------- floating-point cross-check by central differences
function SZfloat(which, x)
    acc = 0.0
    P(k) = (p = Float64.(collect(lifted(k..., xv))); wrap(k) == V0 ? x : p)
    cc(a, b, c) = (d = 2(a[1] * (b[2] - c[2]) + b[1] * (c[2] - a[2]) + c[1] * (a[2] - b[2])); s(p) = p[1]^2 + p[2]^2;
                   [(s(a) * (b[2] - c[2]) + s(b) * (c[2] - a[2]) + s(c) * (a[2] - b[2])) / d, (s(a) * (c[1] - b[1]) + s(b) * (a[1] - c[1]) + s(c) * (b[1] - a[1])) / d])
    for key in edges_v
        (t1, _), (t2, _) = E[key]; a = P(key[1]); b = P(key[2])
        c1 = cc((P(p) for p in T[t1])...); c2 = cc((P(p) for p in T[t2])...)
        w = [-(b[2] - a[2]), b[1] - a[1]]; q = which == :D ? w[1]^2 - w[2]^2 : 2w[1] * w[2]
        acc += sqrt(sum(abs2, b - a)) * sqrt(sum(abs2, c2 - c1)) * q
    end
    acc
end
x0 = Float64.(collect(xv)); ε = 1e-6
fd(which, k) = (dx = zeros(2); dx[k] = ε; (SZfloat(which, x0 + dx) - SZfloat(which, x0 - dx)) / 2ε)
say(@sprintf("central differences: ∂S_D = (%.9e, %.9e), ∂S_S = (%.9e, %.9e)", fd(:D, 1), fd(:D, 2), fd(:S, 1), fd(:S, 2)))
h = 1 / 8
say(@sprintf("in units of h^3 (h = 1/8): ∂S_D = (%.4f, %.4f) h^3, ∂S_S = (%.4f, %.4f) h^3",
             Float64(mid(SD.g1)) / h^3, Float64(mid(SD.g2)) / h^3, Float64(mid(SS.g1)) / h^3, Float64(mid(SS.g2)) / h^3))
mkpath(joinpath(@__DIR__, "..", "..", "output", "ts_data")); write(joinpath(@__DIR__, "..", "..", "output", "ts_data", "hypN_proof.txt"), join(OUT, "\n") * "\n")
