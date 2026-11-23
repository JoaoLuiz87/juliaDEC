# ============================================================================
#  window_skew_check.jl -- is the offset skew over a window a boundary term?
#
#      julia scripts/barrier/window_skew_check.jl
#
#  On the periodic meshes of sh_rate_check.jl (pattern / random / mapped, N = 128),
#  take windows W = {edges with midpoint within distance L of the centre} and compute,
#  exactly for affine fields (Theorem 14(c) and its affine-affine version):
#    S1(L) = s_W(a, v),   a = (1, 0.3) constant,   v = G2 (x - x0)       (constant x affine)
#    S2(L) = s_W(u, v),   u = G1 (x - x0),         v = G2 (x - x0)       (affine x affine)
#  A boundary term scales like the perimeter: S1 ∝ L, S2 ∝ L² (|u| ~ L on the boundary).
#  A bulk density would scale like the area: S1 ∝ L², S2 ∝ L³.
#  Also checks the identities Σ_e |e||ẽ|δ_e n_e = 0, Σ_T |T|(x̄_T - c_T) = 0 and
#  Σ_e |e||ẽ|δ_e t⊗t⊗n = 0 over the whole torus.
# ============================================================================
using LinearAlgebra, Printf, Random
include(joinpath(@__DIR__, "sh_rate_mesh.jl"))     # mesh(N, kind), unwrap, circumcentre, J, Ly

function edge_data(X, T)
    edges = Dict{Tuple{Int,Int},Vector{Int}}()
    for (k, t) in enumerate(T), (a, b) in ((t[1], t[2]), (t[2], t[3]), (t[3], t[1]))
        push!(get!(edges, (min(a, b), max(a, b)), Int[]), k)
    end
    E = []
    for ((a, b), ts) in edges
        xa = X[a]; xb = unwrap(X[b], xa); m = (xa + xb) / 2; t = (xb - xa) / norm(xb - xa); n = J * t
        s = [dot(circumcentre([unwrap(X[p], xa) for p in T[k]]...) - m, n) for k in ts]
        push!(E, (m = m, t = t, n = n, L = norm(xb - xa), Lt = abs(s[2] - s[1]), δ = (s[1] + s[2]) / 2))
    end
    E
end
function tri_identity(X, T)
    D = zeros(2)
    for t in T
        v = [unwrap(X[p], X[t[1]]) for p in t]; A = cross2(v[2] - v[1], v[3] - v[1]) / 2
        D += abs(A) * (sum(v) / 3 - circumcentre(v...))
    end
    D
end

G1 = [0.3 1.1; -0.7 0.4]; G2 = [-0.5 0.2; 0.9 0.6]; a = [1.0, 0.3]
for kind in ("pattern", "random", "mapped")
    X, T, h = mesh(128, kind); E = edge_data(X, T)
    sn = sum(e.L * e.Lt * e.δ * e.n for e in E); D = tri_identity(X, T)
    C = zeros(2, 2, 2)
    for e in E, i in 1:2, j in 1:2, k in 1:2
        C[i, j, k] += e.L * e.Lt * e.δ * e.t[i] * e.t[j] * e.n[k]
    end
    scale = sum(e.L * e.Lt * abs(e.δ) for e in E)
    @printf("%-8s N=128: |Σ|e||ẽ|δ n| = %.1e, |Σ|T|(x̄-c)| = %.1e, max|Σ|e||ẽ|δ t⊗t⊗n| = %.1e   (Σ|e||ẽ||δ| = %.1e)\n",
            kind, norm(sn), norm(D), maximum(abs, C), scale)
    x0 = [0.5, Ly / 2]
    println("   L      #edges   S1 = s_W(a,v)    S2 = s_W(u,v)")
    prev = nothing
    for Lw in (0.05, 0.1, 0.2, 0.4)
        W = [e for e in E if norm(e.m - x0) < Lw]
        S1 = sum(-e.L * e.Lt * e.δ * dot(a, e.t) * dot(e.t, G2 * e.n) for e in W)
        S2 = sum(e.L * e.Lt * e.δ * (dot(G2 * (e.m - x0), e.t) * dot(e.t, G1 * e.n) - dot(G1 * (e.m - x0), e.t) * dot(e.t, G2 * e.n)) for e in W)
        r = prev === nothing ? "" : @sprintf("   slopes in L: %.2f, %.2f", log2(abs(S1 / prev[1])), log2(abs(S2 / prev[2])))
        @printf("   %.2f  %7d   %+12.4e   %+12.4e%s\n", Lw, length(W), S1, S2, r)
        prev = (S1, S2)
    end
end
