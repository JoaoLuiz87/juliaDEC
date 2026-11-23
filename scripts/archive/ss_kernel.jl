# ============================================================================
#  ss_kernel.jl -- the share matrix G: kernel, compatibility, overlap condition, near-kernel under refinement
#
#      julia scripts/archive/ss_kernel.jl        (writes ss_data/kernel.txt)
#
#  For porting lattices n = 10, 20, 40 and SCVT meshes 2-4 (ρ = 2, and ρ = 1.4 where noted):
#   * dim ker G from the dense spectrum, against the prediction 4 (edge fields w_e g(φ_e));
#   * the compatibility ⟨1, λ⟩ = 0 on the kernel (= Theorem 3.12 of the guide);
#   * the overlap condition of the solvability theorem: for neighbouring patches (centre edges sharing a
#     triangle), rank of the skew forms of the common edges (= 7 needed);
#   * the first non-zero eigenvalue λ₅(G) and max|θ| (near-kernel of modulated kernel vectors);
#   * constant anisotropic K: dim ker G_K and the compatibility.
# ============================================================================
include(joinpath(@__DIR__, "..", "..", "SymStar.jl")); using .SymStar
include(joinpath(@__DIR__, "ss_mesh.jl"))
using LinearAlgebra, SparseArrays, Printf
const OUT = joinpath(@__DIR__, "..", "..", "output", "ss_data"); mkpath(OUT)
lines = String[]; logp(s) = (println(s); flush(stdout); push!(lines, s))

function analyse(name, X, T, period, h; rho = 2.0, K = nothing, overlap = true)
    g = edge_geometry(X, T, period); E = length(g.le)
    pats = build_patches(g.me, g.t, g.le, g.lde, g.delta, period, h, rho; K = K)
    Vs = [SymStar.rowspace(skew_map(N, R))[1] for (P, N, R) in pats]
    ranks = [size(V, 2) for V in Vs]
    G = share_matrix(pats, Vs, E)
    F = eigen(Symmetric(Matrix(G))); ev = F.values
    nk = count(<(1e-9 * ev[end]), ev)
    Z = F.vectors[:, 1:nk]
    comp = nk > 0 ? norm(Z' * ones(E)) / sqrt(E) : 0.0                  # component of 1 along ker G
    φ = atan.(g.t[:, 2], g.t[:, 1]); w = g.le .* g.lde .* g.delta
    Kv = hcat(w .* cos.(φ), w .* sin.(φ), w .* cos.(3φ), w .* sin.(3φ))
    inker = norm(Kv - Z * (Z' * Kv)) / norm(Kv)                          # are the predicted vectors in ker G?
    # overlap condition: neighbouring patches = centre edges in a common triangle
    minrank = 15
    if overlap
        eid = Dict(e => i for (i, e) in enumerate(g.edges))
        for tri in T
            es = [eid[(min(a, b), max(a, b))] for (a, b) in ((tri[1], tri[2]), (tri[2], tri[3]), (tri[3], tri[1]))]
            for (c1, c2) in ((es[1], es[2]), (es[2], es[3]), (es[1], es[3]))
                P1, N1, R1 = pats[c1]; common = findall(in(Set(pats[c2][1])), P1)
                S = skew_map(N1[common, :], R1[common, :]); s = svdvals(S)
                minrank = min(minrank, count(>(1e-11 * s[1]), s))
            end
        end
    end
    H, info = symmetric_star(g.me, g.t, g.le, g.lde, g.delta, period, h; rho = rho, K = K, verbose = false)
    logp(@sprintf("%-12s ρ=%.1f %s E=%5d rank S_P %d..%d | dim ker G %d (pred. 4%s) | 1·ker %.1e | predicted in ker %.1e | overlap rank min %s | λ₅ %.2e λ₆ %.2e | max|θ| %.3f | partition %.1e",
                  name, rho, K === nothing ? " " : "K", E, minimum(ranks), maximum(ranks), nk, K === nothing ? "" : "+", comp, inker,
                  overlap ? string(minrank) : "-", ev[nk+1], ev[nk+2], max(abs(info.theta_min), abs(info.theta_max)), info.partition_err))
end

for n in (10, 20, 40)
    X, T, period, h = porting_mesh(n); analyse("lattice n=$n", X, T, period, h)
end
X, T, period, h = porting_mesh(20); analyse("lattice n=20", X, T, period, h; rho = 1.4)
θK = pi / 6; Q = [cos(θK) -sin(θK); sin(θK) cos(θK)]; Kmat = Q * Diagonal([1.0, 0.1]) * Q'
analyse("lattice n=20", X, T, period, h; K = Kmat, overlap = false)
for mn in ("mesh3", "mesh4")
    X, T, period, h = scvt_mesh(mn); analyse("scvt $mn", X, T, period, h)
end
X, T, period, h = scvt_mesh("mesh4"); analyse("scvt mesh4", X, T, period, h; rho = 1.4)
write(joinpath(OUT, "kernel.txt"), join(lines, "\n") * "\n")
