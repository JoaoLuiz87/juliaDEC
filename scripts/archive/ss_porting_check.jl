# ============================================================================
#  ss_porting_check.jl -- validate SymStar.jl against Table 5 of symmetric_star_guide.pdf, and test
#                         the singular share matrix G
#
#      julia scripts/archive/ss_porting_check.jl        (writes ss_data/porting.txt)
#
#  Mesh of the guide's Section 6.5: n = 20, Lx = 1, Ly = √3/2, N = 400, h = sqrt(Lx Ly / N),
#    x_ij = ((i + (j mod 2)/2 + 1/4) Lx/n + 0.1 h sin(2π(3i+2j)/n), (j + 1/2) Ly/n + 0.1 h cos(2π(i+4j)/n + 0.5)) mod (Lx, Ly),
#  triangulated like the equilateral lattice (Delaunay and well-centred: checked below).
# ============================================================================
include(joinpath(@__DIR__, "..", "..", "SymStar.jl")); using .SymStar
using LinearAlgebra, SparseArrays, Printf
const OUT = joinpath(@__DIR__, "..", "..", "output", "ss_data"); mkpath(OUT)
lines = String[]; logp(s) = (println(s); flush(stdout); push!(lines, s))

function porting_mesh(n = 20; Lx = 1.0, Ly = sqrt(3) / 2)
    N = n^2; h = sqrt(Lx * Ly / N); idx(i, j) = mod(j, n) * n + mod(i, n) + 1
    X = [[mod((i + 0.5 * mod(j, 2) + 0.25) * Lx / n + 0.1h * sin(2pi * (3i + 2j) / n), Lx),
          mod((j + 0.5) * Ly / n + 0.1h * cos(2pi * (i + 4j) / n + 0.5), Ly)] for j in 0:n-1 for i in 0:n-1]
    T = NTuple{3,Int}[]
    for j in 0:n-1, i in 0:n-1
        if iseven(j)
            push!(T, (idx(i, j), idx(i + 1, j), idx(i, j + 1))); push!(T, (idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)))
        else
            push!(T, (idx(i, j), idx(i + 1, j + 1), idx(i, j + 1))); push!(T, (idx(i, j), idx(i + 1, j), idx(i + 1, j + 1)))
        end
    end
    X, T, (Lx, Ly), h
end

X, T, period, h = porting_mesh()
g = edge_geometry(X, T, period)
E = length(g.le)
# Delaunay and well-centred: both circumcentres on their own side of every edge (s1 < 0 < s2)
s1 = [dot(g.c1[e, :] - g.me[e, :], [-g.t[e, 2], g.t[e, 1]]) for e in 1:E]
s2 = [dot(g.c2[e, :] - g.me[e, :], [-g.t[e, 2], g.t[e, 1]]) for e in 1:E]
logp(@sprintf("mesh: %d vertices, %d edges; well-centred: %s; max|δ|/|ẽ| = %.4f (guide: 0.044)", length(X), E,
              all(s1 .< 0 .< s2), maximum(abs.(g.delta) ./ g.lde)))

kx, ky = 2pi / period[1], 2pi / period[2]
u(x) = [sin(ky * x[2]) + 0.4cos(kx * x[1]), cos(kx * x[1] + ky * x[2])]
θK = pi / 6; Q = [cos(θK) -sin(θK); sin(θK) cos(θK)]; K = Q * Diagonal([1.0, 0.1]) * Q'

ref = Dict("ρ=2" => (136.7500, -0.090518, 0.113113, 49.801435, 741.544716, 21.738824, 0.381036, 2.1707, 4.0916e-2),
           "ρ=1.4" => (59.5333, -0.357768, 0.888982, 291.609929, 949.074760, 28.508530, 0.429155, 4.3117, 5.8743e-2),
           "ρ=2, K" => (136.7500, -0.091591, 0.133728, 60.054923, 427.834143, 14.528631, 0.052476, 2.5396, NaN))
names = ("nnz/row", "θ min", "θ max", "Σ θ²", "tr H", "‖H‖_F", "λmin(H)", "max C_e", "flux max")
for (lbl, rho, KK) in (("ρ=2", 2.0, nothing), ("ρ=1.4", 1.4, nothing), ("ρ=2, K", 2.0, K))
    H, info = symmetric_star(g.me, g.t, g.le, g.lde, g.delta, period, h; rho = rho, K = KK)
    C, r = row_constants(H, g.me, g.t, g.le, g.lde, period, h)
    fl = NaN
    if KK === nothing
        α, β = reductions(g, u); fl = maximum(abs.(H * α - β) ./ g.lde)
    end
    vals = (info.nnz_per_row, info.theta_min, info.theta_max, info.theta_norm2, tr(H), norm(H), smallest_eigenvalue(H), maximum(C), fl)
    logp(@sprintf("\n%s: partition err %.1e, patch skew %.1e, asym %.1e, affine check %.1e", lbl, info.partition_err,
                  info.patch_skew_max, info.asym, check_affine(H, g.me, g.t, g.le, g.lde, g.delta, period, h; K = KK)))
    for (nm, v, rv) in zip(names, vals, ref[lbl])
        logp(@sprintf("   %-9s  Julia %+.10e   guide %+.6e   rel. diff %.1e", nm, v, rv, isnan(rv) ? NaN : abs(v - rv) / abs(rv)))
    end
end

# ---------------------------------------------------------------- the share matrix G is singular
logp("\nShare matrix G (ρ = 2): spectrum near zero and the predicted kernel")
pats = build_patches(g.me, g.t, g.le, g.lde, g.delta, period, h, 2.0)
Vs = [SymStar.rowspace(skew_map(N, R))[1] for (P, N, R) in pats]
G = share_matrix(pats, Vs, E)
ev = eigvals(Symmetric(Matrix(G)))
logp(@sprintf("   smallest eigenvalues of G: %s;  largest %.2f", join([@sprintf("%.1e", x) for x in ev[1:7]], ", "), ev[end]))
φ = atan.(g.t[:, 2], g.t[:, 1]); w = g.le .* g.lde .* g.delta
Kv = hcat(w .* cos.(φ), w .* sin.(φ), w .* cos.(3φ), w .* sin.(3φ))
logp(@sprintf("   ‖G λ‖/(‖G‖‖λ‖) for λ_e = w_e cos φ, sin φ, cos 3φ, sin 3φ (w = |e||ẽ|δ):  %s",
              join([@sprintf("%.1e", norm(G * Kv[:, k]) / (opnorm(Matrix(G)) * norm(Kv[:, k]))) for k in 1:4], ", ")))
logp(@sprintf("   1 ⊥ kernel (this is Theorem 3.12): |⟨1, λ⟩|/(√E‖λ‖) = %s",
              join([@sprintf("%.1e", abs(sum(Kv[:, k])) / (sqrt(E) * norm(Kv[:, k]))) for k in 1:4], ", ")))
try
    F = cholesky(Symmetric(G)); λ = F \ ones(E)
    logp(@sprintf("   plain cholesky(G) (guide Appendix B): succeeded, ‖λ‖ = %.1e", norm(λ)))
catch err
    logp("   plain cholesky(G) (guide Appendix B): fails with " * string(typeof(err)))
end
write(joinpath(OUT, "porting.txt"), join(lines, "\n") * "\n")
