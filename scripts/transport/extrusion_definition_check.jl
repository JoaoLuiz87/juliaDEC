# ============================================================================
#  extrusion_definition_check.jl -- do our discrete contractions agree with the definition used in the literature?
#
#  Definition (Bossavit 2003; Hirani 2003, Lemma 8.3.2; Desbrun–Hirani–Leok–Marsden 2005, Lemma 10.1; Mullen et al. 2011, eqs. (4) and (13);
#  Eldred–Waruszewski–Roberts 2025, eq. (3.7)):      ∫_S i_u α = d/dτ|_0 ∫_{E_τ(S)} α .
#  Forward extrusion oriented by (u, S); Mullen's backward ("upwind") extrusion E(S, −τ) has the opposite orientation, hence the minus sign of (13).
#
#  (A) sweep actual cells: the swept polygon / curve is computed by integrating the flow (RK4), without using any formula for i_u;
#      forward and backward sweeps, oriented by (u, S), divided by τ and extrapolated to τ → 0, against ∫_S i_u α.
#  (B) our operators Ĩ_2 (Voronoi edges), I_2 (Delaunay edges), I_1 (generators), as used in the paper, against ∫_S i_u α on all cells.
#  (C) the structural identities D1 L1 = L2 D1 and L1 D0 = D0 L0 (Mullen et al. §4.3; EWR eq. (3.5)).
#  (D) lowest-order versions on a square grid against eqs. (15)–(16) of Mullen et al.
#      julia -t 4 scripts/transport/extrusion_definition_check.jl [meshes]        -> output/ts_data/extrusion_definition.txt
# ============================================================================
include(joinpath(@__DIR__, "extrusion_oneform.jl"))
using Random
OUTD = String[]; sd(s) = (println(s); flush(stdout); push!(OUTD, s))
u(x) = [1.0 - sin(π * x[1])^2 * sin(2π * x[2]), sin(2π * x[1]) * sin(π * x[2])^2]              # translation + LeVeque swirl (steady)
ρf(x) = 1 + 0.5 * sin(2π * x[1]) * cos(2π * x[2]) + 0.3 * cos(2π * (x[1] + x[2]))
bf(x) = [sin(2π * x[2]) + 0.3 * cos(2π * x[1]), 0.5 * cos(2π * (x[1] - x[2]))]
ζf(x) = -π * sin(2π * (x[1] - x[2])) - 2π * cos(2π * x[2])                                      # ∂1 b2 − ∂2 b1
function flow(x, τ; n = 8)                                                                         # position after time τ (τ < 0: backwards)
    X = collect(x); d = τ / n
    for _ in 1:n; k1 = u(X); k2 = u(X .+ d / 2 .* k1); k3 = u(X .+ d / 2 .* k2); k4 = u(X .+ d .* k3); X = X .+ d / 6 .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4); end
    return X
end
cr(a, b) = a[1] * b[2] - a[2] * b[1]
"∫ of the 2-form f dx∧dy over the region swept by the segment p→q in time |τ|, oriented by (u, t); no use of i_u."
function swept2(f, p, q, τ; ns = 60, nl = 6)
    S = 0.0; T = abs(τ)
    for i in 1:ns, l in 1:nl
        c(s, σ) = flow(p .+ s .* (q .- p), sign(τ) * σ)
        s0, s1 = (i - 1) / ns, i / ns; σ0, σ1 = (l - 1) / nl * T, l / nl * T
        a, b = c(s0, σ0), c(s1, σ0); a2, b2 = c(s0, σ1), c(s1, σ1)                                  # σ1 is further along ±u
        A, B, C, D = τ > 0 ? (a, a2, b2, b) : (a2, a, b, b2)                                        # first along u, then along t
        area = (cr(B .- A, C .- A) + cr(C .- A, D .- A)) / 2
        S += area * f((A .+ B .+ C .+ D) ./ 4)
    end
    return S
end
"∫ of the 1-form b over the curve swept by the point x in time |τ|, oriented along u."
function swept1(x, τ; n = 40)
    S = 0.0; T = abs(τ)
    for k in 1:n
        P = flow(x, sign(τ) * (k - 1) / n * T); Q = flow(x, sign(τ) * k / n * T); m = (P .+ Q) ./ 2
        S += (τ > 0 ? 1 : -1) * dot(bf(m), Q .- P)
    end
    return S
end
rich(f, τ) = 2 * f(τ / 2) / (τ / 2) - f(τ) / τ                                                       # extrapolation of f(τ)/τ to τ → 0
g5(F, a, b) = sum((w / 2) * F(a .+ (ξ + 1) / 2 .* (b .- a)) for (ξ, w) in _GL5)

sd("(A) sweeping real cells (τ = 2e-3, 1e-3, extrapolated), 150 random cells of mesh 4; max |swept/τ − ∫ i_u α| / max|∫ i_u α|")
let mesh = VoronoiMesh("meshes/mesh4.nc"); Random.seed!(3); es = rand(1:mesh.edges.n, 150); cs = rand(1:mesh.cells.n, 150); τ = 2e-3
    for (lab, sgn) in (("forward  (DHLM, Hirani, Mullen (4))", 1), ("backward (Mullen (13), ours)", -1))
        e2 = e2p = e1 = 0.0; m2 = m2p = m1 = 0.0
        for e in es
            v1, v2, m = dual_edge(mesh, e); t = _unit_t(mesh, e); n = [-t[2], t[1]]
            dot(v2 .- v1, n) < 0 && ((v1, v2) = (v2, v1))                                           # ẽ oriented by n_e = J t_e
            ex = norm(v2 .- v1) * g5(x -> ρf(x) * dot(u(x), t), v1, v2)
            e2 = max(e2, abs(rich(s -> swept2(ρf, v1, v2, sgn * s), τ) - ex)); m2 = max(m2, abs(ex))
            a, b, mm, tt, nc, l = primal_edge(mesh, e); nn = [-tt[2], tt[1]]
            exp_ = l * g5(x -> -ζf(x) * dot(u(x), nn), a, b)
            e2p = max(e2p, abs(rich(s -> swept2(ζf, a, b, sgn * s), τ) - exp_)); m2p = max(m2p, abs(exp_))
        end
        for c in cs
            x = _vec2(mesh.cells.position[c]); ex = dot(u(x), bf(x)); e1 = max(e1, abs(rich(s -> swept1(x, sgn * s), τ) - ex)); m1 = max(m1, abs(ex))
        end
        sd(@sprintf("   %-36s density, Voronoi edges %.1e   vorticity, Delaunay edges %.1e   1-form, generators %.1e", lab, e2 / m2, e2p / m2p, e1 / m1))
    end
end

sd("\n(B) the discrete contractions of the paper against ∫_S i_u α (max error / max value); Ĩ_2 with the three stars, cubic mean-preserving density")
for mn in (isempty(ARGS) ? ["mesh4", "mesh5"] : ARGS)
    G = lie_setup(mn; cfv = (), upw = ()); mesh = G["mesh"]; extrusion_setup!(G; p_rho = 3, rings = 2, constrained = true); oneform_setup!(G)
    v = reduce_primal_edges(mesh, u).values; ρ̄ = cell_averages(mesh, ρf); Qv = ntuple(g -> G["Qg"][g] * v, 3)
    ex2 = zeros(mesh.edges.n); exz = zeros(mesh.edges.n)
    for e in 1:mesh.edges.n
        v1, v2, m = dual_edge(mesh, e); t = _unit_t(mesh, e); ex2[e] = norm(v2 .- v1) * g5(x -> ρf(x) * dot(u(x), t), v1, v2)
        a, b, mm, tt, nc, l = primal_edge(mesh, e); exz[e] = l * g5(x -> ζf(x) * dot(u(x), nc), a, b)
    end
    ld = G["ld"]; le = G["le"]; s = @sprintf("   %s  Ĩ_2 (per unit length):", mn)
    for st in ("H_DEC", "H_P1", "H_P2"); F = extrusion_flux(G, G[st] * v, Qv, ρ̄); s *= @sprintf(" %s %.2e", st, maximum(abs, (F .- ex2) ./ ld) / maximum(abs, ex2 ./ ld)); end
    ŵ, ux, uy = oneform_velocity(G, u); b = edge_integrals(mesh, bf); ζ̄ = G["σT"] .* (G["D1p"] * b) ./ G["AT"]
    Fz = vorticity_flux(G, ŵ, ζ̄); pc = point_contraction(G, ux, uy, b); exn = [dot(u(_vec2(x)), bf(_vec2(x))) for x in mesh.cells.position]
    s *= @sprintf("   I_2 %.2e   I_1 %.2e", maximum(abs, (Fz .- exz) ./ le) / maximum(abs, exz ./ le), maximum(abs, pc .- exn) / maximum(abs, exn))
    sd(s)
    L1, L2, L0 = oneform_matrices(G, ŵ, ux, uy); D1 = G["D1p"]; D0 = G["D0p"]
    K2 = D1 * (vorticity_matrix(G, ŵ) * Diagonal(G["σT"] ./ G["AT"]))
    sd(@sprintf("   %s  (C) ‖D1 L1 − L2 D1‖/‖D1 L1‖ = %.1e   ‖L1 D0 − D0 L0‖/‖L1 D0‖ = %.1e", mn, norm(D1 * L1 - K2 * D1) / norm(D1 * L1), norm(L1 * D0 - D0 * L0) / norm(L1 * D0)))
end

sd("\n(D) lowest order on a square grid (side ℓ, u, v > 0): eqs. (15)–(16) of Mullen et al. against ours with constant reconstructions and a straight extrusion")
let ℓ = 0.1, U = 0.7, V = 0.4, dω = 0.31, dωb = -0.2, ωx = 0.12, ωy = -0.05
    # Mullen: U = flux through vertical edges, V = flux through horizontal edges (integrated: velocity × ℓ); per unit time
    mull_h = -(V * ℓ) * dω / ℓ^2; mull_v = (U * ℓ) * dωb / ℓ^2                                        # horizontal edge (cell below), vertical edge (cell to the left)
    ours(t, ζ) = (nn = [-t[2], t[1]]; ℓ * (-dot([U, V], nn)) * ζ)                                     # |e| (−u·n_e) ζ̄_up
    sd(@sprintf("   i_u dω: horizontal edge  Mullen %+.6f  ours %+.6f ;  vertical edge  Mullen %+.6f  ours %+.6f", mull_h, ours([1.0, 0.0], dω / ℓ^2), mull_v, ours([0.0, 1.0], dωb / ℓ^2)))
    sd(@sprintf("   i_u ω at a node: Mullen %+.6f  ours u♯·b♯ with b♯ = (ωx, ωy)/ℓ of the upstream edges %+.6f", (U * ℓ * ωx + V * ℓ * ωy) / ℓ^2, dot([U, V], [ωx, ωy] ./ ℓ)))
end
write(joinpath(@__DIR__, "..", "..", "output", "ts_data", "extrusion_definition.txt"), join(OUTD, "\n") * "\n")
