# ============================================================================
#  lie_transport_check.jl -- accuracy and stability of the discrete Lie derivatives of lie_derivative.jl
#
#      julia -t 4 scripts/transport/lie_transport_check.jl          ->  output/ts_data/lie_transport.{txt,jls}
#
#  A. Truncation error of the density Lie derivative, u = swirl + ½∇φ (divergent), ρ = 1 + bump:
#        τ_c = [ (D̃ K ρ̄)_c − ∫_c div(ρu) ] / A_c       (max and area-weighted L2, relative to max|div(ρu)|)
#     and the free-stream error  D̃ H v / A  for ρ ≡ 1 and the divergence-free swirl (exact value 0).
#     Scalar Lie derivative (L_u q)_c against u·∇q at the generator, q = bump, same ρ and u.
#  B. Transport of ρ̃ = bump by u(x,t) = (1/T,0) + cos(πt/T) swirl(x - t/T, y), T = 1 (Nair–Lauritzen), RK4, dt = h/8;
#     the exact solution at T is the initial state.  Errors of the cell values ρ̄ at T (max and L2).
#     Also ρ ≡ 1 (free-stream) per star, mass conservation and the change of Σ ρ̃²/A.
#  C. Spectrum of the steady operator -L (divergence-free swirl) on mesh4 (and mesh5 for the main schemes):
#     max Re λ (growth rate; the continuous operator is skew, Re λ = 0).
# ============================================================================
include(joinpath(@__DIR__, "lie_derivative.jl"))
using Printf, Serialization
empty!(lines)

const LMESHES = haskey(ENV, "LIE_MESHES") ? Tuple(String.(split(ENV["LIE_MESHES"], ","))) : ("mesh2", "mesh3", "mesh4", "mesh5", "mesh6", "mesh7")
const LSTARS = ("H_DEC", "H_P1", "H_P2", "exact")
combos() = [(s, v) for s in LSTARS for v in VARIANTS if !(s == "exact" && v in ("DEC", "Ham")) && !(s == "H_DEC" && v in ("DEC", "Ham"))]
label(s, v) = s == "H_DEC" && v == "TRiSK" ? "H_DEC  TRiSK=DEC=Ham" : @sprintf("%-6s %s", s, v)
R = Dict{Any,Any}()
order(a, b, ha, hb) = log(a / b) / log(ha / hb)

for mn in LMESHES
    t0 = time()
    G = lie_setup(mn); mesh = G["mesh"]; A = G["A"]; h = G["h"]; R[(mn, :h)] = h
    logp(@sprintf("\n%s: %d cells, h = %.5f (setup %.0f s)", mn, mesh.cells.n, h, time() - t0))
    # ---------------- A. truncation
    ρf(x) = 1 + bump(x)
    ρ̄ = discretize_scalar_field_cells(mesh, ρf; qdeg = 5).values ./ A
    vd = reduce_primal_edges(mesh, swirl_div).values; Ud = reduce_dual_edges(mesh, star_field(swirl_div)).values
    divex = G["Div"] * reduce_dual_edges(mesh, star_field(x -> ρf(x) .* swirl_div(x))).values      # = ∫_c div(ρu)
    scale = maximum(abs.(divex ./ A))
    v0 = reduce_primal_edges(mesh, swirl).values; U0 = reduce_dual_edges(mesh, star_field(swirl)).values
    qf(x) = bump(x; x0 = (0.4, 0.5), σ = 0.12)
    q = [qf(_vec2(p)) for p in mesh.cells.position]; ρ̃ = ρ̄ .* A
    Lqex = [dot(swirl_div(_vec2(p)), ∇bump(_vec2(p); x0 = (0.4, 0.5), σ = 0.12)) for p in mesh.cells.position]
    qscale = maximum(abs.(Lqex))
    for s in LSTARS[1:3]
        fs = (G["Div"] * (G[s] * v0)) ./ A
        R[(mn, s, :free)] = (inf = maximum(abs.(fs)), l2 = sqrt(sum(A .* fs .^ 2)))
    end
    R[(mn, "exact", :free)] = (inf = maximum(abs.(G["Div"] * U0 ./ A)), l2 = 0.0)
    for (s, var) in combos()
        τ = (G["Div"] * (flux_operator(G, var, s, vd; Uex = Ud) * ρ̄) .- divex) ./ A
        Lq = lie_scalar(G, var, s, vd, ρ̃, q; Uex = Ud) .- Lqex
        R[(mn, s, var, :trunc)] = (inf = maximum(abs.(τ)) / scale, l2 = sqrt(sum(A .* τ .^ 2)) / scale,
                                   qinf = maximum(abs.(Lq)) / qscale, ql2 = sqrt(sum(A .* Lq .^ 2)) / qscale)
    end
    # ---------------- B. transport by the deformational flow (translation + reversing swirl), T = 1
    ρ0 = discretize_scalar_field_cells(mesh, bump; qdeg = 5).values
    dt = h / 8
    for (s, var) in combos()
        Ls = deform_operators(G, var, s)
        t = @elapsed (ρT = rk4_lie(Ls, ρ0, 1.0, dt))
        e = (ρT .- ρ0) ./ A
        R[(mn, s, var, :transport)] = (inf = maximum(abs.(e)), l2 = sqrt(sum(A .* e .^ 2)), mass = abs(sum(ρT) - sum(ρ0)) / sum(ρ0),
                                       var2 = sum(ρT .^ 2 ./ A) / sum(ρ0 .^ 2 ./ A) - 1, time = t)
        s == "H_P2" && var == "CFV3" && serialize(joinpath(TSDIR, "lie_final_$mn.jls"), (ρ0 = ρ0, ρT = ρT, A = A))
    end
    for s in LSTARS[1:3]                                      # free stream: ρ ≡ 1 under the same flow
        ρT = rk4_lie(deform_operators(G, "TRiSK", s), copy(A), 1.0, dt)
        R[(mn, s, :freeT)] = maximum(abs.(ρT ./ A .- 1))
    end
    logp(@sprintf("   free-stream |D̃Hv/A|∞:  H_DEC %.2e   H_P1 %.2e   H_P2 %.2e      after T=1: %.2e  %.2e  %.2e",
                  R[(mn, "H_DEC", :free)].inf, R[(mn, "H_P1", :free)].inf, R[(mn, "H_P2", :free)].inf,
                  R[(mn, "H_DEC", :freeT)], R[(mn, "H_P1", :freeT)], R[(mn, "H_P2", :freeT)]))
    logp("   scheme                      | trunc ρ: max     L2  | trunc q: max     L2  | transport T=1: max      L2    Σρ̃²/A-1   (s)")
    for (s, var) in combos()
        a = R[(mn, s, var, :trunc)]; b = R[(mn, s, var, :transport)]
        logp(@sprintf("   %-27s | %9.2e %9.2e | %9.2e %9.2e | %9.2e %9.2e %+9.1e  %5.1f", label(s, var), a.inf, a.l2, a.qinf, a.ql2,
                      b.inf, b.l2, b.var2, b.time))
    end
    serialize(joinpath(TSDIR, "lie_transport.jls"), R)
end

logp("\nObserved orders between consecutive meshes (" * join(LMESHES, " -> ") * ")")
hs = [R[(m, :h)] for m in LMESHES]
ord(v) = join([@sprintf("%5.2f", order(v[i], v[i+1], hs[i], hs[i+1])) for i in 1:length(v)-1], " ")
for s in LSTARS[1:3]
    logp(@sprintf("   free stream %-6s max: %s", s, ord([R[(m, s, :free)].inf for m in LMESHES])))
end
for (s, var) in combos()
    logp(@sprintf("   %-27s trunc ρ max %s | L2 %s", label(s, var), ord([R[(m, s, var, :trunc)].inf for m in LMESHES]), ord([R[(m, s, var, :trunc)].l2 for m in LMESHES])))
    logp(@sprintf("   %-27s transport max %s | L2 %s", "", ord([R[(m, s, var, :transport)].inf for m in LMESHES]), ord([R[(m, s, var, :transport)].l2 for m in LMESHES])))
end

# ---------------- C. spectra of the steady operator (divergence-free swirl)
logp("\nC. spectrum of -L for the steady swirl: max Re λ (growth rate), max|λ|, and max Re λ of the modes with |λ| < 10")
for mn in (get(ENV, "LIE_SPEC", "1") == "1" ? ("mesh4", "mesh5") : ())
    G = lie_setup(mn; cfv = mn == "mesh4" ? (2, 3) : (3,)); mesh = G["mesh"]
    v0 = reduce_primal_edges(mesh, swirl).values; U0 = reduce_dual_edges(mesh, star_field(swirl)).values
    sel = mn == "mesh4" ? combos() : [(s, v) for (s, v) in combos() if v in ("TRiSK", "Ham", "CFV3")]
    for (s, var) in sel
        λ = eigvals(Matrix(-lie_density(G, flux_operator(G, var, s, v0; Uex = U0))))
        low = filter(z -> abs(z) < 10, λ)
        R[(mn, s, var, :spec)] = λ
        logp(@sprintf("   %s %-27s max Re λ %+.3e   max|λ| %.1f   low modes max Re λ %+.2e", mn, label(s, var), maximum(real.(λ)),
                      maximum(abs.(λ)), isempty(low) ? NaN : maximum(real.(low))))
    end
    serialize(joinpath(TSDIR, "lie_transport.jls"), R)
end
write(joinpath(TSDIR, "lie_transport.txt"), join(lines, "\n") * "\n")
