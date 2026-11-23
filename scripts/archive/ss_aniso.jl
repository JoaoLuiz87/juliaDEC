# ============================================================================
#  ss_aniso.jl -- constant anisotropic permeability: DEC_K, the exact K-star (guide), the uniform-share variant
#
#      julia scripts/archive/ss_aniso.jl        (writes ss_data/aniso.txt)
#
#  K = Q diag(1, 0.1) Qᵀ, Q the rotation by 30° (guide, Table 4).  Flux consistency: max_e |(H α(u) - β_K(u))_e|/|ẽ|,
#  β_K(u) = ∫_ẽ (K u)·t.  Darcy/Poisson: d0ᵀ H d0 p_h = d0ᵀ β_K(∇p), errors as in the guide's Theorem 4.2.
# ============================================================================
include(joinpath(@__DIR__, "..", "..", "SymStar.jl")); using .SymStar
include(joinpath(@__DIR__, "ss_mesh.jl"))
using LinearAlgebra, SparseArrays, Printf
const OUT = joinpath(@__DIR__, "..", "..", "output", "ss_data")
lines = String[]; logp(s) = (println(s); flush(stdout); push!(lines, s))
θK = pi / 6; Q = [cos(θK) -sin(θK); sin(θK) cos(θK)]; const KM = Q * Diagonal([1.0, 0.1]) * Q'
logp(@sprintf("%-11s %6s %-14s %8s %8s %9s %9s %9s %9s %7s", "mesh", "E", "star", "max|θ|", "max C_e", "affine", "flux max", "p max", "flux H⁻¹", "λmin"))

function darcy(g, S, period)
    kx, ky = 2pi / period[1], 2pi / period[2]
    p(x) = sin(kx * x[1] + 0.3) * cos(ky * x[2]) + 0.5cos(kx * x[1] + 2ky * x[2])
    gp(x) = [kx * cos(kx * x[1] + 0.3) * cos(ky * x[2]) - 0.5kx * sin(kx * x[1] + 2ky * x[2]),
             -ky * sin(kx * x[1] + 0.3) * sin(ky * x[2]) - ky * sin(kx * x[1] + 2ky * x[2])]
    _, βg = reductions(g, gp; K = KM)
    nV = size(g.d0, 2); L = g.d0' * S * g.d0
    ph = (lu([L ones(nV); ones(1, nV) 0.0]) \ vcat(g.d0' * βg, 0.0))[1:nV]
    pI = zeros(nV); seen = falses(nV)
    for e in 1:size(g.d0, 1), (k, s) in zip(findnz(g.d0[e, :])...)
        seen[k] && continue
        pI[k] = p(g.me[e, :] + (s > 0 ? 0.5 : -0.5) * g.le[e] * g.t[e, :]); seen[k] = true
    end
    ph .+= sum(pI - ph) / nV
    F = cholesky(Symmetric(S)); Hinv(x) = sqrt(dot(x, F \ x))
    maximum(abs.(ph - pI)) / maximum(abs.(pI)), Hinv(S * g.d0 * ph - βg) / Hinv(βg)
end

function run(name, X, T, period, h)
    g = edge_geometry(X, T, period)
    kx, ky = 2pi / period[1], 2pi / period[2]
    u(x) = [sin(ky * x[2]) + 0.4cos(kx * x[1]), cos(kx * x[1] + ky * x[2])]
    α, β = reductions(g, u; K = KM)
    kn = [dot(g.t[e, :], KM * g.t[e, :]) for e in 1:length(g.le)]
    HdK = spdiagm(g.lde .* kn ./ g.le)
    for (lbl, S, info) in (("DEC_K", HdK, nothing),
                           ((H, i) = symmetric_star(g.me, g.t, g.le, g.lde, g.delta, period, h; K = KM, verbose = false); ("exact (guide)", H, i)),
                           ((H, i) = symmetric_star(g.me, g.t, g.le, g.lde, g.delta, period, h; K = KM, abstol = 1e-3, verbose = false); ("uniform shares", H, i)))
        C, _ = row_constants(S, g.me, g.t, g.le, g.lde, period, h)
        aff = info === nothing ? NaN : check_affine(S, g.me, g.t, g.le, g.lde, g.delta, period, h; K = KM)
        λm = try smallest_eigenvalue(S) catch; NaN end
        pe, fe = λm > 0 ? darcy(g, S, period) : (NaN, NaN)
        logp(@sprintf("%-11s %6d %-14s %8s %8.2f %9.1e %9.2e %9.2e %9.2e %7.4f", name, length(g.le), lbl,
                      info === nothing ? "-" : @sprintf("%.3f", max(abs(info.theta_min), abs(info.theta_max))), maximum(C) / maximum(kn),
                      aff, maximum(abs.(S * α - β) ./ g.lde), pe, fe, λm))
    end
end
for n in (20, 40, 80)
    run("lattice $n", porting_mesh(n)...)
end
for mn in ("mesh3", "mesh4", "mesh5")
    run("scvt $mn", scvt_mesh(mn)...)
end
write(joinpath(OUT, "aniso.txt"), join(lines, "\n") * "\n")
