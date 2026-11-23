# ============================================================================
#  green_charge_check.jl -- the simple proof that a symmetric star exact on linear fields cannot be local.
#  (1) hexagonal lattice, unit edge, generator 0 moved to (ε,0): ΔQ = Δ Σ_v |ṽ| γ(x_v), γ = (x1²−x2²)/2, against (√3/12) ε²(1−ε²)
#      (exact Voronoi cells by clipping; also a move in a general direction: ΔQ = (√3/12)(1−ε²) (a1²−a2²) ?)
#  (2) the patch skew Σ_e [(R∇φ)_e (R̃⋆∇γ)_e − (R∇γ)_e (R̃⋆∇φ)_e], φ = |x|²/2, changes by ±2ΔQ  (what Lemma `interior` of the paper gives)
#  (3) on an SCVT mesh with one generator moved: L = D0ᵀ H_P1 D0 before and after;  Σ_{v∈K,w∉K} ΔL_vw (φ_v γ_w − γ_v φ_w) = 2ΔQ on every circle; H_P2 not.
#      julia -t 4 scripts/barrier/green_charge_check.jl      -> ts_data/green_charge.txt
# ============================================================================
using LinearAlgebra, Printf
OUTG = String[]; sg(s) = (println(s); flush(stdout); push!(OUTG, s))
cr(a, b) = a[1] * b[2] - a[2] * b[1]
function clip(poly, p, q)                       # keep the half plane closer to p than to q
    m = (p .+ q) ./ 2; n = q .- p; f(x) = dot(x .- m, n); out = Vector{Vector{Float64}}()
    for i in eachindex(poly)
        a, b = poly[i], poly[mod1(i + 1, length(poly))]; fa, fb = f(a), f(b)
        fa <= 0 && push!(out, a)
        (fa < 0) != (fb < 0) && push!(out, a .+ (fa / (fa - fb)) .* (b .- a))
    end
    return out
end
area(p) = sum(cr(p[i], p[mod1(i + 1, length(p))]) for i in eachindex(p)) / 2
function vor(X, i; R = 3.0)
    P = [X[i] .+ [R, R], X[i] .+ [-R, R], X[i] .+ [-R, -R], X[i] .+ [R, -R]]
    for j in eachindex(X); j == i && continue; norm(X[j] .- X[i]) < 2R && (P = clip(P, X[i], X[j])); end
    return P
end
hexl(n) = [[i + j / 2, j * sqrt(3) / 2] for i in -n:n for j in -n:n if norm([i + j / 2, j * sqrt(3) / 2]) < n - 0.1]
γ(x) = (x[1]^2 - x[2]^2) / 2
sg("(1) hexagonal lattice: ΔQ = Δ Σ |ṽ| γ(x_v) for the move a of generator 0")
let X0 = hexl(6); i0 = findfirst(x -> norm(x) < 1e-12, X0); K = findall(x -> norm(x) < 3.2, X0)
    Q(X) = sum(area(vor(X, v)) * γ(X[v]) for v in K); Q0 = Q(X0)
    for a in ([0.05, 0.0], [0.2, 0.0], [0.5, 0.0], [0.0, 0.2], [0.15, 0.1])
        X = deepcopy(X0); X[i0] = a; ε2 = dot(a, a)
        sg(@sprintf("   a = (%.2f, %.2f)   ΔQ = %+.10f    (√3/12)(1−ε²)(a1²−a2²) = %+.10f", a[1], a[2], Q(X) - Q0, sqrt(3) / 12 * (1 - ε2) * (a[1]^2 - a[2]^2)))
    end
end
write(joinpath(@__DIR__, "..", "..", "output", "ts_data", "green_charge.txt"), join(OUTG, "\n") * "\n")
