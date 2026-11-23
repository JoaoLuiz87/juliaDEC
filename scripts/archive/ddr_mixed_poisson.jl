# ============================================================================
#  ddr_mixed_poisson.jl -- DDR (k = 0, 1) for the mixed Poisson problem in 2D
#                          and the dual structure hidden in it (Part III of
#                          pitassi_dec_ddr.tex)
#
#      julia scripts/archive/ddr_mixed_poisson.jl
#
#  Problem: σ = -∇u, div σ = f in Ω = (0,1)², u = g on ∂Ω.
#  Discrete spaces (2D face construction of Di Pietro-Droniou, FoCM 2023, div version):
#      X_div,T = { σ_G ∈ ∇P^k(T) (cell mean for k = 1), σ^c ∈ J(x - x_T) P^{k-1}(T),
#                  σ_E ∈ P^k(E) normal flux density w.r.t. a global normal n_E },
#      u ∈ P^k(T).
#  Operators: D_T ∈ P^k (divergence), P_T ∈ P^k(T)² (potential), L²-product
#      (σ,τ)_T = ∫ PσPτ + Σ_E h_E ∫_E (Pσ·n_E - σ_E)(Pτ·n_E - τ_E).
#  Scheme: (σ,τ)_h - (u, Dτ) = -∫_∂Ω g τ·n,   (Dσ, q) = (f, q).
#
#  Checks printed: polynomial consistency, convergence, and the dual-structure
#  identities on the computed solutions (cell flux unknown = dual reconstruction).
#  Writes output/figures/ddr_mixed_data.txt (convergence) and output/figures/ddr_mixed_cells.txt.
# ============================================================================
using LinearAlgebra, SparseArrays, Printf, Random

const J = [0.0 -1.0; 1.0 0.0]
cross2(a, b) = a[1] * b[2] - a[2] * b[1]

# ---------------------------------------------------------------- mesh
struct Mesh
    X::Vector{Vector{Float64}}
    cells::Vector{Vector{Int}}           # counter-clockwise vertex lists
    edges::Vector{Tuple{Int,Int}}        # global orientation a -> b; global normal n_E = -J t_E (right of a -> b)
    cedges::Vector{Vector{Int}}          # local edge i = (V_i, V_{i+1}) -> global edge
    csign::Vector{Vector{Int}}           # ω = +1 if n_E is the outward normal of the cell
    bnd::Vector{Bool}
end
function Mesh(X, cells)
    dict = Dict{Tuple{Int,Int},Int}(); E = Tuple{Int,Int}[]; ce = Vector{Int}[]; cs = Vector{Int}[]; cnt = Int[]
    for c in cells
        li = Int[]; ls = Int[]
        for i in eachindex(c)
            a, b = c[i], c[mod1(i + 1, length(c))]
            key = (min(a, b), max(a, b))
            if haskey(dict, key)
                e = dict[key]; push!(li, e); push!(ls, E[e] == (a, b) ? 1 : -1); cnt[e] += 1
            else
                push!(E, (a, b)); e = length(E); dict[key] = e; push!(li, e); push!(ls, 1); push!(cnt, 1)
            end
        end
        push!(ce, li); push!(cs, ls)
    end
    Mesh(X, cells, E, ce, cs, cnt .== 1)
end
function gridmesh(N; kind = :tri, pert = 0.2, seed = 1)
    rng = MersenneTwister(seed); h = 1 / N
    idx(i, j) = j * (N + 1) + i + 1
    X = [[i * h, j * h] for j in 0:N for i in 0:N]
    for j in 1:N-1, i in 1:N-1
        X[idx(i, j)] += pert * h * (2rand(rng, 2) .- 1)
    end
    cells = Vector{Int}[]
    for j in 0:N-1, i in 0:N-1
        a, b, c, d = idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)
        if kind == :quad
            push!(cells, [a, b, c, d])
        elseif iseven(i + j)
            push!(cells, [a, b, c]); push!(cells, [a, c, d])
        else
            push!(cells, [a, b, d]); push!(cells, [b, c, d])
        end
    end
    Mesh(X, cells)
end

# ---------------------------------------------------------------- quadrature and local geometry
const A1, B1 = 0.059715871789770, 0.470142064105115
const A2, B2 = 0.797426985353087, 0.101286507323456
const TRI = vcat([((1 / 3, 1 / 3, 1 / 3), 0.225)], [((A1, B1, B1), 0.132394152788506), ((B1, A1, B1), 0.132394152788506), ((B1, B1, A1), 0.132394152788506)],
                 [((A2, B2, B2), 0.125939180544827), ((B2, A2, B2), 0.125939180544827), ((B2, B2, A2), 0.125939180544827)])
const GL = [(-0.861136311594053, 0.347854845137454), (-0.339981043584856, 0.652145154862546),
            (0.339981043584856, 0.652145154862546), (0.861136311594053, 0.347854845137454)]
struct Cell
    V::Vector{Vector{Float64}}; x̄::Vector{Float64}; area::Float64; h::Float64
    qx::Vector{Vector{Float64}}; qw::Vector{Float64}            # cell quadrature (degree 5)
end
function Cell(V)
    n = length(V); A = sum(cross2(V[i], V[mod1(i + 1, n)]) for i in 1:n) / 2
    x̄ = sum((V[i] + V[mod1(i + 1, n)]) * cross2(V[i], V[mod1(i + 1, n)]) for i in 1:n) / (6A)
    h = maximum(norm(V[i] - V[j]) for i in 1:n, j in 1:n)
    qx = Vector{Float64}[]; qw = Float64[]
    for i in 1:n
        a, b = V[i], V[mod1(i + 1, n)]; ar = cross2(a - x̄, b - x̄) / 2
        for ((l1, l2, l3), w) in TRI
            push!(qx, l1 * x̄ + l2 * a + l3 * b); push!(qw, w * ar)
        end
    end
    Cell(V, x̄, A, h, qx, qw)
end
edgequad(a, b) = (L = norm(b - a); ([(a + b) / 2 + (t / 2) * (b - a) for (t, _) in GL], [w * L / 2 for (_, w) in GL]))
ψ1(C, x) = (ξ = (x - C.x̄) / C.h; [1.0, ξ[1], ξ[2]])
∇ψ1(C) = [[0.0, 0.0], [1 / C.h, 0.0], [0.0, 1 / C.h]]
ψ2x(C, x) = (ξ = (x - C.x̄) / C.h; [ξ[1], ξ[2], ξ[1]^2, ξ[1] * ξ[2], ξ[2]^2])                     # P2 tests modulo constants
∇ψ2x(C, x) = (ξ = (x - C.x̄) / C.h; [[1, 0] / C.h, [0, 1] / C.h, [2ξ[1], 0] / C.h, [ξ[2], ξ[1]] / C.h, [0, 2ξ[2]] / C.h])
vecP1(C, x) = (p = ψ1(C, x); [[p[1], 0.0], [p[2], 0.0], [p[3], 0.0], [0.0, p[1]], [0.0, p[2]], [0.0, p[3]]])
"global edge basis {1, s} with s = (x - m_E)·t_E/|E| ∈ [-1/2, 1/2] (global orientation)"
edgebasis(a, b, x) = [1.0, dot(x - (a + b) / 2, b - a) / dot(b - a, b - a)]

# ---------------------------------------------------------------- local DDR operators
"""
Local operators on cell c for degree k. Local σ DOFs:
  k = 1: [σ_G (2); σ^c (1); edge i: 2 coefficients in the global edge basis]
  k = 0: [edge i: 1 coefficient]
Returns (D, P, A, B, info): D maps DOFs to P^k coefficients (ψ1 basis, first k == 0 ? 1 : 3),
P to vector P^k coefficients (k=1: vecP1, k=0: constant vector), A local L²-product, B = ∫ q D.
"""
function localops(M::Mesh, c, k)
    V = M.X[M.cells[c]]; C = Cell(V); n = length(V)
    E = [(M.X[M.edges[e][1]], M.X[M.edges[e][2]]) for e in M.cedges[c]]; ω = M.csign[c]
    nE = k == 1 ? 2 : 1; off = k == 1 ? 3 : 0; nσ = off + nE * n
    eslot(i) = off + nE * (i - 1) .+ (1:nE)
    if k == 1
        M1 = sum(w * ψ1(C, x) * transpose(ψ1(C, x)) for (x, w) in zip(C.qx, C.qw))
        rhsD = zeros(3, nσ)
        for j in 1:3
            g = ∇ψ1(C)[j]; rhsD[j, 1:2] = -C.area * g                         # -∫ σ_G·∇ψ_j
            for i in 1:n
                xs, ws = edgequad(E[i]...)
                rhsD[j, eslot(i)] = ω[i] * sum(w * ψ1(C, x)[j] * edgebasis(E[i]..., x) for (x, w) in zip(xs, ws))
            end
        end
        D = M1 \ rhsD
        # potential: rows r ∈ {ξ1, ξ2, ξ1², ξ1ξ2, ξ2²} and w = J(x - x_T), x_T = x̄
        Apot = zeros(6, 6); rhsP = zeros(6, nσ)
        for (x, w) in zip(C.qx, C.qw)
            vb = vecP1(C, x); gr = ∇ψ2x(C, x); r = ψ2x(C, x); jw = J * (x - C.x̄)
            for i in 1:5, j in 1:6
                Apot[i, j] += w * dot(vb[j], gr[i])
            end
            for j in 1:6
                Apot[6, j] += w * dot(vb[j], jw)
            end
            for i in 1:5
                rhsP[i, :] -= w * r[i] * (transpose(ψ1(C, x)) * D)[:]                 # -∫ D r
            end
            rhsP[6, 3] += w * dot(jw, jw)                                               # ∫ σ^c J(x-x_T)·J(x-x_T)
        end
        for i in 1:n
            xs, ws = edgequad(E[i]...)
            for (x, w) in zip(xs, ws), l in 1:5
                rhsP[l, eslot(i)] += ω[i] * w * ψ2x(C, x)[l] * edgebasis(E[i]..., x)
            end
        end
        P = Apot \ rhsP
        Mv = sum(w * [dot(a, b) for a in vecP1(C, x), b in vecP1(C, x)] for (x, w) in zip(C.qx, C.qw))
        A = transpose(P) * Mv * P
        for i in 1:n
            a, b = E[i]; nE_ = -J * (b - a) / norm(b - a); hE = norm(b - a); xs, ws = edgequad(a, b)
            for (x, w) in zip(xs, ws)
                g = transpose([dot(v, nE_) for v in vecP1(C, x)]) * P
                g[eslot(i)] -= edgebasis(a, b, x)
                A += w * hE * transpose(g) * g
            end
        end
        B = M1 * D
    else
        D = zeros(1, nσ); P = zeros(2, nσ)
        for i in 1:n
            a, b = E[i]; L = norm(b - a)
            D[1, i] = ω[i] * L / C.area
            P[:, i] = ω[i] * L * ((a + b) / 2 - C.x̄) / C.area                          # Pitassi: (1/|T|) Σ Φ_E (m_E - x̄)
        end
        A = C.area * transpose(P) * P
        for i in 1:n
            a, b = E[i]; nE_ = -J * (b - a) / norm(b - a); L = norm(b - a)
            g = transpose(nE_) * P; g[i] -= 1
            A += L * L * transpose(g) * g                                              # h_E ∫_E (P·n - σ_E)², integrand constant
        end
        B = C.area * D
    end
    return (D = D, P = P, A = A, B = B, C = C, E = E, ω = ω, nσ = nσ, eslot = eslot)
end

# ---------------------------------------------------------------- interpolation, assembly, solve
"Interpolate σ: returns local DOF vector on cell c."
function interp(M, c, k, σ, L)
    C = L.C; n = length(L.E); x = zeros(L.nσ)
    if k == 1
        x[1:2] = sum(w * σ(q) for (q, w) in zip(C.qx, C.qw)) / C.area
        x[3] = sum(w * dot(σ(q), J * (q - C.x̄)) for (q, w) in zip(C.qx, C.qw)) / sum(w * dot(q - C.x̄, q - C.x̄) for (q, w) in zip(C.qx, C.qw))
    end
    for i in 1:n
        a, b = L.E[i]; nE_ = -J * (b - a) / norm(b - a); xs, ws = edgequad(a, b)
        if k == 1
            Me = sum(w * edgebasis(a, b, q) * transpose(edgebasis(a, b, q)) for (q, w) in zip(xs, ws))
            x[L.eslot(i)] = Me \ sum(w * dot(σ(q), nE_) * edgebasis(a, b, q) for (q, w) in zip(xs, ws))
        else
            x[L.eslot(i)] .= sum(w * dot(σ(q), nE_) for (q, w) in zip(xs, ws)) / norm(b - a)
        end
    end
    x
end
function solve(M::Mesh, k, uex, σex, f)
    nc = length(M.cells); ne = length(M.edges)
    ncs = k == 1 ? 3 : 0; nE = k == 1 ? 2 : 1; nu = k == 1 ? 3 : 1
    Nσ = ncs * nc + nE * ne; Nu = nu * nc
    gσ(c, L) = vcat(collect(ncs * (c - 1) .+ (1:ncs)), vcat([ncs * nc + nE * (e - 1) .+ (1:nE) for e in M.cedges[c]]...))
    Ii = Int[]; Jj = Int[]; Vv = Float64[]; rhs = zeros(Nσ + Nu); locs = []
    for c in 1:nc
        L = localops(M, c, k); push!(locs, L); gs = gσ(c, L); gu = Nσ .+ nu * (c - 1) .+ (1:nu)
        for (i, gi) in enumerate(gs), (j, gj) in enumerate(gs)
            push!(Ii, gi); push!(Jj, gj); push!(Vv, L.A[i, j])
        end
        for (i, gi) in enumerate(gu), (j, gj) in enumerate(gs)          # [A Bᵀ; B 0] [σ; -u]
            push!(Ii, gi); push!(Jj, gj); push!(Vv, L.B[i, j]); push!(Ii, gj); push!(Jj, gi); push!(Vv, L.B[i, j])
        end
        C = L.C
        rhs[gu] += sum(w * f(q) * (k == 1 ? ψ1(C, q) : [1.0]) for (q, w) in zip(C.qx, C.qw))
    end
    for e in 1:ne
        M.bnd[e] || continue
        a, b = M.X[M.edges[e][1]], M.X[M.edges[e][2]]; xs, ws = edgequad(a, b)   # boundary edges are stored outward
        g = sum(w * uex(q) * (k == 1 ? edgebasis(a, b, q) : [1.0]) for (q, w) in zip(xs, ws))
        rhs[ncs * nc + nE * (e - 1) .+ (1:nE)] -= g
    end
    K = sparse(Ii, Jj, Vv, Nσ + Nu, Nσ + Nu)
    sol = K \ rhs
    σh = sol[1:Nσ]; uh = -sol[Nσ+1:end]
    # errors
    num = 0.0; den = 0.0; eu = 0.0; nuex = 0.0; eP = 0.0; nP = 0.0
    for c in 1:nc
        L = locs[c]; gs = gσ(c, L); C = L.C
        xi = interp(M, c, k, σex, L); d = σh[gs] - xi
        num += dot(d, L.A * d); den += dot(xi, L.A * xi)
        # u: compare with the L² projection of u_ex
        Mloc = k == 1 ? sum(w * ψ1(C, q) * transpose(ψ1(C, q)) for (q, w) in zip(C.qx, C.qw)) : fill(C.area, 1, 1)
        πu = Mloc \ sum(w * uex(q) * (k == 1 ? ψ1(C, q) : [1.0]) for (q, w) in zip(C.qx, C.qw))
        du = uh[nu*(c-1).+(1:nu)] - πu
        eu += dot(du, Mloc * du); nuex += dot(πu, Mloc * πu)
        pc = L.P * σh[gs]
        for (q, w) in zip(C.qx, C.qw)
            Pq = k == 1 ? sum(pc[j] * vecP1(C, q)[j] for j in 1:6) : pc
            eP += w * norm(Pq - σex(q))^2; nP += w * norm(σex(q))^2
        end
    end
    return (σ = sqrt(num / den), u = sqrt(eu / nuex), P = sqrt(eP / nP), σh = σh, uh = uh, locs = locs, gσ = gσ, h = 1 / sqrt(nc))
end

# ---------------------------------------------------------------- the exact solution
uex(x) = sin(pi * x[1]) * cos(pi * x[2]) + x[1] * x[2]^2
∇uex(x) = [pi * cos(pi * x[1]) * cos(pi * x[2]) + x[2]^2, -pi * sin(pi * x[1]) * sin(pi * x[2]) + 2x[1] * x[2]]
σex(x) = -∇uex(x)
fex(x) = 2pi^2 * sin(pi * x[1]) * cos(pi * x[2]) - 2x[1]                         # f = -Δu

if abspath(PROGRAM_FILE) == @__FILE__
# ---------------------------------------------------------------- A. polynomial consistency of the local operators
println("A. local consistency (k = 1): P(Iσ) = σ for σ ∈ P1², D(Iσ) = π¹ div σ for σ ∈ P2²")
let M = gridmesh(3; kind = :quad, pert = 0.25, seed = 3)
    G = randn(2, 3); σ1(x) = G * [1, x[1], x[2]]
    Q = randn(2, 6); σ2(x) = Q * [1, x[1], x[2], x[1]^2, x[1] * x[2], x[2]^2]
    divσ2(x) = Q[1, 2] + 2Q[1, 4] * x[1] + Q[1, 5] * x[2] + Q[2, 3] + Q[2, 5] * x[1] + 2Q[2, 6] * x[2]
    worstP = 0.0; worstD = 0.0; worstS = 0.0
    for c in 1:length(M.cells)
        L = localops(M, c, 1); C = L.C
        x = interp(M, c, 1, σ1, L); pc = L.P * x
        worstP = max(worstP, maximum(norm(sum(pc[j] * vecP1(C, q)[j] for j in 1:6) - σ1(q)) for q in C.qx))
        worstS = max(worstS, abs(dot(x, L.A * x) - sum(w * norm(σ1(q))^2 for (q, w) in zip(C.qx, C.qw))))   # stabilisation vanishes on P1²
        x2 = interp(M, c, 1, σ2, L); dc = L.D * x2
        M1 = sum(w * ψ1(C, q) * transpose(ψ1(C, q)) for (q, w) in zip(C.qx, C.qw))
        πd = M1 \ sum(w * divσ2(q) * ψ1(C, q) for (q, w) in zip(C.qx, C.qw))
        worstD = max(worstD, norm(dc - πd))
    end
    @printf("   max |P(Iσ) - σ| = %.1e,  max |(Iσ,Iσ)_T - ∫|σ|²| = %.1e,  max |D(Iσ) - π¹div σ| = %.1e\n", worstP, worstS, worstD)
end

# ---------------------------------------------------------------- B. convergence
println("\nB. convergence (relative errors: discrete σ norm, ‖Pσ_h - σ‖, ‖u_h - π u‖)")
open(joinpath(@__DIR__, "..", "..", "output", "figures", "ddr_mixed_data.txt"), "w") do io
    println(io, "# kind k N h err_sigma err_P err_u")
    for kind in (:tri, :quad), k in (0, 1)
        prev = nothing
        for N in (4, 8, 16, 32, 64)
            M = gridmesh(N; kind = kind, pert = 0.2, seed = 7)
            r = solve(M, k, uex, σex, fex)
            rates = prev === nothing ? "" : @sprintf("  rates %.2f %.2f %.2f", log2(prev.σ / r.σ), log2(prev.P / r.P), log2(prev.u / r.u))
            @printf("   %-4s k=%d N=%3d: σ %.3e  Pσ %.3e  u %.3e%s\n", kind, k, N, r.σ, r.P, r.u, rates)
            println(io, "$kind $k $N $(1 / N) $(r.σ) $(r.P) $(r.u)")
            prev = r
        end
    end
end

# ---------------------------------------------------------------- C. the dual structure on the computed k = 1 solution
println("\nC. dual structure on the k = 1 solution (perturbed triangles, N = 16)")
let M = gridmesh(16; kind = :tri, pert = 0.2, seed = 7), k = 1
    r = solve(M, k, uex, σex, fex)
    worst1 = 0.0; worst2 = 0.0; worst3 = 0.0; worst4 = 0.0; gapG = 0.0; scale = 0.0
    open(joinpath(@__DIR__, "..", "..", "output", "figures", "ddr_mixed_cells.txt"), "w") do io
        println(io, "# cell xbar_x xbar_y sigmaG_x sigmaG_y Rflux_x Rflux_y gradu_x gradu_y")
        for c in eachindex(M.cells)
            L = r.locs[c]; C = L.C; x = r.σh[r.gσ(c, L)]; n = length(L.E)
            σG = x[1:2]
            # outward fluxes and first moments in the local counter-clockwise parameter
            Φ = zeros(n); μ = zeros(n); φm = zeros(n); φp = zeros(n); Rf = zeros(2); Rh = zeros(2)
            for i in 1:n
                a, b = L.E[i]; ω = L.ω[i]; s = x[L.eslot(i)]; len = norm(b - a)
                # global edge function σ_E(x) = s1 + s2 * σ_glob(x); outward density = ω σ_E; local ccw tangent = ω t_E
                Φ[i] = ω * s[1] * len
                μ[i] = ω * (ω * s[2]) * len / 12                                        # ∫ (ω σ_E) σ_loc ds, σ_loc = ω σ_glob
                τl = ω * (b - a) / len; ml = (a + b) / 2
                Rf += Φ[i] * (ml - C.x̄) + len * μ[i] * τl
                # half-edge fluxes (Bruni Bruno weights) and the 1/6, 5/6 points
                φm[i] = (Φ[i] - 3μ[i]) / 2; φp[i] = (Φ[i] + 3μ[i]) / 2
                qm = ml - len / 3 * τl; qp = ml + len / 3 * τl
                Rh += φm[i] * (qm - C.x̄) + φp[i] * (qp - C.x̄)
            end
            Rf /= C.area; Rh /= C.area
            fmom = sum(w * fex(q) * (q - C.x̄) for (q, w) in zip(C.qx, C.qw))
            worst1 = max(worst1, norm(σG - (Rf - fmom / C.area)) / norm(σG))            # σ_G = R_flux - (1/|T|)∫ f (x - x̄)
            worst2 = max(worst2, abs(sum(Φ) - sum(w * fex(q) for (q, w) in zip(C.qx, C.qw))) / maximum(abs.(Φ)))
            pc = L.P * x
            meanP = sum(w * sum(pc[j] * vecP1(C, q)[j] for j in 1:6) for (q, w) in zip(C.qx, C.qw)) / C.area
            worst3 = max(worst3, norm(meanP - σG) / norm(σG))
            worst4 = max(worst4, norm(Rh - Rf) / norm(Rf))
            gu = r.uh[3(c-1)+2:3(c-1)+3] / C.h                                          # ∇u_h (ψ1 = 1, ξ1, ξ2)
            gapG = max(gapG, norm(σG + gu)); scale = max(scale, norm(σG))
            @printf(io, "%d %.6f %.6f %.6e %.6e %.6e %.6e %.6e %.6e\n", c, C.x̄..., σG..., Rf..., gu...)
        end
    end
    @printf("   σ_G = R_flux(Φ, μ) - |T|⁻¹∫f(x-x̄):  %.1e\n   Σ_E Φ_E = ∫_T f:  %.1e\n   mean(Pσ) = σ_G:  %.1e\n", worst1, worst2, worst3)
    @printf("   R_flux from half-edge fluxes at the 1/6, 5/6 points = R_flux from (Φ, μ):  %.1e\n", worst4)
    @printf("   max |σ_G + ∇u_h| / max|σ_G| = %.2e  (Darcy's law at the dual node holds only up to discretisation error)\n", gapG / scale)
end
end # main
