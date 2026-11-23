# ============================================================================
#  route2_check.jl -- the Pitassi form H = Rᵀ M R + γ² W Wᵀ at degree k (Proposition "route2")
#
#      julia scripts/archive/route2_check.jl
#
#  A. k = 0 on the acute triangle (0,0),(4,0),(1,3): R = P⁺ is the centroid reconstruction,
#     so H P = |T| R_x̄ᵀ (barycentric dual vectors), not |T| R_cᵀ (Voronoi, DEC).
#  B. k = 1 on a stencil of a perturbed periodic mesh: P (stencil edges x 6), R = P⁺,
#     M = mass of the basis over the central triangle, W = basis of ker Pᵀ.
#       H P = Rᵀ M for every γ;  (P d)ᵀ H (P c) = dᵀ M c;  Rᵀ M versus the Voronoi fluxes B;
#       PᵀB - (PᵀB)ᵀ equals the matrix of s_W(φ_j, φ_i) over the stencil edges.
# ============================================================================
using LinearAlgebra, Printf, Random
include(joinpath(@__DIR__, "..", "barrier", "sh_rate_mesh.jl"))     # mesh(N, kind), unwrap, circumcentre, J, Ly, GL, segint

# ---------------------------------------------------------------- A
println("A. k = 0, triangle (0,0),(4,0),(1,3)")
let V = [[0.0, 0.0], [4.0, 0.0], [1.0, 3.0]]
    T = 6.0; x̄ = sum(V) / 3; c = circumcentre(V...)
    E = [(V[1], V[2]), (V[2], V[3]), (V[3], V[1])]
    P = vcat([transpose(b - a) for (a, b) in E]...)
    R = pinv(P); Rx = hcat([J * ((a + b) / 2 - x̄) for (a, b) in E]...) / T; Rc = hcat([J * ((a + b) / 2 - c) for (a, b) in E]...) / T
    H = T * R' * R + 0.7^2 * (nullspace(P') * nullspace(P')')
    @printf("   |P⁺ - R_x̄| = %.1e,  |H P - |T| R_x̄ᵀ| = %.1e,  |H P - |T| R_cᵀ| / |T R_cᵀ| = %.2f\n",
            norm(R - Rx), norm(H * P - T * Rx'), norm(H * P - T * Rc') / norm(T * Rc'))
    println("   rows of H P (induced dual vectors):  ", join([@sprintf("(%.3f, %.3f)", (H*P)[i, :]...) for i in 1:3], "  "))
    println("   Voronoi dual vectors J(m_E - c):     ", join([@sprintf("(%.3f, %.3f)", (J * ((a + b) / 2 - c))...) for (a, b) in E], "  "))
end

# ---------------------------------------------------------------- B
println("\nB. k = 1, stencil around one triangle of a perturbed periodic mesh (random 0.08h, N = 16)")
let (X, Tr, h) = mesh(16, "random")
    # edges with their two triangles
    edges = Dict{Tuple{Int,Int},Vector{Int}}()
    for (k, t) in enumerate(Tr), (a, b) in ((t[1], t[2]), (t[2], t[3]), (t[3], t[1]))
        push!(get!(edges, (min(a, b), max(a, b)), Int[]), k)
    end
    T0 = 137; tv = Tr[T0]; xc = X[tv[1]]
    # stencil: all edges of triangles that share a vertex with T0
    tris = [k for (k, t) in enumerate(Tr) if !isempty(intersect(collect(t), collect(tv)))]
    St = unique([(min(a, b), max(a, b)) for k in tris for (a, b) in ((Tr[k][1], Tr[k][2]), (Tr[k][2], Tr[k][3]), (Tr[k][3], Tr[k][1]))])
    V = [unwrap(X[p], xc) for p in tv]; x̄ = sum(V) / 3; area = abs(cross2(V[2] - V[1], V[3] - V[1])) / 2
    basis = [x -> [1.0, 0.0], x -> [0.0, 1.0], x -> [x[1] - x̄[1], 0.0], x -> [x[2] - x̄[2], 0.0], x -> [0.0, x[1] - x̄[1]], x -> [0.0, x[2] - x̄[2]]]
    nE = length(St); P = zeros(nE, 6); B = zeros(nE, 6)
    for (r, (a, b)) in enumerate(St)
        xa = unwrap(X[a], xc); xb = unwrap(X[b], xa); t = (xb - xa) / norm(xb - xa); n = J * t; m = (xa + xb) / 2
        cs = [circumcentre([unwrap(X[p], xa) for p in Tr[k]]...) for k in edges[(a, b)]]
        for j in 1:6
            P[r, j] = segint(x -> dot(basis[j](x), t), xa, xb)
            B[r, j] = segint(x -> dot(basis[j](x), t), cs[1], cs[2])
        end
    end
    # mass matrix over T0 (edge-midpoint rule, exact for degree 2)
    mids = [(V[1] + V[2]) / 2, (V[2] + V[3]) / 2, (V[3] + V[1]) / 2]
    M = [area / 3 * sum(dot(basis[i](x), basis[j](x)) for x in mids) for i in 1:6, j in 1:6]
    R = pinv(P); W = nullspace(P')
    @printf("   stencil: %d edges, rank P = %d, |R P - I| = %.1e\n", nE, rank(P), norm(R * P - I))
    for γ in (0.1, 1.0, 10.0)
        H = R' * M * R + γ^2 * W * W'
        @printf("   γ = %5.1f:  |H P - Rᵀ M| = %.1e,  |Pᵀ H P - M| = %.1e\n", γ, norm(H * P - R' * M), norm(P' * H * P - M))
    end
    A = P' * B
    # s_W(φ_j, φ_i) = Σ_e α_e(φ_i) β_e(φ_j) - α_e(φ_j) β_e(φ_i) over the stencil edges
    S = [sum(P[r, i] * B[r, j] - P[r, j] * B[r, i] for r in 1:nE) for i in 1:6, j in 1:6]
    @printf("   |(PᵀB - (PᵀB)ᵀ) - [s_W(φ_j,φ_i)]| = %.1e;  |skew(PᵀB)| / |PᵀB| = %.1e (a window sum: boundary-type)\n",
            norm((A - A') - S), norm(A - A') / norm(A))
end

# ---------------------------------------------------------------- C. the assembled cell-based sandwich, row by row
println("\nC. assembled sandwich H = Σ_T S_Tᵀ (R_Tᵀ M_T R_T + γ² W_T W_Tᵀ) S_T, stencil = 1-ring of T, R_T = P_T⁺ exact on [P1]²")
println("   row e applied to a polynomial u: (H α(u))_e = Σ_{T: e ∈ S_T} (R_Tᵀ M_T c_T(u))_e  (γ drops out); compared with β_e(u)")
for kind in ("lattice", "random")
    X, Tr, h = mesh(16, kind)
    edges = Dict{Tuple{Int,Int},Vector{Int}}()
    for (k, t) in enumerate(Tr), (a, b) in ((t[1], t[2]), (t[2], t[3]), (t[3], t[1]))
        push!(get!(edges, (min(a, b), max(a, b)), Int[]), k)
    end
    incident = Dict{Int,Vector{Int}}()
    for (k, t) in enumerate(Tr), p in t
        push!(get!(incident, p, Int[]), k)
    end
    cells = map(eachindex(Tr)) do k
        tv = Tr[k]; xc = X[tv[1]]
        tris = unique(vcat([incident[p] for p in tv]...))
        St = unique([(min(a, b), max(a, b)) for q in tris for (a, b) in ((Tr[q][1], Tr[q][2]), (Tr[q][2], Tr[q][3]), (Tr[q][3], Tr[q][1]))])
        V = [unwrap(X[p], xc) for p in tv]; x̄ = sum(V) / 3; area = abs(cross2(V[2] - V[1], V[3] - V[1])) / 2
        basis = [x -> [1.0, 0.0], x -> [0.0, 1.0], x -> [x[1] - x̄[1], 0.0], x -> [x[2] - x̄[2], 0.0], x -> [0.0, x[1] - x̄[1]], x -> [0.0, x[2] - x̄[2]]]
        P = zeros(length(St), 6)
        for (r, (a, b)) in enumerate(St)
            xa = unwrap(X[a], xc); xb = unwrap(X[b], xa); t = (xb - xa) / norm(xb - xa)
            for j in 1:6
                P[r, j] = segint(x -> dot(basis[j](x), t), xa, xb)
            end
        end
        mids = [(V[1] + V[2]) / 2, (V[2] + V[3]) / 2, (V[3] + V[1]) / 2]
        M = [area / 3 * sum(dot(basis[i](x), basis[j](x)) for x in mids) for i in 1:6, j in 1:6]
        (St = St, RtM = pinv(P)' * M, x̄ = x̄)
    end
    member = Dict{Tuple{Int,Int},Vector{Tuple{Int,Int}}}()
    for (k, c) in enumerate(cells), (r, e) in enumerate(c.St)
        push!(get!(member, e, Tuple{Int,Int}[]), (k, r))
    end
    worst0 = 0.0; worst1 = 0.0; dec1 = 0.0
    for ((a, b), ts) in edges
        xa = X[a]; xb = unwrap(X[b], xa); m = (xa + xb) / 2; t = (xb - xa) / norm(xb - xa); n = J * t
        cs = [circumcentre([unwrap(X[p], xa) for p in Tr[k]]...) for k in ts]
        L̃ = abs(dot(cs[2] - cs[1], n)); m̃ = (cs[1] + cs[2]) / 2
        for (u0, G, isconst) in (([1.0, 0.4], zeros(2, 2), true), ([0.0, 0.0], [0.3 1.0; -0.6 0.2], false))
            u(x) = u0 + G * (x - m)
            β = L̃ * dot(u(m̃), t)
            Hu = 0.0
            for (k, r) in member[(a, b)]
                c = cells[k]; x̄ = unwrap(c.x̄, m)
                coef = vcat(u0 + G * (x̄ - m), [G[1, 1], G[1, 2], G[2, 1], G[2, 2]])
                Hu += (c.RtM * coef)[r]
            end
            sc = L̃ * (norm(u0) + norm(G) * h)
            if isconst
                worst0 = max(worst0, abs(Hu - β) / sc)
            else
                worst1 = max(worst1, abs(Hu - β) / sc)
                dec1 = max(dec1, abs(L̃ * dot(u(m), t) - β) / sc)
            end
        end
    end
    @printf("   %-8s N=16: sandwich: max relative row error, constants %.1e, affine %.1e | diagonal star, affine %.1e\n",
            kind, worst0, worst1, dec1)
end
