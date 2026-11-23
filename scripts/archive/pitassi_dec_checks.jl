# ============================================================================
#  pitassi_dec_checks.jl -- checks for Part I of pitassi_dec_ddr.tex
#
#      julia scripts/archive/pitassi_dec_checks.jl
#
#  On single triangles (no mesh files needed):
#   A. every symmetric H with PᵀHP = |T| I is |T| R_pᵀR_p + σ 11ᵀ for a unique (p, σ)
#   B. DEC diagonal star: p = circumcentre, σ = 1/(2 Σ tan θ) = 1/(2 Π tan θ)
#   C. Whitney (Galerkin) mass matrix: p = centroid, σ = Σ|E|² / (144 |T|)
#   D. DDR k = 0: p = centroid, σ = (h_T/9) Σ 1/|E|
#   E. the local Laplacian d0ᵀ H d0 is the P1 stiffness (cotangent) matrix for all of them
#   F. the diagonal star's error on affine fields is the offset: ε_e = -|ẽ| δ_e tᵀ G n (two-triangle diamond)
# ============================================================================
using LinearAlgebra, Printf, Random

const J = [0.0 -1.0; 1.0 0.0]
cross2(a, b) = a[1] * b[2] - a[2] * b[1]
area(a, b, c) = cross2(b - a, c - a) / 2                               # > 0 for counter-clockwise (a,b,c)
circumcentre(a, b, c) = (d = 2cross2(b - a, c - a); a + J * (dot(c - a, c - a) * (b - a) - dot(b - a, b - a) * (c - a)) / d)
edges(V) = [(V[1], V[2]), (V[2], V[3]), (V[3], V[1])]                # E_i opposite V_{i+2}
Pmat(V) = vcat([transpose(b - a) for (a, b) in edges(V)]...)             # circulations of a constant field
Rmat(V, p) = hcat([J * ((a + b) / 2 - p) for (a, b) in edges(V)]...) / area(V...)
"Recover (p, σ) from a symmetric H with PᵀHP = |T| I."
function node_and_sigma(H, V)
    T = area(V...); P = Pmat(V); x̄ = sum(V) / 3
    Rt = transpose(H * P) / T                                               # the reconstruction read off H
    v = (Rt - Rmat(V, x̄)) * ones(3) / 3                                    # Rt - R_x̄ = v 1ᵀ
    p = x̄ + J * (T * v)                                                     # R_p = R_x̄ + J(x̄ - p) 1ᵀ/|T|
    S = H - T * transpose(Rt) * Rt
    return p, S[1, 1], norm(S - S[1, 1] * ones(3, 3)) / norm(H), norm(Rt - Rmat(V, p)) / norm(Rt), norm(transpose(P) * H * P - T * I) / T
end
angles(V) = [(u = V[mod1(i + 1, 3)] - V[i]; w = V[mod1(i + 2, 3)] - V[i]; atan(cross2(u, w), dot(u, w))) for i in 1:3]
"DEC diagonal star on one triangle: ½ cot of the angle opposite each edge (edge i = (V_i, V_{i+1}) is opposite V_{i+2})."
Hdec(V) = (θ = angles(V); Diagonal([cot(θ[mod1(i + 2, 3)]) / 2 for i in 1:3]))
"Whitney edge mass matrix, edge i = (V_i, V_{i+1}), w_i = λ_a ∇λ_b - λ_b ∇λ_a (exact quadrature, degree 2)."
function Hwhitney(V)
    T = area(V...)
    gl = [J * (V[mod1(i + 2, 3)] - V[mod1(i + 1, 3)]) / (2T) for i in 1:3]     # ∇λ_i (counter-clockwise)
    λ(x) = (A = [V[1] V[2] V[3]; 1 1 1]; A \ [x; 1])
    w(i, x) = (a = i; b = mod1(i + 1, 3); l = λ(x); l[a] * gl[b] - l[b] * gl[a])
    mids = [(V[1] + V[2]) / 2, (V[2] + V[3]) / 2, (V[3] + V[1]) / 2]
    [T / 3 * sum(dot(w(i, x), w(j, x)) for x in mids) for i in 1:3, j in 1:3]
end
"DDR k = 0 star in circulation unknowns: |T| RᵀR + (I-PR)ᵀ diag(h/|E|) (I-PR), centroid node."
function Hddr(V)
    T = area(V...); P = Pmat(V); R = Rmat(V, sum(V) / 3)
    h = maximum(norm(a - b) for (a, b) in edges(V))
    D = Diagonal([h / norm(b - a) for (a, b) in edges(V)])
    T * transpose(R) * R + transpose(I - P * R) * D * (I - P * R)
end
d0 = [-1.0 1 0; 0 -1 1; 1 0 -1]
stiff(V) = (T = area(V...); g = [J * (V[mod1(i + 2, 3)] - V[mod1(i + 1, 3)]) / (2T) for i in 1:3]; [T * dot(g[i], g[j]) for i in 1:3, j in 1:3])

Random.seed!(11)
tris = [("equilateral", [[0.0, 0.0], [1.0, 0.0], [0.5, sqrt(3) / 2]]),
        ("right (0,0),(3,0),(0,3)", [[0.0, 0.0], [3.0, 0.0], [0.0, 3.0]]),
        ("acute (0,0),(4,0),(1,3)", [[0.0, 0.0], [4.0, 0.0], [1.0, 3.0]]),
        ("obtuse (0,0),(4,0),(3,1)", [[0.0, 0.0], [4.0, 0.0], [3.0, 1.0]])]
for k in 1:3
    V = [randn(2) for _ in 1:3]; area(V...) < 0 && (V = V[[1, 3, 2]]); push!(tris, ("random $k", V))
end
@printf("%-26s %-6s %10s %10s | %9s %9s %9s %9s\n", "triangle", "star", "σ", "σ formula", "|p-pform|", "rank-1", "R=R_p", "PᵀHP")
for (name, V) in tris
    T = area(V...); x̄ = sum(V) / 3; c = circumcentre(V...); θ = angles(V)
    for (lbl, H, pf, σf) in (("DEC", Hdec(V), c, 1 / (2sum(tan.(θ)))),
                             ("Whit", Hwhitney(V), x̄, sum(norm(a - b)^2 for (a, b) in edges(V)) / (144T)),
                             ("DDR0", Hddr(V), x̄, maximum(norm(a - b) for (a, b) in edges(V)) / 9 * sum(1 / norm(a - b) for (a, b) in edges(V))))
        if lbl == "DEC" && abs(θ[1] - pi / 2) < 1e-12 || abs(θ[2] - pi / 2) < 1e-12 || abs(θ[3] - pi / 2) < 1e-12
            σf = lbl == "DEC" ? 0.0 : σf                                    # tan 90° = ∞: σ_DEC = 0
        end
        p, σ, r1, rr, ph = node_and_sigma(Matrix(H), V)
        @printf("%-26s %-6s %10.5f %10.5f | %9.1e %9.1e %9.1e %9.1e   L-stiff %.1e\n", name, lbl, σ, σf, norm(p - pf) / sqrt(T), r1, rr, ph,
                norm(transpose(d0) * Matrix(H) * d0 - stiff(V)) / norm(stiff(V)))
    end
    @printf("%-26s        Σtanθ = %.4f, Πtanθ = %.4f\n", "", sum(tan.(θ)), prod(tan.(θ)))
end

# ---------------------------------------------------------------- F. the diagonal star's P1 error is the offset
println("\nF. two triangles sharing e = [x_a, x_b]: diagonal star flux error on affine fields")
for k in 1:4
    xa, xb = randn(2), randn(2); t = (xb - xa) / norm(xb - xa); n = J * t
    p1 = (xa + xb) / 2 + 0.3randn() * (xb - xa) - (0.5 + rand()) * norm(xb - xa) * n          # T1 on the -n side
    p2 = (xa + xb) / 2 + 0.3randn() * (xb - xa) + (0.5 + rand()) * norm(xb - xa) * n          # T2 on the +n side
    c1, c2 = circumcentre(xa, xb, p1), circumcentre(xa, xb, p2)
    s1, s2 = dot(c1 - (xa + xb) / 2, n), dot(c2 - (xa + xb) / 2, n)
    L̃ = s2 - s1; δ = (s1 + s2) / 2; R1, R2 = norm(c1 - xa), norm(c2 - xa)
    u0, G = randn(2), randn(2, 2); u(x) = u0 + G * x
    α = dot(u((xa + xb) / 2), xb - xa)                                     # exact for affine u
    β = dot(u((c1 + c2) / 2), t) * L̃                                       # flux through the dual edge, ∫_ẽ u·t
    ε = (L̃ / norm(xb - xa)) * α - β
    @printf("   diamond %d: ε = %+.6e,  -|ẽ|δ tᵀGn = %+.6e,  |ẽ|δ = %+.5f, ½(R2²-R1²) = %+.5f, δ/|ẽ| = %+.3f\n",
            k, ε, -L̃ * δ * dot(t, G * n), L̃ * δ, (R2^2 - R1^2) / 2, δ / L̃)
end

# ---------------------------------------------------------------- G. exact worked examples quoted in the text
println("\nG. worked examples (exact rationals)")
let Q = Rational{BigInt}
    Jq(v) = [-v[2], v[1]]
    simpson(f) = (f(Q(0)) + 4f(Q(1, 2)) + f(Q(1))) / 6
    # the diamond of Example 6: e = [(0,0),(2,0)], T1 apex (1,-2), T2 apex (1/2, 3/2), v = (y, 0)
    c1 = Q[1, -3 // 4]; c2 = Q[1, 1 // 2]
    R1sq = Q(1) + (Q(3, 4))^2; R2sq = Q(1) + (Q(1, 2))^2; L̃ = c2[2] - c1[2]; δ = (c1[2] + c2[2]) / 2
    β = simpson(t -> (c1 + t * (c2 - c1))[2] * L̃)                     # ∫_ẽ v·t ds with v·t = y
    println("   diamond: |ẽ| = $L̃, δ = $δ, |ẽ|δ = $(L̃ * δ), ½(R2²-R1²) = $((R2sq - R1sq) / 2), β_e(v) = $β, ε_e = $(0 - β)")
    println("   s_e(a,v) for a = (1,0): α_e(v)β_e(a) - α_e(a)β_e(v) = $(0 * L̃ - 2 * β), formula -|e||ẽ|δ = $(-2 * L̃ * δ)")
    # Part III example: reference triangle, σ = (-x², 0), f = div σ = -2x
    V = [Q[0, 0], Q[1, 0], Q[0, 1]]; T = Q(1, 2); x̄ = Q[1 // 3, 1 // 3]
    σ(x) = [-x[1]^2, Q(0)]
    Φ = Q[]; μ = Q[]; Rf = Q[0, 0]; Rh = Q[0, 0]; φs = Q[]
    for i in 1:3
        a, b = V[i], V[mod1(i + 1, 3)]; ν = -Jq(b - a)                    # |E| ν (outward, scaled)
        Φi = simpson(t -> dot(σ(a + t * (b - a)), ν)); μi = simpson(t -> dot(σ(a + t * (b - a)), ν) * (t - Q(1, 2)))
        φm = (Φi - 3μi) / 2; φp = (Φi + 3μi) / 2
        m = (a + b) / 2; qm = m - (b - a) / 3; qp = m + (b - a) / 3
        Rf += Φi * (m - x̄) + μi * (b - a); Rh += φm * (qm - x̄) + φp * (qp - x̄)
        push!(Φ, Φi); push!(μ, μi); push!(φs, φm, φp)
    end
    # ∫_T g for polynomials of degree ≤ 2: edge-midpoint rule
    tri(g) = T * (g((V[1] + V[2]) / 2) + g((V[2] + V[3]) / 2) + g((V[3] + V[1]) / 2)) / 3
    σbar = [tri(x -> σ(x)[1]), tri(x -> σ(x)[2])] / T
    fm = [tri(x -> -2x[1] * (x[1] - x̄[1])), tri(x -> -2x[1] * (x[2] - x̄[2]))]
    println("   triangle: Φ = $Φ, μ = $μ, half-edge fluxes = $φs")
    println("   |T| R_flux = $Rf (from half edges: $Rh), ∫ f (x - x̄) = $fm")
    println("   R_flux - |T|⁻¹∫f(x-x̄) = $(Rf / T - fm / T),  mean σ = $σbar")
end

# ---------------------------------------------------------------- H. which part of ∇v the offset sees
println("\nH. tᵀGn = tᵀG₀n - curl/2 (dilations invisible); rotation part cancels around a triangle, shear part does not")
let
    worst = 0.0
    for k in 1:100
        G = randn(2, 2); θ = 2pi * rand(); t = [cos(θ), sin(θ)]; n = J * t
        dv = G[1, 1] + G[2, 2]; cv = G[2, 1] - G[1, 2]; G0 = (G + G') / 2 - dv / 2 * I
        worst = max(worst, abs(dot(t, G * n) - (dot(t, G0 * n) - cv / 2)))
    end
    @printf("   decomposition, 100 random (G, t): max deviation %.1e\n", worst)
    # diagonal star on a random diamond, dilation field: error zero
    xa, xb = randn(2), randn(2); t = (xb - xa) / norm(xb - xa); n = J * t
    p1 = (xa + xb) / 2 + 0.3randn() * (xb - xa) - (0.5 + rand()) * norm(xb - xa) * n
    p2 = (xa + xb) / 2 + 0.3randn() * (xb - xa) + (0.5 + rand()) * norm(xb - xa) * n
    c1, c2 = circumcentre(xa, xb, p1), circumcentre(xa, xb, p2); L̃ = dot(c2 - c1, n)
    v(x) = [0.3, -0.7] + 1.9 * (x - [0.2, 0.5])
    ε = (L̃ / norm(xb - xa)) * dot(v((xa + xb) / 2), xb - xa) - L̃ * dot(v((c1 + c2) / 2), t)
    @printf("   diagonal star on the dilation v = v0 + 1.9(x - x0): ε_e = %.1e  (offset δ/|ẽ| = %.2f)\n", ε, dot((c1 + c2) / 2 - (xa + xb) / 2, n) / L̃)
    # loop sums around the 3-4-5 triangle (counter-clockwise), a = (1,0)
    Q = Rational{BigInt}; V = [Q[0, 0], Q[4, 0], Q[0, 3]]; a = Q[1, 0]
    Jq(w) = [-w[2], w[1]]; S0 = Q[1 0; 0 -1]
    rot = sum(dot(a, b - c) * dot(b - c, Jq(Jq(b - c))) // dot(b - c, b - c) for (c, b) in zip(V, circshift(V, -1)))
    she = sum(dot(a, b - c) * dot(b - c, S0 * Jq(b - c)) // dot(b - c, b - c) for (c, b) in zip(V, circshift(V, -1)))
    println("   triangle (0,0),(4,0),(0,3), a = (1,0): Σ |e|(a·t)(tᵀJn) = $rot (rotation), Σ |e|(a·t)(tᵀG₀n) = $she for G₀ = diag(1,-1)")
end

# ---------------------------------------------------------------- I. Example "the offset goes to zero, the relative skew does not"
println("\nI. diamond refined by similarity s: a = (1,0), v_s = (1 + y/s, 0) (transported), v = (1 + y, 0) (fixed)")
let Q = Rational{BigInt}
    for s in (Q(1), Q(1, 2), Q(1, 4), Q(1, 8))
        c1y = -Q(3, 4) * s; c2y = Q(1, 2) * s; L̃ = c2y - c1y; δ = (c1y + c2y) / 2; e = 2s
        βvs = L̃ + (c2y^2 - c1y^2) / (2s); βv = L̃ + (c2y^2 - c1y^2) / 2
        ss = e * L̃ - e * βvs; sf = e * L̃ - e * βv; E = e * L̃
        println("   s = $s: δ = $δ, s_e(a,v_s) = $ss, energy = $E, relative = $(ss / E); s_e(a,v) = $sf, relative = $(sf / E)")
    end
end
