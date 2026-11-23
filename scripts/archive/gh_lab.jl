# ============================================================================
#  gh_lab.jl -- Galerkin Hodge lab: the Hodge star extracted from a bilinear form
#
#      a(x, y) = Σ_C (x|S_C)ᵀ [ R_Cᵀ M_C R_C + σ_C (I - P_C R_C)ᵀ D_C (I - P_C R_C) ] (y|S_C)
#
#  C runs over cells (triangles or diamonds) that tile the torus, S_C is a stencil of primal edges,
#  P_C the reduction (circulations) of a basis of [P_k]² on S_C, M_C the mass matrix of that basis on C,
#  and R_C a LEFT INVERSE of P_C (R_C P_C = I).  R_C = P_C⁺ is the least-squares choice;
#  R_C = P_C⁺ + Z_C W_Cᵀ (W_C an orthonormal basis of ker P_Cᵀ) is the general one.
#  The Hodge star is the matrix H of a in the primal/dual pairing Σ_e x_e y_e.
#
#  constrained_R solves, for Z_C, the row-wise Voronoi moment conditions
#       Σ_C ∫_C φ̃_{C,e} · u  =  β_e(u) = ∫_ẽ u·t_e      for every edge e and every u ∈ [P_k]²
#  (φ̃_{C,e} = column e of R_C as a field on C), with minimal Σ_C ‖Z_C W_Cᵀ‖²_{M_C}.
#  Meshes: periodic Delaunay triangulations, periods Px × Py, circumcentric (Voronoi) dual.
# ============================================================================
using LinearAlgebra, SparseArrays, Printf, Random, Statistics

const J = [0.0 -1.0; 1.0 0.0]
cross2(a, b) = a[1] * b[2] - a[2] * b[1]
circumcentre(a, b, c) = (d = 2cross2(b - a, c - a); a + J * (dot(c - a, c - a) * (b - a) - dot(b - a, b - a) * (c - a)) / d)

# Gauss-Legendre (4 points, exact to degree 7) on a segment, and a degree-5 rule on a triangle (signed area)
const GL4 = [(-0.861136311594053, 0.347854845137454), (-0.339981043584856, 0.652145154862546),
             (0.339981043584856, 0.652145154862546), (0.861136311594053, 0.347854845137454)]
segint(f, a, b) = sum(w * f((a + b) / 2 + (s / 2) * (b - a)) for (s, w) in GL4) * norm(b - a) / 2
const TRI5 = let a1 = 0.059715871789770, b1 = 0.470142064105115, a2 = 0.797426985353087, b2 = 0.101286507323456,
                 w0 = 0.225, w1 = 0.132394152788506, w2 = 0.125939180544827
    [((1 / 3, 1 / 3, 1 / 3), w0), ((a1, b1, b1), w1), ((b1, a1, b1), w1), ((b1, b1, a1), w1),
     ((a2, b2, b2), w2), ((b2, a2, b2), w2), ((b2, b2, a2), w2)]
end
"∫ f over the triangle (p1,p2,p3), with the SIGNED area"
triint(f, p1, p2, p3) = cross2(p2 - p1, p3 - p1) / 2 * sum(w * f(l[1] * p1 + l[2] * p2 + l[3] * p3) for (l, w) in TRI5)

# ---------------------------------------------------------------------------- mesh
struct PMesh
    X::Vector{Vector{Float64}}
    Tr::Vector{NTuple{3,Int}}
    Px::Float64
    Py::Float64
    E::Vector{Tuple{Int,Int}}          # edges (a < b), tangent from a to b
    Etri::Vector{Vector{Int}}          # the two triangles of each edge
    Tedges::Vector{NTuple{3,Int}}      # the three edges of each triangle
    inc::Vector{Vector{Int}}           # triangles around each vertex
    h::Float64                         # mean edge length
end
unwrap(M::PMesh, y, x) = y - [M.Px * round((y[1] - x[1]) / M.Px), M.Py * round((y[2] - x[2]) / M.Py)]

function PMesh(X, Tr, Px, Py)
    eid = Dict{Tuple{Int,Int},Int}(); E = Tuple{Int,Int}[]; Etri = Vector{Int}[]; Tedges = NTuple{3,Int}[]
    for (k, t) in enumerate(Tr)
        ids = Int[]
        for (a, b) in ((t[1], t[2]), (t[2], t[3]), (t[3], t[1]))
            key = (min(a, b), max(a, b))
            id = get(eid, key, 0)
            if id == 0
                push!(E, key); push!(Etri, Int[]); id = length(E); eid[key] = id
            end
            push!(Etri[id], k); push!(ids, id)
        end
        push!(Tedges, (ids[1], ids[2], ids[3]))
    end
    inc = [Int[] for _ in X]
    for (k, t) in enumerate(Tr), p in t
        push!(inc[p], k)
    end
    M0 = PMesh(X, Tr, Px, Py, E, Etri, Tedges, inc, 0.0)
    h = mean(norm(unwrap(M0, X[b], X[a]) - X[a]) for (a, b) in E)
    PMesh(X, Tr, Px, Py, E, Etri, Tedges, inc, h)
end

"geometry of edge e unwrapped about ref: endpoints, midpoint, tangent, normal, circumcentres c1 (-n side) and c2 (+n side)"
function egeo(M::PMesh, e, ref)
    a, b = M.E[e]; xa = unwrap(M, M.X[a], ref); xb = unwrap(M, M.X[b], xa)
    m = (xa + xb) / 2; L = norm(xb - xa); t = (xb - xa) / L; n = J * t
    cs = [circumcentre([unwrap(M, M.X[p], m) for p in M.Tr[T]]...) for T in M.Etri[e]]
    s = [dot(c - m, n) for c in cs]
    if s[1] > s[2]
        cs = reverse(cs); s = reverse(s)
    end
    (xa = xa, xb = xb, m = m, t = t, n = n, L = L, c1 = cs[1], c2 = cs[2], Ld = s[2] - s[1], δ = (s[1] + s[2]) / 2)
end

# ---------------------------------------------------------------------------- polynomial basis of [P_k]²
mons(k) = [(i, d - i) for d in 0:k for i in d:-1:0]
nbas(k) = 2length(mons(k))
"2 × N matrix: the vector monomials ((x - xc)/h)^α e_c"
function basis(x, xc, h, k)
    ξ = (x - xc) / h; ms = mons(k); B = zeros(2, 2length(ms))
    for (q, (i, j)) in enumerate(ms)
        v = ξ[1]^i * ξ[2]^j
        B[1, 2q-1] = v; B[2, 2q] = v
    end
    B
end

# ---------------------------------------------------------------------------- cells
struct Cell
    xc::Vector{Float64}
    regs::Vector{Tuple{Vector{Float64},Vector{Float64},Vector{Float64},Float64}}  # signed sub-triangles
    S::Vector{Int}                                                                # stencil edges
    segs::Vector{Tuple{Vector{Float64},Vector{Float64}}}                          # stencil edges unwrapped near xc
end
stencil_segs(M, S, xc) = [(xa = unwrap(M, M.X[M.E[e][1]], xc); (xa, unwrap(M, M.X[M.E[e][2]], xa))) for e in S]
edges_of(M, tris) = unique(vcat([collect(M.Tedges[q]) for q in tris]...))

"triangle cells; stencil = edges of the triangles that share a vertex with T (ring=1), or one more layer (ring=2)"
function tri_cells(M::PMesh; ring = 1)
    map(eachindex(M.Tr)) do T
        V = [unwrap(M, M.X[p], M.X[M.Tr[T][1]]) for p in M.Tr[T]]; xc = sum(V) / 3
        tris = unique(vcat([M.inc[p] for p in M.Tr[T]]...))
        if ring == 2
            tris = unique(vcat([M.inc[p] for q in tris for p in M.Tr[q]]...))
        elseif ring == 0
            tris = [T]
        end
        S = edges_of(M, tris); own = collect(M.Tedges[T]); S = vcat(own, setdiff(S, own))
        sg = sign(cross2(V[2] - V[1], V[3] - V[1]))
        Cell(xc, [(V[1], V[2], V[3], sg)], S, stencil_segs(M, S, xc))
    end
end

"diamond cells D_e = conv(e ∪ ẽ) (signed); stencil = edges of the 2 triangles of e and of their edge-neighbours (13 on a lattice)"
function dia_cells(M::PMesh; wide = false)
    map(eachindex(M.E)) do e
        g = egeo(M, e, M.X[M.E[e][1]])
        tris0 = M.Etri[e]
        tris1 = unique(vcat(tris0, [q for T in tris0 for e2 in M.Tedges[T] for q in M.Etri[e2]]))
        if wide
            tris1 = unique(vcat(tris1, [q for T in tris1 for e2 in M.Tedges[T] for q in M.Etri[e2]]))
        end
        S = edges_of(M, tris1); S = vcat([e], setdiff(S, [e]))
        Cell(g.m, [(g.xa, g.xb, g.c2, 1.0), (g.xa, g.xb, g.c1, -1.0)], S, stencil_segs(M, S, g.m))
    end
end

cellP(M, C, k) = vcat([transpose(segint(x -> basis(x, C.xc, M.h, k)' * ((xb - xa) / norm(xb - xa)), xa, xb)) for (xa, xb) in C.segs]...)
cellint(C, f) = sum(sg * triint(f, p1, p2, p3) for (p1, p2, p3, sg) in C.regs)
cellM(M, C, k) = (A = cellint(C, x -> (B = basis(x, C.xc, M.h, k); B' * B)); (A + A') / 2)
"mixed mass ∫_C φ_{C,j} · ψ_{e,i}, ψ_e the basis centred at the midpoint of stencil edge r"
function cellN(M, C, r, k)
    me = (C.segs[r][1] + C.segs[r][2]) / 2
    cellint(C, x -> basis(x, C.xc, M.h, k)' * basis(x, me, M.h, k))
end

# ---------------------------------------------------------------------------- cell data and the moment system
struct CellData
    P::Matrix{Float64}; Mm::Matrix{Float64}; R0::Matrix{Float64}; W::Matrix{Float64}
    Nm::Vector{Matrix{Float64}}
end
celldata(M, cells, k) = [(P = cellP(M, C, k); Mm = cellM(M, C, k);
                          CellData(P, Mm, pinv(P), nullspace(P'), [cellN(M, C, r, k) for r in eachindex(C.S)])) for C in cells]

"β_e(ψ_{e,i}) for the basis centred at m_e, all edges stacked (N per edge)"
betavec(M, geos, k) = vcat([segint(x -> basis(x, g.m, M.h, k)' * g.t, g.c1, g.c2) for g in geos]...)

"row residuals Σ_C N_{C,e}ᵀ R_C[:, e] - β_e (stacked)"
function moment_residual(M, cells, D, Rs, bβ, k)
    N = nbas(k); r = -copy(bβ)
    for (C, d, R) in zip(cells, D, Rs), (q, e) in enumerate(C.S)
        r[(e-1)*N+1:e*N] .+= d.Nm[q]' * R[:, q]
    end
    r
end

"vectors λ^κ of the translation-invariant kernels κ(x-y) of degree ≤ k (the predicted kernel of Aᵀ), grouped by degree"
function kernel_vectors(M, geos, k; rng = MersenneTwister(1))
    N = nbas(k); cols = Vector{Float64}[]; deg = Int[]
    for (i, j) in mons(k), a in 1:2, b in 1:2
        λ = zeros(N * length(geos))
        for (e, g) in enumerate(geos)
            pts = [g.m + M.h * (2rand(rng, 2) .- 1) for _ in 1:2N]
            Λ(x) = segint(y -> (z = (x - y) / M.h; zeros(2) + [a == 1, a == 2] * g.t[b] * z[1]^i * z[2]^j), g.xa, g.xb)
            A = vcat([basis(x, g.m, M.h, k) for x in pts]...); rhs = vcat([Λ(x) for x in pts]...)
            λ[(e-1)*N+1:e*N] = A \ rhs
        end
        push!(cols, λ); push!(deg, i + j)
    end
    hcat(cols...), deg
end

"""
    constrained_R(M, cells, D, geos, k; project=true)

Minimal-change left inverses R_C = P_C⁺ + Z_C W_Cᵀ satisfying the row moment conditions.  Returns (Rs, info).
If the conditions are incompatible, the part of the right-hand side along the kernel vectors is removed first
(least-squares solution) and reported in info.obstruction.
"""
function constrained_R(M, cells, D, geos, k; project = true, reg = 1e-13)
    N = nbas(k); nE = length(M.E)
    bβ = betavec(M, geos, k)
    rhs = -moment_residual(M, cells, D, [d.R0 for d in D], bβ, k)     # β - (least-squares part)
    blocks = Dict{Tuple{Int,Int},Matrix{Float64}}()
    for (C, d) in zip(cells, D)
        Mi = inv(d.Mm); MN = [Mi * Nq for Nq in d.Nm]
        for (q1, e1) in enumerate(C.S), (q2, e2) in enumerate(C.S)
            w = dot(d.W[q1, :], d.W[q2, :]); abs(w) < 1e-14 && continue
            blk = w * (d.Nm[q1]' * MN[q2])
            haskey(blocks, (e1, e2)) ? (blocks[(e1, e2)] .+= blk) : (blocks[(e1, e2)] = blk)
        end
    end
    I_, J_, V_ = Int[], Int[], Float64[]
    for ((e1, e2), blk) in blocks, j in 1:N, i in 1:N
        push!(I_, (e1 - 1) * N + i); push!(J_, (e2 - 1) * N + j); push!(V_, blk[i, j])
    end
    G = sparse(I_, J_, V_, N * nE, N * nE); G = (G + G') / 2
    Kv, kdeg = kernel_vectors(M, geos, k); Q = Matrix(qr(Kv).Q)[:, 1:size(Kv, 2)]
    obstr = Dict(d => norm(Q[:, kdeg .== d] * (Q[:, kdeg .== d]' * rhs)) / norm(rhs) for d in 0:k)
    kerG = norm(G * Kv) / (norm(G, 1) * norm(Kv))                      # ‖G λ^κ‖: is the predicted kernel a kernel?
    r = project ? rhs - Q * (Q' * rhs) : rhs
    ε = reg * maximum(abs, diag(G))
    μ = cholesky(Symmetric(G + ε * I)) \ r
    Rs = map(zip(cells, D)) do (C, d)
        Z = zeros(N, size(d.W, 2)); Mi = inv(d.Mm)
        for (q, e) in enumerate(C.S)
            Z .+= Mi * d.Nm[q] * μ[(e-1)*N+1:e*N] * d.W[q, :]'
        end
        d.R0 + Z * d.W'
    end
    res = moment_residual(M, cells, D, Rs, bβ, k)
    dev = maximum(sqrt(tr((R - d.R0)' * d.Mm * (R - d.R0)) / tr(d.R0' * d.Mm * d.R0)) for (R, d) in zip(Rs, D))
    Rs, (resid = norm(res) / norm(bβ), resid0 = norm(rhs) / norm(bβ), obstruction = obstr, kerG = kerG,
         dev = dev, G = G, Kv = Kv, rhs = rhs)
end

"R_C = P_C⁺ + Z_C W_Cᵀ from the multipliers μ (one N-vector per edge)"
function R_from_mu(cells, D, μ, N)
    map(zip(cells, D)) do (C, d)
        Z = zeros(N, size(d.W, 2)); Mi = inv(d.Mm)
        for (q, e) in enumerate(C.S)
            Z .+= Mi * d.Nm[q] * μ[(e-1)*N+1:e*N] * d.W[q, :]'
        end
        d.R0 + Z * d.W'
    end
end

"coefficients of the basis centred at `from` in the basis centred at `to` (exact, N × N)"
function shiftmat(from, to, h, k; rng = MersenneTwister(2))
    N = nbas(k); pts = [to + h * (2rand(rng, 2) .- 1) for _ in 1:N]
    vcat([basis(x, to, h, k) for x in pts]...) \ vcat([basis(x, from, h, k) for x in pts]...)
end

"""
selection operator for 'strong [P_ks]² rows + vertex patch test on [P_k]²': rows (e, i ≤ dim[P_ks]²) select the
row moment conditions, rows (vertex v, l > dim[P_ks]²) are Σ_e (d0)_{ev} T_{e,v}ᵀ (row conditions of e), i.e. the
condition (d0ᵀ(Hα(u) - β(u)))_v = 0 for the basis field u = ψ_{v,l} centred at the vertex.
"""
function select_op(M, geos, k, ks)
    N = nbas(k); Ns = nbas(ks); nE = length(M.E); I_, J_, V_ = Int[], Int[], Float64[]; row = 0
    for e in 1:nE, i in 1:Ns
        row += 1; push!(I_, row); push!(J_, (e - 1) * N + i); push!(V_, 1.0)
    end
    vedges = [Int[] for _ in M.X]
    for (e, (a, b)) in enumerate(M.E)
        push!(vedges[a], e); push!(vedges[b], e)
    end
    for v in eachindex(M.X)
        T = Dict(e => shiftmat(unwrap(M, M.X[v], geos[e].m), geos[e].m, M.h, k) for e in vedges[v])
        for l in Ns+1:N
            row += 1
            for (e, Te) in T
                sgn = M.E[e][2] == v ? 1.0 : -1.0
                for m in 1:N
                    push!(I_, row); push!(J_, (e - 1) * N + m); push!(V_, sgn * Te[m, l])
                end
            end
        end
    end
    sparse(I_, J_, V_, row, N * nE)
end

# ---------------------------------------------------------------------------- the Hodge matrix
"H = Σ_C S_Cᵀ [R_Cᵀ M_C R_C + σ_C (I-P_C R_C)ᵀ diag(|e|⁻²) (I-P_C R_C)] S_C, σ_C trace-matched (τ)"
function assemble_H(M, cells, D, Rs; τ = 1.0)
    nE = length(M.E); I_, J_, V_ = Int[], Int[], Float64[]
    for (C, d, R) in zip(cells, D, Rs)
        B = R' * d.Mm * R
        if τ > 0
            Pr = I - d.P * R; lens = [norm(b - a) for (a, b) in C.segs]
            Sm = Pr' * Diagonal(1 ./ lens .^ 2) * Pr
            tS = tr(Sm); tS > 1e-14 * tr(B) && (B += τ * tr(B) / tS * Sm)
        end
        for (q1, e1) in enumerate(C.S), (q2, e2) in enumerate(C.S)
            push!(I_, e1); push!(J_, e2); push!(V_, B[q1, q2])
        end
    end
    H = sparse(I_, J_, V_, nE, nE)
    (H + H') / 2
end

# ---------------------------------------------------------------------------- diagnostics
αvec(M, geos, u) = [segint(x -> dot(u(x), g.t), g.xa, g.xb) for g in geos]
βvec(M, geos, u) = [segint(x -> dot(u(x), g.t), g.c1, g.c2) for g in geos]
function d0mat(M)
    nE = length(M.E); I_ = vcat(1:nE, 1:nE); J_ = vcat([a for (a, b) in M.E], [b for (a, b) in M.E])
    sparse(I_, J_, vcat(-ones(nE), ones(nE)), nE, length(M.X))
end

"""
row test on local polynomials: for every edge e and every basis field u of [P_kt]² centred at m_e,
(H α(u))_e versus β_e(u), with α evaluated in the chart of e.  Returns max |error| / (|ẽ| · max|u| on the row support).
"""
function row_test(M, H, geos, kt)
    N = nbas(kt); worst = 0.0
    rows = rowvals(H); vals = nonzeros(H)
    for (e, g) in enumerate(geos)
        rng = nzrange(H, e)                                   # H symmetric: column e = row e
        ms = rows[rng]; hv = vals[rng]
        αs = zeros(length(ms), N)
        for (q, mm) in enumerate(ms)
            a, b = M.E[mm]; xa = unwrap(M, M.X[a], g.m); xb = unwrap(M, M.X[b], xa); t = (xb - xa) / norm(xb - xa)
            αs[q, :] = segint(x -> basis(x, g.m, M.h, kt)' * t, xa, xb)
        end
        βs = segint(x -> basis(x, g.m, M.h, kt)' * g.t, g.c1, g.c2)
        rad = maximum(norm(unwrap(M, M.X[M.E[mm][1]], g.m) - g.m) for mm in ms) / M.h + 1
        worst = max(worst, maximum(abs.(αs' * hv - βs)) / (g.Ld * rad^kt))
    end
    worst
end

"""
vertex patch test on [P_k]²: for every vertex v and basis field u centred at v, (d0ᵀ(Hα(u) - β(u)))_v, with α and β
evaluated in the chart of v.  Returns max |defect| / (Σ_{e∋v} |ẽ| · rad^k).
"""
function weak_test(M, H, geos, k)
    N = nbas(k); worst = 0.0; rows = rowvals(H); vals = nonzeros(H)
    vedges = [Int[] for _ in M.X]
    for (e, (a, b)) in enumerate(M.E)
        push!(vedges[a], e); push!(vedges[b], e)
    end
    for v in eachindex(M.X)
        xv = M.X[v]; tot = zeros(N); scale = 0.0; rad = 1.0
        for e in vedges[v]
            sgn = M.E[e][2] == v ? 1.0 : -1.0
            g = geos[e]; me = unwrap(M, g.m, xv); sh = me - g.m
            Hα = zeros(N)
            for idx in nzrange(H, e)
                mm = rows[idx]; a, b = M.E[mm]
                xa = unwrap(M, M.X[a], me); xb = unwrap(M, M.X[b], xa); t = (xb - xa) / norm(xb - xa)
                Hα .+= vals[idx] * segint(x -> basis(x, xv, M.h, k)' * t, xa, xb)
                rad = max(rad, norm(xa - xv) / M.h + 1)
            end
            β = segint(x -> basis(x, xv, M.h, k)' * g.t, g.c1 + sh, g.c2 + sh)
            tot .+= sgn * (Hα - β); scale += g.Ld
        end
        worst = max(worst, maximum(abs, tot) / (scale * rad^k))
    end
    worst
end

Λnorm(H) = maximum(sum(abs, H[:, e]) for e in 1:size(H, 2))
function wspec(H, w)
    S = Diagonal(1 ./ sqrt.(w)); ev = eigvals(Symmetric(Matrix(S * H * S)))
    ev[1], ev[end]
end

"Poisson: relative gradient error in the H-norm, and the linear patch defect"
function poisson(M, geos, H, p, gp)
    d0 = d0mat(M); L = d0' * H * d0; n = size(L, 1)
    Aug = [L ones(n); ones(1, n) 0.0]; F = lu(sparse(Aug))     # mean-zero constraint (L singular on constants)
    solve(b) = (F \ vcat(b, 0.0))[1:n]
    ag = αvec(M, geos, gp); bg = βvec(M, geos, gp)
    ph = solve(d0' * bg); z = d0 * ph - ag
    Hn(x) = sqrt(abs(dot(x, H * x)))
    err = Hn(z) / Hn(ag)
    a = [1.0, 0.7]; ra = d0' * (H * αvec(M, geos, x -> a) - βvec(M, geos, x -> a))
    err, sqrt(abs(dot(ra, solve(ra)))) / Hn(αvec(M, geos, x -> a))
end

"Hodge error of a smooth periodic field in the W⁻¹ norm (W = DEC weights), relative"
function flux_err(M, geos, H, u)
    w = [g.Ld / g.L for g in geos]; ε = H * αvec(M, geos, u) - βvec(M, geos, u); b = βvec(M, geos, u)
    sqrt(sum(ε .^ 2 ./ w)) / sqrt(sum(b .^ 2 ./ w))
end

# ---------------------------------------------------------------------------- meshes
"periodic triangulated lattice (equilateral), optionally mapped / randomly perturbed / periodic pattern"
function latmesh(N, kind; amp = 0.08)
    Ly = sqrt(3) / 2; h = 1 / N; idx(i, j) = mod(j, N) * N + mod(i, N) + 1
    X = [[(i + 0.5 * mod(j, 2)) * h, j * h * Ly] for j in 0:N-1 for i in 0:N-1]
    rng = MersenneTwister(N)
    pat = Dict((0, 0) => [0.10, 0.03], (1, 0) => [-0.06, 0.08], (0, 1) => [0.04, -0.09], (1, 1) => [-0.08, -0.02])
    for j in 0:N-1, i in 0:N-1
        k = idx(i, j); x = X[k]
        if kind == "mapped"
            X[k] = x + 0.012 * [sin(2pi * x[1]) * cos(2pi * x[2] / Ly), sin(2pi * (x[1] + 0.3)) * sin(2pi * x[2] / Ly)]
        elseif kind == "random"
            X[k] = x + amp * h * (2rand(rng, 2) .- 1)
        elseif kind == "pattern"
            X[k] = x + h * pat[(mod(i, 2), mod(j, 2))]
        end
    end
    T = NTuple{3,Int}[]
    for j in 0:N-1, i in 0:N-1
        if iseven(j)
            push!(T, (idx(i, j), idx(i + 1, j), idx(i, j + 1))); push!(T, (idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)))
        else
            push!(T, (idx(i, j), idx(i + 1, j + 1), idx(i, j + 1))); push!(T, (idx(i, j), idx(i + 1, j), idx(i + 1, j + 1)))
        end
    end
    PMesh(X, T, 1.0, Ly)
end

"SCVT meshes exported by gh_export_scvt.jl (plain text: Px Py / nX / X / nT / T)"
function readmesh(file)
    L = readlines(file); Px, Py = parse.(Float64, split(L[1])); nX = parse(Int, L[2])
    X = [parse.(Float64, split(L[2+i])) for i in 1:nX]; nT = parse(Int, L[3+nX])
    T = [Tuple(parse.(Int, split(L[3+nX+i]))) for i in 1:nT]
    PMesh(X, NTuple{3,Int}.(T), Px, Py)
end
