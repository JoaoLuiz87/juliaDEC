# ============================================================================
#  p2_symmetric_sdp.jl -- is there a POSITIVE symmetric star exact on [P2]^2 on the 4-ring, on SCVT meshes 2-4?
#
#      julia -t 4 scripts/stars/p2_symmetric_sdp.jl mesh2 mesh3 mesh4    (writes output/ts_data/p2_symmetric_sdp.{txt,jls})
#
#  The least-norm symmetric star (p2_symmetric_check.jl) has μ = λmin(Hd^-1/2 H Hd^-1/2) = +0.90, +0.57, -0.40 on meshes 2-4.
#  Here we maximise μ over ALL symmetric stars on the same 4-ring pattern whose rows are exact on [P2]^2:
#      H(c) = Hd + C(c),  C symmetric on the pattern,  G c = rhs  (the exactness equations of symmetric_correction)
#  The feasible set is c0 + null(G) (c0 = least-norm solution).  μ(c) is concave in c, so we maximise the smoothed
#      f_t(c) = -(1/t) log Σ_i exp(-t λ_i(S(c))),   S(c) = Hd^-1/2 H(c) Hd^-1/2,   f_t ≤ μ ≤ f_t + log(n)/t,
#  by projected gradient ascent (∇f_t = Σ π_i v_i v_iᵀ mapped to the pairs, projected onto null(G)), with t increasing.
#  Any iterate with μ > 0 is a certificate that a positive exact symmetric star exists on that pattern.
#  For the best star we also report Λ, the independent row-exactness check, the Hodge error and the Poisson error.
# ============================================================================
include(joinpath(@__DIR__, "three_stars.jl"))
empty!(lines)

"The exactness system of symmetric_correction (commuting_check.jl): G c = rhs, pairs (e, j), e ≤ j, and the least-norm c0."
function sym_system(mesh, k, r)
    n = mesh.edges.n; N = nb_of(k)
    lp = edge_lengths(mesh, true); ld = edge_lengths(mesh, false); hd = ld ./ lp
    S = [ring(mesh, e, r) for e in 1:n]
    nb = [Set(S[e]) for e in 1:n]
    for e in 1:n, j in S[e]; push!(nb[j], e); end
    nbv = [sort(collect(nb[e])) for e in 1:n]
    pid = Dict{Tuple{Int,Int},Int}()
    for e in 1:n, j in nbv[e]
        key = minmax(e, j); haskey(pid, key) || (pid[key] = length(pid) + 1)
    end
    np = length(pid); pe = zeros(Int, np); pj = zeros(Int, np)
    for (key, p) in pid; pe[p] = key[1]; pj[p] = key[2]; end
    I_, J_, V_ = Int[], Int[], Float64[]; rhs = zeros(n * N)
    for e in 1:n
        xv = mesh.edges.position[e]; x0 = _pt(xv); s = lp[e]
        q1, q2 = edge_endpoints(mesh, e, false; about = xv)
        b = seg_moments(q1, q2, x0, s, k; rotate = true)
        for j in nbv[e]
            p1_, p2_ = edge_endpoints(mesh, j, true; about = xv)
            a = seg_moments(p1_, p2_, x0, s, k)
            j == e && (b .-= hd[e] .* a)
            for m in 1:N
                push!(I_, (e - 1) * N + m); push!(J_, pid[minmax(e, j)]); push!(V_, a[m] / ld[e])
            end
        end
        rhs[(e-1)*N+1:e*N] .= b ./ ld[e]
    end
    G = sparse(I_, J_, V_, n * N, np)
    w = [pe[p] == pj[p] ? 1.0 : 2.0 for p in 1:np]
    K = G * Diagonal(1 ./ w) * G'
    δ = 1e-14 * sum(diag(K)) / size(K, 1)
    c0 = (1 ./ w) .* (G' * (cholesky(Symmetric(K + δ * I)) \ rhs))       # least-norm (Frobenius) solution, as in symmetric_correction
    return (G = G, rhs = rhs, pe = pe, pj = pj, c0 = c0, hd = hd, n = n, np = np)
end

assembleH(sys, c) = sparse(vcat(sys.pe, sys.pj[sys.pe .!= sys.pj], 1:sys.n), vcat(sys.pj, sys.pe[sys.pe .!= sys.pj], 1:sys.n),
                           vcat(c, c[sys.pe .!= sys.pj], sys.hd), sys.n, sys.n)

function scaledS(sys, c)
    n = sys.n; sq = sqrt.(sys.hd); S = Matrix(Diagonal(ones(n)))
    @inbounds for p in eachindex(c)
        e, j = sys.pe[p], sys.pj[p]; v = c[p] / (sq[e] * sq[j])
        S[e, j] += v; e != j && (S[j, e] += v)
    end
    return Symmetric(S)
end

"smoothed λmin, its gradient in c, and the true λmin; the lowest m eigenpairs suffice when the weights beyond them are negligible"
function softmin(sys, c, t; m = 120)
    S = scaledS(sys, c); n = sys.n
    while true
        F = m >= n ? eigen(S) : eigen(S, 1:min(m, n))
        λ = F.values; λ1 = λ[1]
        if m < n && t * (λ[end] - λ1) < 40; m = min(2m, n); continue; end
        z = exp.(-t .* (λ .- λ1)); Z = sum(z); π = z ./ Z
        f = λ1 - log(Z) / t
        V = F.vectors; sq = sqrt.(sys.hd)
        g = zeros(length(c))
        @inbounds for p in eachindex(c)
            e, j = sys.pe[p], sys.pj[p]; acc = 0.0
            for i in eachindex(π); acc += π[i] * V[e, i] * V[j, i]; end
            g[p] = (e == j ? acc : 2acc) / (sq[e] * sq[j])
        end
        return f, g, λ1, m
    end
end

"""
Exact projectors from a rank-revealing sparse QR of A = Gᵀ (np × neq): A[prow, pcol] = Q R, rank r.
  proj(x): orthogonal projection onto null(G) = range(A)^⊥.
  feas(c): c minus the least-norm correction Δ ∈ range(A) with G Δ = G c − rhs (the dependent equations are consistent).
"""
function exact_projectors(G)
    F = qr(sparse(G')); r = rank(F); np = size(G, 2)
    R11 = UpperTriangular(Matrix(F.R[1:r, 1:r])); prow = F.prow; pcol = F.pcol
    function proj(x)
        y = F.Q' * x[prow]; y[1:r] .= 0                     # keep the component orthogonal to range(A)
        out = similar(x); out[prow] = F.Q * y; return out
    end
    function feas(c, rhs)
        b = (G * c .- rhs)[pcol]
        v = zeros(np); v[1:r] = R11' \ b[1:r]
        Δ = similar(c); Δ[prow] = F.Q * v
        return c .- Δ
    end
    return proj, feas, r
end

"""
The smallest positive exact symmetric star:  minimise ½‖C‖_F² = ½ Σ_p w_p c_p²  over c ∈ c0 + null(G)  subject to  S(c) ⪰ μ0 I,
by a log-det barrier,  φ_τ(c) = ½ Σ w_p c_p² − (1/τ) log det(S(c) − μ0 I),  with projected gradient descent (BB steps, backtracking
that keeps S − μ0 I positive definite) and τ = n/gap for decreasing duality gaps.  Starts from a strictly feasible c.
"""
function min_norm_positive(sys, cstart, μ0, proj, feas; gaps = (0.1, 0.01, 1e-3), iters = 150, tag = "")
    w = [sys.pe[p] == sys.pj[p] ? 1.0 : 2.0 for p in 1:sys.np]; sq = sqrt.(sys.hd); n = sys.n
    function φ(c, τ)
        X = Matrix(scaledS(sys, c)) - μ0 * I
        Ch = cholesky(Symmetric(X); check = false)
        issuccess(Ch) || return Inf, nothing
        val = 0.5 * sum(w .* c .^ 2) - logdet(Ch) / τ
        Xi = inv(Ch)
        g = w .* c
        @inbounds for p in eachindex(c)
            e, j = sys.pe[p], sys.pj[p]
            g[p] -= (e == j ? Xi[e, e] : 2Xi[e, j]) / (sq[e] * sq[j]) / τ
        end
        return val, g
    end
    c = copy(cstart)
    for gap in gaps
        τ = n / gap; f, g = φ(c, τ); d = proj(g); α = 1.0 / max(norm(d), 1e-12); dold = d; cold = c
        for it in 1:iters
            nd = norm(d); nd < 1e-9 && break
            while true
                cn = c .- α .* d; fn, gn = φ(cn, τ)
                if fn < f - 1e-4 * α * nd^2
                    s = cn .- c; dn = proj(gn); y = dn .- d
                    c, f, g, cold, dold, d = cn, fn, gn, c, d, dn
                    sy = dot(s, y); α = sy > 0 ? dot(s, s) / sy : 2α          # Barzilai-Borwein step
                    break
                end
                α *= 0.3; α * nd < 1e-14 && break
            end
            if it % 50 == 0
                c = feas(c); f, g = φ(c, τ); d = proj(g)
            end
            α * nd < 1e-14 && break
        end
        c = feas(c)
        μ = μof(sys, c)
        logp(@sprintf("   %s μ0 = %.2f, gap %.0e: ‖C‖_F = %.4f, μ = %+.4f", tag, μ0, gap, sqrt(sum(w .* c .^ 2)), μ))
    end
    return c
end

function maximise_mu(sys; ts = (20.0, 50.0, 120.0, 300.0), iters = 150, tag = "", μstop = Inf)
    G = sys.G; projQ, feasQ, r = exact_projectors(G)
    logp(@sprintf("   %s QR of Gᵀ: rank %d of %d equations (%d dependent), null space of dimension %d", tag, r, size(G, 1), size(G, 1) - r, sys.np - r))
    proj(g) = projQ(g)
    feas(c) = feasQ(c, sys.rhs)
    c = feas(copy(sys.c0)); best = (μ = -Inf, c = c); m = 120
    logp(@sprintf("   %s start: constraint residual %.1e after the exact correction", tag, norm(G * c .- sys.rhs) / norm(sys.rhs)))
    for t in ts
        f, g, μ, m = softmin(sys, c, t; m = m); α = 0.05 / max(norm(proj(g)), 1e-12)
        for it in 1:iters
            d = proj(g); nd = norm(d)
            nd < 1e-10 && break
            while true
                cn = c .+ α .* d
                fn, gn, μn, m = softmin(sys, cn, t; m = m)
                if fn > f + 1e-4 * α * nd^2
                    c, f, g, μ = cn, fn, gn, μn; α *= 1.6; break
                end
                α *= 0.4; α * nd < 1e-12 && break
            end
            μ > best.μ && (best = (μ = μ, c = copy(c)))
            if best.μ >= μstop
                bc = feas(best.c); bμ = μof(sys, bc)
                if bμ >= μstop
                    logp(@sprintf("   %s reached μ = %+.4f ≥ %.2f at t = %.0f, it %d: a strictly feasible start for the barrier", tag, bμ, μstop, t, it))
                    return (μ = bμ, c = bc, proj = proj, feas = feas, converged = false)
                end
            end
            it % 25 == 0 && (c = feas(c))
            it % 25 == 0 && (logp(@sprintf("   %s t = %5.0f  it %3d  f_t %+.4f  μ %+.4f  |Pg| %.2e  step %.1e", tag, t, it, f, μ, nd, α * nd)); flush(stdout))
            α * nd < 1e-12 && break
        end
        c = feas(c); f, g, μ, m = softmin(sys, c, t; m = m)
        μ > best.μ && (best = (μ = μ, c = copy(c)))
        logp(@sprintf("   %s end of t = %5.0f: μ %+.4f (upper estimate of μ* from this smoothing: f_t + log(n)/t = %+.4f)", tag, t, μ, f + log(sys.n) / t))
    end
    return (μ = best.μ, c = best.c, proj = proj, feas = feas, converged = true)
end

# independent row-exactness check (as in p2_symmetric_check.jl)
const GL5X = [-0.9061798459386640, -0.5384693101056831, 0.0, 0.5384693101056831, 0.9061798459386640]
const GL5W = [0.2369268850561891, 0.4786286704993665, 0.5688888888888889, 0.4786286704993665, 0.2369268850561891]
segint(Fn, a, b, τ) = norm(b .- a) * sum(w / 2 * dot(Fn(a .+ (ξ + 1) / 2 .* (b .- a)), τ) for (ξ, w) in zip(GL5X, GL5W))
vmonos(k, x0, s) = vcat([x -> [((x[1] - x0[1]) / s)^i * ((x[2] - x0[2]) / s)^(k - i), 0.0] for i in k:-1:0],
                        [x -> [0.0, ((x[1] - x0[1]) / s)^i * ((x[2] - x0[2]) / s)^(k - i)] for i in k:-1:0])
function row_exactness(mesh, H, k, rows)
    Ht = sparse(H'); worst = 0.0
    for e in rows
        xe = mesh.edges.position[e]; x0 = _pt(xe)
        a, b = edge_endpoints(mesh, e, true; about = xe); s = norm(b .- a); te = (b .- a) ./ s
        v1, v2 = edge_endpoints(mesh, e, false; about = xe); le = norm(v2 .- v1)
        for Fn in vcat(vmonos(0, x0, s), vmonos(1, x0, s), vmonos(k, x0, s))
            lhs = 0.0
            for p in nzrange(Ht, e)
                j = rowvals(Ht)[p]; q1, q2 = edge_endpoints(mesh, j, true; about = xe)
                lhs += nonzeros(Ht)[p] * segint(Fn, q1, q2, (q2 .- q1) ./ norm(q2 .- q1))
            end
            worst = max(worst, abs(lhs - segint(Fn, v1, v2, te)) / le)
        end
    end
    return worst
end
μof(sys, c) = eigen(scaledS(sys, c), 1:1).values[1]

f_out = joinpath(TSDIR, "p2_symmetric_sdp.jls")
R = isfile(f_out) ? deserialize(f_out) : Dict{Any,Any}()
logp("Positive symmetric stars exact on [P2]^2 on the 4-ring: maximise μ = λmin(Hd^-1/2 H Hd^-1/2) over the exact symmetric stars")
for mn in (isempty(ARGS) ? ["mesh2", "mesh3", "mesh4"] : ARGS)
    mesh = VoronoiMesh("meshes/$mn.nc"); t0 = time()
    sys = sym_system(mesh, 2, 4)
    logp(@sprintf("%s: %d edges, %d pair unknowns, %d equations; least-norm star μ = %+.4f", mn, sys.n, sys.np, size(sys.G, 1), μof(sys, sys.c0)))
    best = maximise_mu(sys; tag = mn, μstop = 0.6)
    logp(@sprintf("   %s phase A: largest μ found %+.4f%s", mn, best.μ, best.converged ? " (optimiser ran to the end)" : " (stopped early)"))
    cands = [("least-norm", sys.c0)]
    μln = μof(sys, sys.c0)
    for μ0 in (0.0, 0.25, 0.5)
        μ0 < best.μ - 0.02 || continue
        if μln > μ0 + 1e-6
            logp(@sprintf("   %s μ0 = %.2f: the least-norm star already has μ = %+.4f ≥ μ0, so it is the smallest one", mn, μ0, μln)); continue
        end
        c = min_norm_positive(sys, best.c, μ0, best.proj, best.feas; tag = mn)
        push!(cands, (@sprintf("min-norm μ≥%.2f", μ0), c))
    end
    push!(cands, ("max-μ", best.c))
    wts = [sys.pe[p] == sys.pj[p] ? 1.0 : 2.0 for p in 1:sys.np]; Cln = sqrt(sum(wts .* sys.c0 .^ 2))
    for (lab, c) in cands
        H = assembleH(sys, c)
        res = norm(sys.G * c .- sys.rhs) / norm(sys.rhs)
        rows = sys.n <= 3072 ? (1:sys.n) : (1:7:sys.n)
        ex = row_exactness(mesh, H, 2, rows)
        Λ = bigLambda(mesh, H); μ = μof(sys, c)
        ed = hodge_error(mesh, H, HFIELDS.d...); eu = hodge_error(mesh, H, HFIELDS.u2...)
        pe = try poisson_errors(mesh, H).l2 catch; NaN end
        CF = sqrt(sum(wts .* c .^ 2))
        R[(mn, lab)] = (μ = μ, Λ = Λ, res = res, exact = ex, hodge = ed.l2, hinf = ed.inf + eu.inf, poisson = pe,
                        cmax = maximum(abs, c) / maximum(sys.hd), CF = CF, CFrel = CF / Cln)
        logp(@sprintf("   %-16s μ %+.4f  ‖C‖_F/‖C_ln‖_F %6.2f  Λ %7.2f  constraint res %.1e  row exactness %.1e  r_2(d) %.2e  r∞(u2)+r∞(d) %.2e  Poisson L2 %.2e",
                      lab, μ, CF / Cln, Λ, res, ex, ed.l2, ed.inf + eu.inf, pe))
    end
    logp(@sprintf("   (%s done in %.0f s)", mn, time() - t0))
    serialize(f_out, R); write(joinpath(TSDIR, "p2_symmetric_sdp.txt"), join(lines, "\n") * "\n")
end
