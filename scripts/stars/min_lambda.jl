# ============================================================================
#  min_lambda.jl -- the SMALLEST possible Λ = max_e (1/|ẽ_e|) Σ_j |H_ej| |e_j| over ALL symmetric stars exact on [P_k]² on a given pattern (ring r).
#  Iteratively reweighted least squares:  minimise Σ_e λ_e Σ_j w_ej H_ej²/(|H_ej^old|+η),  w_ej = |e_j|/|ẽ_e|,  over  H = H_DEC + C,  G c = rhs;
#  λ (a probability over the rows) by multiplicative weights towards the rows of largest Σ_j w_ej|H_ej|.
#  CERTIFICATE (weak duality).  If Z = (Z_ej) satisfies  BᵀZ ∈ row space(G)  (B: pairs -> directed entries), then ⟨Z,H⟩ is the same for every feasible H, and
#      ⟨Z,H⟩ ≤ Σ_e (Σ_j |H_ej||e_j|) max_j |Z_ej|/|e_j| ≤ Λ(H) Σ_e |ẽ_e| max_j |Z_ej|/|e_j| ,
#  so every symmetric exact star on the pattern has Λ ≥ ⟨Z,H⟩ / Σ_e |ẽ_e| max_j |Z_ej|/|e_j|.   Z = Ω∘H of each weighted least-squares solve qualifies.
#      julia -t 4 scripts/stars/min_lambda.jl k r mesh ...        -> appends to ts_data/min_lambda.txt ;  ENV ITS = number of iterations
# ============================================================================
include(joinpath(@__DIR__, "three_stars.jl")); empty!(lines)
src_sdp = read(joinpath(@__DIR__, "p2_symmetric_sdp.jl"), String)
for nm in ("function sym_system", "assembleH(sys, c)")
    i = findfirst(nm, src_sdp)[1]; j = nm == "assembleH(sys, c)" ? findnext("\n\n", src_sdp, i)[1] : findnext("\nend\n", src_sdp, i)[end]
    include_string(Main, src_sdp[i:j])
end
function irls_lambda(mn, k, r; iters = 60, η0 = 1e-2, verbose = true)
    mesh = VoronoiMesh("meshes/$mn.nc"); sys = sym_system(mesh, k, r); lp = edge_lengths(mesh, true); ld = edge_lengths(mesh, false); n = sys.n; np = sys.np; G = sys.G; Gt = sparse(G')
    I_ = Int[]; J_ = Int[]; P_ = Int[]
    for p in 1:np
        push!(I_, sys.pe[p]); push!(J_, sys.pj[p]); push!(P_, p)
        sys.pe[p] != sys.pj[p] && (push!(I_, sys.pj[p]); push!(J_, sys.pe[p]); push!(P_, p))
    end
    h0 = [I_[q] == J_[q] ? sys.hd[I_[q]] : 0.0 for q in eachindex(I_)]; wr = [lp[J_[q]] / ld[I_[q]] for q in eachindex(I_)]
    rowsum(Y) = (s = zeros(n); for q in eachindex(Y); s[I_[q]] += wr[q] * abs(Y[q]); end; s)
    dualnorm(Z) = (m = zeros(n); for q in eachindex(Z); m[I_[q]] = max(m[I_[q]], abs(Z[q]) / lp[J_[q]]); end; sum(ld .* m))
    H(c) = h0 .+ c[P_]
    function solve(W, Ωh)                                        # min ½ cᵀWc + cᵀΩh  s.t. G c = rhs
        Wi = 1 ./ W; K = G * Diagonal(Wi) * Gt; δ = 1e-13 * sum(diag(K)) / size(K, 1); F = cholesky(Symmetric(K + δ * I))
        y = zeros(size(G, 1)); b = sys.rhs .+ G * (Wi .* Ωh)
        for _ in 1:6; y .+= F \ (b .- K * y); end
        c = Wi .* (Gt * y .- Ωh)
        return c, y
    end
    c = copy(sys.c0); Y = H(c); λ = fill(1 / n, n); best = Inf; lower = -Inf; cb = copy(c)
    for it in 1:iters
        η = η0 * 0.9^it + 1e-5; a = abs.(Y); sc = maximum(a); Ω = [λ[I_[q]] * wr[q] / (a[q] + η * sys.hd[I_[q]]) for q in eachindex(Y)]
        W = zeros(np); for q in eachindex(Ω); W[P_[q]] += Ω[q]; end
        Ωh = zeros(np); for q in eachindex(Ω); Ωh[P_[q]] += Ω[q] * h0[q]; end
        c, y = solve(W, Ωh); Y = H(c); rs = rowsum(Y); Λ = maximum(rs)
        Z = Ω .* Y; val = dot(Z, Y); val2 = dot(Z, h0) + dot(y, sys.rhs); lb = min(val, val2) / dualnorm(Z)          # certificate (BᵀZ = Gᵀy to the residual of the solve)
        res = norm(G * c .- sys.rhs) / norm(sys.rhs); bz = zeros(np); for q in eachindex(Z); bz[P_[q]] += Z[q]; end; kkt = norm(bz .- Gt * y) / norm(bz)
        (res < 1e-7 && Λ < best) && (best = Λ; cb = copy(c)); (res < 1e-7 && kkt < 1e-6) && (lower = max(lower, lb))
        λ .*= exp.(0.5 .* (rs ./ Λ .- 1)) .* (rs ./ Λ) .^ 0.5; λ .= max.(λ, 1e-6 / n * maximum(λ)); λ ./= sum(λ)
        verbose && (it % 10 == 0 || it == 1) && (println(@sprintf("   %s k=%d r=%d it %3d  Λ %.4f  best %.4f  lower bound %.4f   res %.1e  kkt %.1e   |val−val2|/val %.1e", mn, k, r, it, Λ, best, lower, res, kkt, abs(val - val2) / val)); flush(stdout))
        best - lower < 2e-3 * best && break
    end
    Λ0 = maximum(rowsum(H(sys.c0)))
    s = @sprintf("%s  k=%d  ring %d  (%d edges, %d pairs, %d equations)  least-change star Λ %.3f   smallest Λ: feasible %.3f   certified lower bound %.3f", mn, k, r, n, np, size(G, 1), Λ0, best, lower)
    println(s); flush(stdout); open(io -> println(io, s), joinpath(TSDIR, "min_lambda.txt"), "a")
    return best, lower, assembleH(sys, cb), mesh, sys
end
if abspath(PROGRAM_FILE) == @__FILE__
    k = parse(Int, ARGS[1]); r = parse(Int, ARGS[2]); its = haskey(ENV, "ITS") ? parse(Int, ENV["ITS"]) : 60
    for mn in ARGS[3:end]; irls_lambda(mn, k, r; iters = its); end
end
