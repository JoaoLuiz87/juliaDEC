# ============================================================================
#  p2_min_asym_check.jl -- how symmetric can a star exact on [P2]^2 be made?  (and a local P1-exact one)
#
#      julia -t 4 scripts/stars/p2_min_asym_check.jl        ->  output/ts_data/p2_min_asym.{txt,jls}
#
#  On the symmetrised 3-level pattern let A_k = {C : rows of H_DEC + C exact on [P_k]^2} and S = {C = Cᵀ}.
#  Alternating projections E ← P_A P_S E from E_0 = P_A 0 stay in A_k (every iterate is exact, row by row) and move
#  towards S: they converge to a point of A_k closest to S, i.e. to an exact star of least ‖H − Hᵀ‖_F on that pattern.
#  For k = 1 the limit is symmetric (H_P1).  For k = 2 the barrier says a symmetric limit would need coefficients ~ 1/h,
#  so the question is how far the asymmetry can be reduced while the coefficients stay bounded.
#  E_0 is the row-wise least-change correction on this pattern: for k = 2 it is H_P2 (on 23 instead of 19 entries), for
#  k = 1 the LOCAL star exact on [P1]^2.
#
#  Per sweep count: ‖H−Hᵀ‖_F/‖H‖_F, operator norm of Hd^-1/2 H_a Hd^-1/2, largest |C| entry / largest DEC entry, Λ,
#  Gershgorin g, Hodge error of the field d; on meshes 3-4 also the Laplacian spectrum (max|Im λ|/λmax, share of
#  non-real eigenvalues, wave growth γ).
# ============================================================================
ENV["LS_MAIN"] = "0"
include(joinpath(@__DIR__, "local_symmetric_check.jl"))
empty!(OUT)
const RM = Dict{Any,Any}()
const KM = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
"‖B‖₂ by power iteration on BᵀB."
function opn(B; iters = 200)
    x = randn(size(B, 2)); x ./= norm(x); s = 0.0
    for _ in 1:iters
        y = B' * (B * x); s = norm(y); s == 0 && return 0.0; x = y ./ s
    end
    return sqrt(s)
end

for mn in ("mesh3", "mesh4", "mesh5"), k in (2, 1)
    mesh = VoronoiMesh("meshes/$mn.nc"); S = cached_stars(mn; mesh); Hd = hdiag(mesh); hd = diag(Hd); Dm = Diagonal(1 ./ sqrt.(hd))
    D = p1_rows(mesh; k = k); n = D.n; N = mesh.cells.n; d0 = sparse(Float64.(d0_matrix_primal(mesh))); A = mesh.cells.area
    logp(@sprintf("\n%s, exact on [P%d]^2 (%d conditions per row, %.1f entries per row)", mn, k, nb_of(k), sum(length, D.nbv) / n))
    logp("   sweeps  ‖H−Hᵀ‖_F/‖H‖_F  ‖Hd^-½H_aHd^-½‖₂  max|C|/max Hd    Λ      g     Hodge r∞(d)   r2(d)    row residual" *
         (mesh.cells.n <= 1024 ? "   max|Im λ|/λmax  non-real   wave γ" : ""))
    rows = project_rows(D, [zeros(length(D.nbv[e])) for e in 1:n])
    for it in 0:maximum(KM)
        if it in KM
            C = tosparse(D, rows); H = Hd + C; he = hodge_error(mesh, H, fields.d, supn.d)
            res = maximum(maximum(abs, x) for x in residual(D, rows))
            q = (asymF = norm(H - H') / norm(H), asym2 = opn(Dm * ((H - H') / 2) * Dm), cmax = maximum(abs.(nonzeros(C))) / maximum(hd),
                 Lam = bigLambda(mesh, H), g = gersh_sym(mesh, H), rinf = he.inf, r2 = he.l2, res = res)
            s = @sprintf("   %5d     %.3e        %.3e        %.3f      %5.2f  %+.3f   %.2e    %.2e   %.0e", it, q.asymF, q.asym2, q.cmax, q.Lam, q.g, q.rinf, q.r2, q.res)
            if N <= 1024
                λ = eigvals(Matrix(Diagonal(1 ./ A) * (d0' * H * d0))); λmax = maximum(abs.(λ))
                imr = maximum(abs.(imag.(λ))) / λmax; frac = count(abs.(imag.(λ)) .> 1e-9 * λmax) / N
                γ = maximum(abs.(imag.(sqrt.(complex.(λ))))) / sqrt(λmax)
                q = merge(q, (im = imr, frac = frac, gamma = γ, minre = minimum(real.(λ)) / λmax))
                s *= @sprintf("      %.2e      %4.1f%%    %.1e", imr, 100frac, γ)
            end
            RM[(mn, k, it)] = q; logp(s)
        end
        it < maximum(KM) && (rows = project_rows(D, symmetrize(D, rows)))
    end
    serialize(joinpath(@__DIR__, "..", "..", "output", "ts_data", "p2_min_asym.jls"), RM)
end
write(joinpath(@__DIR__, "..", "..", "output", "ts_data", "p2_min_asym.txt"), join(OUT, "\n") * "\n")
