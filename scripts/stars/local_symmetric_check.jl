# ============================================================================
#  local_symmetric_check.jl -- how far do "symmetric" and "local" go together?
#
#      julia -t 6 scripts/stars/local_symmetric_check.jl        ->  output/ts_data/local_symmetric.{txt,jls}
#
#  H_P1 = H_DEC + C is the Frobenius projection of 0 onto  A ∩ S, with
#       A = {C on the pattern : every row of H_DEC + C exact on [P1]^2}      (row by row: LOCAL)
#       S = {C on the pattern : C = Cᵀ}                                       (entry pairs: LOCAL)
#  Both are affine/linear subspaces, so alternating projections C ← P_S P_A C from C = 0 converge to that projection,
#  i.e. to H_P1 (von Neumann).  One sweep:
#       P_A: row e  ←  row e + A_e (A_eᵀA_e)^-1 (b_e − A_eᵀ row e)            (smallest change making row e exact)
#       P_S: C ← ½ (C + Cᵀ)
#  After k sweeps the star is symmetric, has the SAME stencil as H_P1 (23 entries per row), and its coefficients depend
#  on the mesh only within k stencil radii: a local rule of radius ~ k (Definition "local rule" of barrier_hodge).
#  By Theorem "local" it cannot be exact; this script measures how the P1 defect decays with k.
#  In multiplier form the sweep is the block-Jacobi Richardson iteration for K λ = r, K = G W^-1 Gᵀ; "PCG" below is
#  the conjugate-gradient acceleration with the same local preconditioner (same reach per step, but its step lengths
#  are global inner products, so it is not a local rule in the strict sense).
#
#  Reported for the symmetric iterates C_k:  P1 defect (max and rms over rows and the 4 linear basis fields, relative
#  to H_DEC's), distance to H_P1, Hodge error of the smooth field d, Laplacian truncation, Gershgorin bound g.
#  For the exact iterates P_A C_k (exact on [P1]^2, local, not symmetric): ‖H − Hᵀ‖_F/‖H‖_F.
# ============================================================================
isdefined(Main, :cached_stars) || include(joinpath(@__DIR__, "stars_cache.jl"))
using Printf, LinearAlgebra, SparseArrays, Serialization
const OUT = String[]
logp(s) = (println(s); flush(stdout); push!(OUT, s))
const LSM = haskey(ENV, "LS_MESHES") ? Tuple(String.(split(ENV["LS_MESHES"], ","))) : ("mesh3", "mesh4", "mesh5", "mesh6")
const KS = (0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512)
R = Dict{Any,Any}()

"Rows of the P1 exactness system on the symmetrised 3-level pattern (same scaling as symmetric_correction)."
function p1_rows(mesh; k = 1, r = 3)
    n = mesh.edges.n; N = nb_of(k)
    lp = edge_lengths(mesh, true); ld = edge_lengths(mesh, false); hd = ld ./ lp
    S = [ring(mesh, e, r) for e in 1:n]
    nb = [Set(S[e]) for e in 1:n]
    for e in 1:n, j in S[e]; push!(nb[j], e); end
    nbv = [sort(collect(nb[e])) for e in 1:n]
    Am = Vector{Matrix{Float64}}(undef, n); br = Vector{Vector{Float64}}(undef, n); Pi = Vector{Matrix{Float64}}(undef, n)
    Di = Vector{Matrix{Float64}}(undef, n)
    Threads.@threads for e in 1:n
        xv = mesh.edges.position[e]; x0 = _pt(xv); s = lp[e]
        q1, q2 = edge_endpoints(mesh, e, false; about = xv)
        b = seg_moments(q1, q2, x0, s, k; rotate = true)
        A = zeros(length(nbv[e]), N)
        for (i, j) in enumerate(nbv[e])
            p1_, p2_ = edge_endpoints(mesh, j, true; about = xv)
            A[i, :] = seg_moments(p1_, p2_, x0, s, k)
            j == e && (b .-= hd[e] .* A[i, :])
        end
        Am[e] = A ./ ld[e]; br[e] = b ./ ld[e]
        Di[e] = inv(Symmetric(Am[e]' * Am[e])); Pi[e] = Am[e] * Di[e]            # least-norm solution map of A_eᵀ c = y
    end
    pos = [[searchsortedfirst(nbv[j], e) for j in nbv[e]] for e in 1:n]
    return (n = n, nbv = nbv, A = Am, b = br, Pi = Pi, Di = Di, pos = pos, hd = hd)
end
symmetrize(D, rows) = [[(rows[e][i] + rows[j][D.pos[e][i]]) / 2 for (i, j) in enumerate(D.nbv[e])] for e in 1:D.n]
residual(D, rows) = [D.b[e] .- D.A[e]' * rows[e] for e in 1:D.n]
project_rows(D, rows) = (r = residual(D, rows); [rows[e] .+ D.Pi[e] * r[e] for e in 1:D.n])
tosparse(D, rows) = sparse(reduce(vcat, [fill(e, length(D.nbv[e])) for e in 1:D.n]), reduce(vcat, D.nbv), reduce(vcat, rows), D.n, D.n)
"K λ for λ = (λ_e): rows A_e λ_e, symmetrised, then A_eᵀ row."
applyK(D, λ) = (rows = symmetrize(D, [D.A[e] * λ[e] for e in 1:D.n]); ([D.A[e]' * rows[e] for e in 1:D.n], rows))
vdot(a, b) = sum(dot(a[e], b[e]) for e in eachindex(a))

function report!(key, D, mesh, Hd, C1, rows, r0; exactrows = nothing)
    r = residual(D, rows); C = tosparse(D, rows); H = Hd + C
    lin = 3:6                                                          # the four linear basis fields (1:2 are the constants)
    dmax = maximum(maximum(abs.(x[lin])) for x in r); drms = sqrt(sum(sum(abs2, x[lin]) for x in r) / (4D.n))
    he = hodge_error(mesh, H, fields.d, supn.d)
    asym = exactrows === nothing ? NaN : (Ce = tosparse(D, exactrows); norm(Ce - Ce') / norm(Hd + Ce))
    R[key] = (dmax = dmax / r0.dmax, drms = drms / r0.drms, dist = norm(C - C1) / norm(C1), rinf = he.inf, r2 = he.l2,
              tau = laplacian_truncation(mesh, H), g = gersh_sym(mesh, H), sym = norm(C - C') / max(norm(C), 1e-300), asymA = asym,
              cmax = maximum(abs.(nonzeros(C)); init = 0.0) / maximum(D.hd))
    return R[key]
end

if get(ENV, "LS_MAIN", "1") == "1"                  # LS_MAIN=0 only loads the functions
for mn in LSM
    mesh = VoronoiMesh("meshes/$mn.nc"); S = cached_stars(mn; mesh); Hd = hdiag(mesh); C1 = S["H_P1"] - Hd
    t0 = time(); D = p1_rows(mesh); n = D.n
    rows0 = [zeros(length(D.nbv[e])) for e in 1:n]; r = residual(D, rows0)
    r0 = (dmax = maximum(maximum(abs.(x[3:6])) for x in r), drms = sqrt(sum(sum(abs2, x[3:6]) for x in r) / (4n)))
    logp(@sprintf("\n%s: %d edges, h = 1/%d  (rows built in %.0f s).  P1 defect of H_DEC: max %.3f, rms %.3f (flux error on a linear field with unit variation per edge length, relative to |ẽ|)",
                  mn, n, round(Int, sqrt(mesh.cells.n)), time() - t0, r0.dmax, r0.drms))
    logp(@sprintf("   reference H_P1 (global solve): Hodge r∞(d) %.2e r2 %.2e  tau %.2e  g %.3f;   H_DEC: r∞ %.2e r2 %.2e tau %.2e",
                  hodge_error(mesh, S["H_P1"], fields.d, supn.d).inf, hodge_error(mesh, S["H_P1"], fields.d, supn.d).l2, laplacian_truncation(mesh, S["H_P1"]),
                  gersh_sym(mesh, S["H_P1"]), hodge_error(mesh, Hd, fields.d, supn.d).inf, hodge_error(mesh, Hd, fields.d, supn.d).l2, laplacian_truncation(mesh, Hd)))
    # ---------------- alternating projections (block-Jacobi Richardson): a strictly local rule
    logp("   alternating projections  C ← P_S P_A C   (k sweeps; symmetric iterate)")
    logp("      k   defect max   rms    |C_k−C_P1|/|C_P1|   Hodge r∞(d)   r2(d)      tau     g      | exact iterate P_A C_k: ‖H−Hᵀ‖/‖H‖")
    rows = rows0; t0 = time()
    for k in 0:maximum(KS)
        if k in KS
            ex = project_rows(D, rows)
            q = report!((mn, :ap, k), D, mesh, Hd, C1, rows, r0; exactrows = ex)
            logp(@sprintf("   %4d   %.2e  %.2e      %.2e        %.2e    %.2e  %.2e  %.3f   |  %.2e", k, q.dmax, q.drms, q.dist, q.rinf, q.r2, q.tau, q.g, q.asymA))
        end
        k < maximum(KS) && (rows = symmetrize(D, project_rows(D, rows)))
    end
    logp(@sprintf("      (%d sweeps in %.1f s, including the diagnostics)", maximum(KS), time() - t0))
    # ---------------- PCG on K λ = r with the block-Jacobi preconditioner D^-1 = blockdiag((A_eᵀA_e)^-1)
    logp("   preconditioned conjugate gradients on K λ = r (same reach per step; global step lengths)")
    logp("      k   defect max   rms    |C_k−C_P1|/|C_P1|   Hodge r∞(d)   r2(d)      tau     g")
    λ = [zeros(6) for _ in 1:n]; res = [copy(x) for x in D.b]; z = [D.Di[e] * res[e] for e in 1:n]; p = [copy(x) for x in z]
    rz = vdot(res, z); rows = rows0; t0 = time()
    for k in 0:maximum(KS)
        if k in KS
            q = report!((mn, :cg, k), D, mesh, Hd, C1, rows, r0)
            logp(@sprintf("   %4d   %.2e  %.2e      %.2e        %.2e    %.2e  %.2e  %.3f", k, q.dmax, q.drms, q.dist, q.rinf, q.r2, q.tau, q.g))
            q.drms < 1e-10 && break
        end
        Kp, rowsp = applyK(D, p); α = rz / vdot(p, Kp)
        rows = [rows[e] .+ α .* rowsp[e] for e in 1:n]
        for e in 1:n; λ[e] .+= α .* p[e]; res[e] .-= α .* Kp[e]; end
        z = [D.Di[e] * res[e] for e in 1:n]; rz2 = vdot(res, z); βc = rz2 / rz; rz = rz2
        for e in 1:n; p[e] .= z[e] .+ βc .* p[e]; end
    end
    serialize(joinpath(@__DIR__, "..", "..", "output", "ts_data", "local_symmetric.jls"), R)
end
write(joinpath(@__DIR__, "..", "..", "output", "ts_data", "local_symmetric.txt"), join(OUT, "\n") * "\n")
end   # LS_MAIN
