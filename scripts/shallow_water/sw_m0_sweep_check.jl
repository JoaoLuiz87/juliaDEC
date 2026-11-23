# ============================================================================
#  sw_m0_sweep_check.jl -- the sweeps of local_symmetric_check.jl applied to the corrected 0-form star M0 = A + C0
#
#      julia -t 6 scripts/shallow_water/sw_m0_sweep_check.jl        ->  output/ts_data/sw_m0_sweep.{txt,jls}
#
#  C0 (mass0_corrected of sw_linear_check.jl): row c is the smallest correction on the 2-ring of cells that makes
#  point values -> cell integrals exact on P3 (10 conditions, about 19 cells).  It is not symmetric.
#  One sweep: C0 <- (C0 + C0')/2, then every row is made exact again with the least change.  The 2-ring relation is
#  symmetric, so the pattern does not change.  "t sweeps" below.
#  Part 1: the star itself: asymmetry, smallest eigenvalue of A^-1/2 sym(M0) A^-1/2, row residual.
#  Part 2: spectrum of the wave operator M0^-1 D0' H D0 with H = DEC, and H_P2 after 1 and 4 sweeps.
#  Part 3: accuracy at t = 1 (RK4, dt = 1e-3), H = H_P2 after 1 sweep.
# ============================================================================
ENV["LS_MAIN"] = "0"; include(joinpath(@__DIR__, "..", "stars", "local_symmetric_check.jl"))
ENV["SW_MAIN"] = "0"; include(joinpath(@__DIR__, "sw_linear_check.jl"))
using Printf, LinearAlgebra, SparseArrays, Serialization
BLAS.set_num_threads(max(1, Threads.nthreads()))
const LG = String[]
say(s) = (println(s); flush(stdout); push!(LG, s))
RM0 = Dict{Any,Any}()

"Row data of the P_k exactness of the 0-form star: stencils (2-ring of cells), condition matrices Q_c, targets b_c − A_c Q_c[:,c]."
function m0_rows(mesh; k = 3)
    nc = mesh.cells.n; A = mesh.cells.area; Lx, Ly = mesh.x_period, mesh.y_period
    nbr = [Set{Int}() for _ in 1:nc]
    for e in 1:mesh.edges.n
        c1, c2 = mesh.edges.cells[e]; push!(nbr[c1], c2); push!(nbr[c2], c1)
    end
    mons = [(i, d - i) for d in 0:k for i in d:-1:0]; bq, wq = tri_rule(5)
    S = Vector{Vector{Int}}(undef, nc); Q = Vector{Matrix{Float64}}(undef, nc); b = Vector{Vector{Float64}}(undef, nc); P = Vector{Matrix{Float64}}(undef, nc)
    for c in 1:nc
        x0 = _pt(mesh.cells.position[c]); s = sqrt(A[c])
        r1 = union(Set([c]), nbr[c]); S[c] = sort(collect(union(r1, (nbr[j] for j in r1)...)))
        mono(p) = [((p[1] - x0[1]) / s)^α[1] * ((p[2] - x0[2]) / s)^α[2] for α in mons]
        bc = zeros(length(mons))
        for (q1, q2, q3) in voronoi_cell_triangles(mesh, c)
            Ar = _tri_area(q1, q2, q3)
            for (bb, ww) in zip(bq, wq); bc .+= (ww * Ar) .* mono(bb[1] .* q1 .+ bb[2] .* q2 .+ bb[3] .* q3); end
        end
        Q[c] = reduce(hcat, [mono(_pt(closest(mesh.cells.position[c], mesh.cells.position[j], Lx, Ly))) for j in S[c]])
        b[c] = bc .- A[c] .* Q[c][:, findfirst(==(c), S[c])]; P[c] = pinv(Q[c])
    end
    pos = [[searchsortedfirst(S[j], c) for j in S[c]] for c in 1:nc]
    @assert all(S[j][pos[c][i]] == c for c in 1:nc for (i, j) in enumerate(S[c]))          # the pattern is symmetric
    return (n = nc, S = S, Q = Q, b = b, P = P, pos = pos)
end
m0_project(D, rows) = [rows[c] .+ D.P[c] * (D.b[c] .- D.Q[c] * rows[c]) for c in 1:D.n]
m0_sym(D, rows) = [[(rows[c][i] + rows[j][D.pos[c][i]]) / 2 for (i, j) in enumerate(D.S[c])] for c in 1:D.n]
m0_matrix(D, rows, A) = Diagonal(A) + sparse(reduce(vcat, [fill(c, length(D.S[c])) for c in 1:D.n]), reduce(vcat, D.S), reduce(vcat, rows), D.n, D.n)
"M0 after t sweeps, for every t in ts."
function m0_swept(mesh, ts)
    D = m0_rows(mesh); rows = m0_project(D, [zeros(length(D.S[c])) for c in 1:D.n]); out = Dict{Int,Any}()
    for t in 0:maximum(ts)
        t in ts && (out[t] = (M = m0_matrix(D, rows, mesh.cells.area), res = maximum(maximum(abs, D.b[c] .- D.Q[c] * rows[c]) / mesh.cells.area[c] for c in 1:D.n)))
        t < maximum(ts) && (rows = m0_project(D, m0_sym(D, rows)))
    end
    return out
end
"H_P2 on the 23-entry pattern after the sweeps in ss."
function hp2_swept(mesh, ss)
    Hd = hdiag(mesh); D2 = p1_rows(mesh; k = 2); rows = project_rows(D2, [zeros(length(D2.nbv[e])) for e in 1:D2.n]); out = Dict{Int,Any}()
    for s in 0:maximum(ss)
        s in ss && (out[s] = Hd + tosparse(D2, rows))
        s < maximum(ss) && (rows = project_rows(D2, symmetrize(D2, rows)))
    end
    return out
end
const TS = (0, 1, 2, 4, 16)

say("Part 1/2: the swept 0-form star and the wave spectrum.  t = sweeps of M0;  growth = max σ/ω_max, growing = share of non-real eigenvalues")
for mn in ("mesh3", "mesh4", "mesh5")
    mesh = VoronoiMesh("meshes/$mn.nc"); N = mesh.cells.n; A = mesh.cells.area; Hd = hdiag(mesh)
    M = m0_swept(mesh, TS); HP = hp2_swept(mesh, (1, 4)); d0 = sparse(Float64.(d0_matrix_primal(mesh)))
    @assert norm(M[0].M - mass0_corrected(mesh), Inf) < 1e-10 * maximum(A)
    say(@sprintf("%s (%d cells)", mn, N))
    for t in TS
        M0 = M[t].M; Ds = Diagonal(1 ./ sqrt.(A)); Ms = Ds * ((M0 + M0') / 2) * Ds
        q = (asym = norm(M0 - M0') / norm(M0), lmin = N <= 4096 ? eigmin(Symmetric(Matrix(Ms))) : NaN, res = M[t].res,
             cmax = maximum(abs.(M0 - Diagonal(A))) / maximum(A))
        line = @sprintf("   t = %2d: ‖M0−M0ᵀ‖/‖M0‖ %.2e  λmin(sym)/A %.3f  max|C0|/A %.3f  row residual %.0e  |", t, q.asym, q.lmin, q.cmax, q.res)
        for (hn, H) in (("DEC", Hd), ("P2 s=1", HP[1]), ("P2 s=4", HP[4]))
            λ = eigvals(Matrix(M0) \ Matrix(d0' * H * d0)); λmax = maximum(abs.(λ))
            g = maximum(abs.(imag.(sqrt.(complex.(λ))))) / sqrt(λmax); fr = count(abs.(imag.(λ)) .> 1e-9 * λmax) / N
            RM0[(mn, t, hn)] = (growth = g, frac = fr, minre = minimum(real.(λ)) / λmax)
            line *= @sprintf("  %s: growth %.1e, growing %4.1f%%", hn, g, 100fr)
        end
        RM0[(mn, t, :star)] = q; say(line)
    end
    serialize(joinpath(@__DIR__, "..", "..", "output", "ts_data", "sw_m0_sweep.jls"), RM0)
end

say("\nPart 3: accuracy at t = 1 (RK4, Δt = 1e-3) with H = H_P2 after 1 sweep; e(h) = max|h − h_exact|/max|h|")
for mn in ("mesh3", "mesh4", "mesh5", "mesh6")
    mesh = VoronoiMesh("meshes/$mn.nc"); nc = mesh.cells.n; hf, uf = wave(mesh); pts = [_pt(p) for p in mesh.cells.position]
    x0 = Vector{Float64}(vcat(hf.(pts, 0.0), reduce_primal_edges(mesh, x -> uf(x, 0.0)).values)); hT = hf.(pts, 1.0)
    d0 = 1.0 * d0_matrix_primal(mesh); H = hp2_swept(mesh, (1,))[1]; M = m0_swept(mesh, (0, 1, 4, 16)); out = String[]
    for t in (0, 1, 4, 16)
        F = lu(M[t].M); fc(x) = vcat(DEPTH .* (F \ (d0' * (H * x[nc+1:end]))), -GRAV .* (d0 * x[1:nc]))
        xc = rk4(fc, x0, 1e-3, 1000); eh = maximum(abs.(xc[1:nc] .- hT)) / AMP
        RM0[(mn, t, :acc)] = eh; push!(out, @sprintf("t = %d: %.2e", t, eh))
    end
    say(@sprintf("   %s (h = 1/%d)   %s", mn, round(Int, sqrt(nc)), join(out, "   ")))
    serialize(joinpath(@__DIR__, "..", "..", "output", "ts_data", "sw_m0_sweep.jls"), RM0)
end
write(joinpath(@__DIR__, "..", "..", "output", "ts_data", "sw_m0_sweep.txt"), join(LG, "\n") * "\n")
