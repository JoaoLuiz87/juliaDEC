# ============================================================================
#  green_charge_mesh.jl -- the balance behind the non-locality theorem on SCVT mesh 4 (simple form, for the Laplacian L = D0ᵀ H D0):
#      Σ_{v∈K, w∉K} L_vw (φ_v γ_w − γ_v φ_w) = 2 Σ_{v∈K} |ṽ| γ(x_v) + (antisymmetric part inside K),   φ = |x|²/2, γ = (x1²−x2²)/2 about x0,
#  exact for every star exact on [P1]²; the last sum is zero for symmetric stars.  One generator x0 is moved by 0.15 h; the charge 2ΔQ and the
#  change of the crossing sum are compared on circles of growing radius for H_P1 (symmetric) and H_P2 (not symmetric).
#      julia -t 4 scripts/barrier/green_charge_mesh.jl      -> ts_data/green_charge_mesh.{txt,jls}
# ============================================================================
ENV["LOC_MAIN"] = "0"; include(joinpath(@__DIR__, "locality_check.jl")); include(joinpath(@__DIR__, "..", "stars", "stars_cache.jl"))
using Serialization
OUTM = String[]; sm(s) = (println(s); flush(stdout); push!(OUTM, s))
m0 = VoronoiMesh("meshes/mesh4.nc"); X = [_pt(p) for p in m0.cells.position]; h = 1 / sqrt(length(X))
v0 = argmin([norm(x .- [0.5, 0.5]) for x in X]); x0 = X[v0]
wrap(d) = d .- round.(d)
function setup(X)
    mesh = delaunay_mesh(first.(X), last.(X)); P = [_pt(p) for p in mesh.cells.position]
    D0 = sparse(Float64.(d0_matrix_primal(mesh)))
    H1 = symmetric_correction(mesh, 1, 3)[1]; H2 = rowwise_correction(mesh, hdiag(mesh), 2, 3)[1]
    return (mesh = mesh, P = P, rel = [wrap(p .- x0) for p in P], A = Float64.(mesh.cells.area), L1 = D0' * H1 * D0, L2 = D0' * H2 * D0, H1 = H1, H2 = H2, n = length(P))
end
S0 = setup(X); X1 = deepcopy(X); X1[v0] = x0 .+ 0.15h .* [1.0, 0.0]; S1 = setup(X1)
sm(@sprintf("mesh 4: %d cells, same ordering of generators: %s, same number of edges: %s", S0.n, maximum(norm(wrap(S0.P[i] .- X[i])) for i in eachindex(X)) < 1e-9, S0.mesh.edges.n == S1.mesh.edges.n))
φ(x) = dot(x, x) / 2; γ(x) = (x[1]^2 - x[2]^2) / 2
function balance(S, L, r)
    K = [norm(x) < r for x in S.rel]; ph = φ.(S.rel); ga = γ.(S.rel); rv = rowvals(L); nz = nonzeros(L); cross = 0.0; inside = 0.0
    for w in 1:S.n, p in nzrange(L, w)
        v = rv[p]; K[v] || continue
        t = nz[p] * (ph[v] * ga[w] - ga[v] * ph[w]); K[w] ? (inside += t) : (cross += t)
    end
    return cross, inside, 2 * sum(S.A[v] * ga[v] for v in 1:S.n if K[v]), count(K)
end
D = Dict{Any,Any}()
for (nm, f) in (("H_P1", :L1), ("H_P2", :L2))
    sm("$nm:  circle radius r/h, cells inside;  before the move: crossing + inside sums against the charge 2Σ|ṽ|γ;  change when the generator moves by 0.15 h")
    for rh in (3.0, 4.0, 6.0, 8.0, 10.0, 12.0)
        c0, i0, q0, n = balance(S0, getfield(S0, f), rh * h); c1, i1, q1, _ = balance(S1, getfield(S1, f), rh * h)
        sm(@sprintf("   r = %4.1f h (%3d cells)  crossing %+.6e  inside %+.6e  charge %+.6e  error %.1e  |  Δcrossing %+.6e  Δinside %+.6e  Δcharge %+.6e", rh, n, c0, i0, q0, abs(c0 + i0 - q0) / (abs(q0) + h^4), c1 - c0, i1 - i0, q1 - q0))
        D[(nm, rh)] = (c0, i0, q0, c1, i1, q1, n)
    end
end
# how many rows of H change, and by how much with the distance: the two meshes number (and orient) their edges differently -> align by the end cells
key(m, e) = (c = m.edges.cells[e]; (minmax(Int(c[1]), Int(c[2]))))
idx0 = Dict(key(S0.mesh, e) => e for e in 1:S0.mesh.edges.n); π0 = [get(idx0, key(S1.mesh, e), 0) for e in 1:S1.mesh.edges.n]
sg = [π0[e] == 0 ? 0.0 : (Int(S1.mesh.edges.cells[e][1]) == Int(S0.mesh.edges.cells[π0[e]][1]) ? 1.0 : -1.0) for e in 1:S1.mesh.edges.n]
sm(@sprintf("edges aligned: %d of %d (same connectivity: %s)", count(>(0), π0), length(π0), all(>(0), π0)))
for (nm, f) in (("H_P1", :H1), ("H_P2", :H2))
    H0 = getfield(S0, f)[π0, π0]; H0 = Diagonal(sg) * H0 * Diagonal(sg); Δ = getfield(S1, f) .- H0; Δt = sparse(Δ')
    geo = [wrap(_pt(S1.mesh.edges.position[e]) .- x0) for e in 1:S1.mesh.edges.n]
    rowch = [isempty(nzrange(Δt, e)) ? 0.0 : maximum(abs, nonzeros(Δt)[nzrange(Δt, e)]) for e in 1:S1.mesh.edges.n]
    sm(@sprintf("%s: rows whose largest change exceeds 1e-9: %d of %d, 1e-6: %d, 1e-5: %d", nm, count(>(1e-9), rowch), length(rowch), count(>(1e-6), rowch), count(>(1e-5), rowch)))
    D[(nm, :rows)] = (geo, rowch)
end
serialize(joinpath(TSDIR, "green_charge_mesh.jls"), (D, h, x0)); write(joinpath(TSDIR, "green_charge_mesh.txt"), join(OUTM, "\n") * "\n")
