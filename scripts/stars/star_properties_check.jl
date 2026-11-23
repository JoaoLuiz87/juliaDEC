# ============================================================================
#  star_properties_check.jl -- which properties of the Hodge star survive from H_DEC to H_P1 and H_P2, and what the
#  lost ones do to the solutions.  Companion of hodge_properties_note.tex.
#
#      julia -t 4 scripts/stars/star_properties_check.jl        ->  output/ts_data/star_properties.{txt,jls}
#
#  A. construction: build time of H_P1 (one global sparse solve) and H_P2 (one small pinv per row), nnz per row,
#     relative antisymmetric part ‖H - Hᵀ‖_F / ‖H‖_F; sign convention D̃ = ±D0ᵀ; total build time on mesh 7 (cache)
#  B. the Laplacian L = -D̃ H D0 (cell integrals of -Δp): positive off-diagonal entries (M-matrix test), their weight
#     Σ_{j≠i} max(L_ij, 0) / L_ii, and the column sums (exact conservation)
#  C. heat equation  A dU/dt = -L U, backward Euler, 10 steps of dt = 0.1h², h², 10h²:
#     from a point (10 cells) and from the indicator of a disc (r = 0.15): worst undershoot min U / max U0, worst
#     overshoot (max U - max U0)/max U0, and the drift of the total heat Σ A U
#  D. Helmholtz split v = D0 φ + w with D̃ H w = 0, for a smooth field u = ∇φ0 + J∇ψ0: the energy cross term
#     ½vᵀHv − ½(D0φ)ᵀH(D0φ) − ½wᵀHw (zero for symmetric H)
# ============================================================================
isdefined(Main, :cached_stars) || include(joinpath(@__DIR__, "stars_cache.jl"))
isdefined(Main, :symmetric_correction) || include(joinpath(@__DIR__, "commuting_check.jl"))
using Printf, Serialization, Random, SparseArrays, LinearAlgebra
const OUT = String[]
logp(s) = (println(s); flush(stdout); push!(OUT, s))
R = Dict{Any,Any}()
const SMESHES = haskey(ENV, "SP_MESHES") ? Tuple(String.(split(ENV["SP_MESHES"], ","))) : ("mesh3", "mesh4", "mesh5", "mesh6")
const STARS = ("H_DEC", "H_P1", "H_P2")

for mn in SMESHES
    mesh = VoronoiMesh("meshes/$mn.nc"); S = cached_stars(mn; mesh)         # relative path: NetCDF fails on non-ASCII absolute paths
    A = mesh.cells.area; N = mesh.cells.n; h = sqrt(1 / N)
    D0 = sparse(Float64.(d0_matrix_primal(mesh))); Dt = sparse(Float64.(d1_matrix_dual(mesh)))
    sgn = norm(Dt + D0', Inf) < 1e-12 ? -1 : norm(Dt - D0', Inf) < 1e-12 ? 1 : 0
    logp(@sprintf("\n%s: %d cells, %d edges, h = 1/%d.   D̃ = %s D0ᵀ", mn, N, mesh.edges.n, round(Int, 1 / h), sgn == -1 ? "−" : sgn == 1 ? "+" : "?"))
    # ---------------- A. construction
    Hd = hdiag(mesh)
    t1 = @elapsed symmetric_correction(mesh, 1, 3); t2 = @elapsed rowwise_correction(mesh, Hd, 2, 3)
    t1 = @elapsed symmetric_correction(mesh, 1, 3); t2 = @elapsed rowwise_correction(mesh, Hd, 2, 3)     # second call: no compilation
    R[(mn, :build)] = (P1 = t1, P2 = t2)
    for s in STARS
        H = S[s]; R[(mn, s, :struct)] = (nnzrow = nnz(H) / size(H, 1), asym = norm(H - H', 2) / norm(H, 2))
    end
    logp(@sprintf("   A. build time: H_P1 %.2f s (global solve), H_P2 %.2f s (row by row);  nnz/row %.1f / %.1f / %.1f;  ‖H−Hᵀ‖/‖H‖ (Frobenius) %.1e / %.1e / %.1e",
                  t1, t2, (R[(mn, s, :struct)].nnzrow for s in STARS)..., (R[(mn, s, :struct)].asym for s in STARS)...))
    # ---------------- B. the Laplacian
    for s in STARS
        L = -(Dt * S[s] * D0); minimum(diag(L)) < 0 && (L = -L)
        Lt = sparse(L'); I_, J_, V_ = findnz(L); off = I_ .!= J_; scale = maximum(abs.(V_))
        pos = off .& (V_ .> 1e-13 * scale)
        rowpos = zeros(N); for k in findall(pos); rowpos[I_[k]] += V_[k]; end
        R[(mn, s, :lap)] = (fracpos = count(pos) / count(off), weight = maximum(rowpos ./ diag(L)), colsum = maximum(abs.(vec(sum(L, dims = 1)))) / scale,
                            rowsum = maximum(abs.(vec(sum(L, dims = 2)))) / scale)
        logp(@sprintf("   B. %-5s L: positive off-diagonals %5.1f%%, max Σpos/diag %.3f, max|column sum| %.0e, max|row sum| %.0e (relative)",
                      s, 100R[(mn, s, :lap)].fracpos, R[(mn, s, :lap)].weight, R[(mn, s, :lap)].colsum, R[(mn, s, :lap)].rowsum))
    end
    # ---------------- C. heat equation
    Random.seed!(1); cells = rand(1:N, 10)
    pos = [_pt(x) for x in mesh.cells.position]
    disc = [((p[1] - 0.5)^2 + (p[2] - 0.5)^2 < 0.15^2 ? 1.0 : 0.0) for p in pos]
    for s in STARS
        L = -(Dt * S[s] * D0); minimum(diag(L)) < 0 && (L = -L)
        out = String[]
        for dtf in (0.1, 1.0, 10.0)
            dt = dtf * h^2; F = lu(Diagonal(A) + dt * L)
            under = 0.0; over = 0.0; drift = 0.0
            for u0 in vcat([(z = zeros(N); z[c] = 1 / A[c]; z) for c in cells], [disc])
                u = copy(u0); m0 = maximum(u0); tot0 = sum(A .* u0)
                for _ in 1:10
                    u = F \ (A .* u)
                    under = min(under, minimum(u) / m0); over = max(over, (maximum(u) - m0) / m0); drift = max(drift, abs(sum(A .* u) - tot0) / tot0)
                end
            end
            R[(mn, s, dtf, :heat)] = (under = under, over = over, drift = drift)
            push!(out, @sprintf("dt=%gh²: under %+.1e over %+.1e", dtf, under, over))
        end
        logp(@sprintf("   C. %-5s heat (10 backward-Euler steps, point and disc data): %s;  total-heat drift ≤ %.0e", s, join(out, "  "),
                      maximum(R[(mn, s, d, :heat)].drift for d in (0.1, 1.0, 10.0))))
    end
    # ---------------- D. Helmholtz split and the energy cross term
    φ0(x) = sin(TP * x[1]) * cos(TP * x[2]); ψ0(x) = cos(TP * x[1]) + 0.5 * sin(2TP * x[2])
    ufield(x) = [TP * cos(TP * x[1]) * cos(TP * x[2]), -TP * sin(TP * x[1]) * sin(TP * x[2])] .+
                [-(0.5 * 2TP * cos(2TP * x[2])), -TP * sin(TP * x[1])]          # ∇φ0 + J∇ψ0, J∇ψ = (-ψ_y, ψ_x)
    v = reduce_primal_edges(mesh, ufield).values
    for s in STARS
        H = S[s]; K = Dt * H * D0; b = Dt * H * v
        φ = zeros(N); φ[2:end] = K[2:end, 2:end] \ b[2:end]                        # pin φ_1 = 0 (constants are the kernel)
        g = D0 * φ; w = v - g
        E = dot(v, H * v) / 2; Eg = dot(g, H * g) / 2; Ew = dot(w, H * w) / 2; cross = E - Eg - Ew
        R[(mn, s, :helm)] = (cross = cross / E, divw = norm(Dt * H * w) / norm(b), alt = dot(g, H' * w) / E)
        logp(@sprintf("   D. %-5s Helmholtz split: |D̃Hw|/|D̃Hv| %.0e;  energy cross term (E − E_grad − E_rot)/E = %+.2e  (= gᵀHᵀw/E = %+.2e)",
                      s, R[(mn, s, :helm)].divw, R[(mn, s, :helm)].cross, R[(mn, s, :helm)].alt))
    end
    serialize(joinpath(@__DIR__, "..", "..", "output", "ts_data", "star_properties.jls"), R)
end
c7 = joinpath(STARDIR, "mesh7.jls")
isfile(c7) && logp(@sprintf("\nmesh7: both stars built in %.0f s (stars_cache)", deserialize(c7)["build_time"]))
write(joinpath(@__DIR__, "..", "..", "output", "ts_data", "star_properties.txt"), join(OUT, "\n") * "\n")
