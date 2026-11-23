# ============================================================================
#  lift_vs_p2_check.jl -- the lift star of hodge_stars.jl (row e = b_eᵀ Π_e, Π_e = pinv(A_e)) against H_P2
#
#      julia -t 4 scripts/stars/lift_vs_p2_check.jl        ->  output/ts_data/lift_vs_p2.txt
#
#  Both are exact, row by row, on the same space with the same stencil (P2: [P2]² on get_neighbors_3level_primal).
#  Row e solves the same underdetermined system A_eᵀ h = b_e (12 conditions, ~19-23 unknowns); they pick different
#  solutions:
#      lift:  h = A_e (A_eᵀA_e)⁻¹ b_e = Π_eᵀ b_e               the minimum-norm solution (closest to 0)
#      H_P2:  h = argmin ‖h − h_DEC‖  s.t. A_eᵀ h = b_e          the solution closest to the DEC row
#   ⇒  H_P2 row = lift row + (I − Q_e) h_DEC,   Q_e = A_e Π_e  (orthogonal projector onto range A_e).
#  Part 1 verifies this identity.  Part 2 compares: antisymmetry, positivity of the symmetric part (Gershgorin g and
#  λmin μ), Laplacian spectrum, Hodge error, Poisson error, for H_DEC, lift P1 (13 edges), lift P2, H_P1, H_P2.
# ============================================================================
isdefined(Main, :cached_stars) || include(joinpath(@__DIR__, "stars_cache.jl"))
using Printf, LinearAlgebra, SparseArrays
const OUT = String[]
logp(s) = (println(s); flush(stdout); push!(OUT, s))

for mn in ("mesh3", "mesh4", "mesh5")
    mesh = VoronoiMesh("meshes/$mn.nc"); S = cached_stars(mn; mesh); n = mesh.edges.n
    Hd = hdiag(mesh); hd = diag(Hd)
    L1 = hodge_star_lift_matrix(mesh; only_linear = true)          # [P1]^2, 13-edge stencil
    L2 = hodge_star_lift_matrix(mesh; only_linear = false)         # [P2]^2, 3-level stencil
    logp(@sprintf("\n%s: %d cells, %d edges", mn, mesh.cells.n, n))
    # ---------------- Part 1: H_P2 = lift + (I − Q_e) h_DEC, row by row
    err = 0.0; sc = maximum(abs.(nonzeros(S["H_P2"])))
    Hp2t = sparse(S["H_P2"]'); L2t = sparse(L2')
    for e in 1:n
        L = lsq_lift(mesh, e; only_linear = false)
        hdec = [j == e ? hd[e] : 0.0 for j in L.neighbors]
        pred = (I - L.Q) * hdec
        for (k, j) in enumerate(L.neighbors)
            err = max(err, abs(S["H_P2"][e, j] - L2[e, j] - pred[k]))
        end
    end
    stencil_same = all(Set(rowvals(Hp2t)[nzrange(Hp2t, e)]) ⊆ Set(lsq_lift(mesh, e; only_linear = false).neighbors) for e in 1:min(n, 200))
    logp(@sprintf("   Part 1: max |H_P2 − lift_P2 − (I−Q)h_DEC| / max|H_P2| = %.1e   (H_P2 stencil within the lift stencil: %s)", err / sc, stencil_same))
    # ---------------- Part 2: properties
    stars = (("H_DEC", Hd), ("lift P1", L1), ("lift P2", L2), ("H_P1", S["H_P1"]), ("H_P2", S["H_P2"]))
    for (name, H) in stars
        asym = norm(H - H') / norm(H)
        g = gersh_sym(mesh, H)
        Λ = bigLambda(mesh, H)
        offd = maximum(abs(H[e, j]) / hd[e] for e in 1:n for j in rowvals(sparse(H'))[nzrange(sparse(H'), e)] if j != e; init = 0.0)
        r = hodge_error(mesh, H, fields.d, supn.d)
        p = try poisson_errors(mesh, H).inf catch; NaN end
        s = @sprintf("   %-8s ‖H−Hᵀ‖/‖H‖ %.2e  Gershgorin g %+.3f  Λ %5.2f  max|offdiag|/diag %.2f  Hodge r∞ %.2e r2 %.2e  Poisson max %.2e",
                     name, asym, g, Λ, offd, r.inf, r.l2, p)
        if mn in ("mesh3", "mesh4")
            μ, im, γ = wave_stats(mesh, H)
            d0 = d0_matrix_primal(mesh)
            λ = eigvals(Matrix(Diagonal(1 ./ mesh.cells.area) * (d0' * H * d0)))
            s *= @sprintf("  μ %+.3f  Laplacian: min Re λ/max|λ| %+.1e, max|Im λ|/max|λ| %.1e, wave growth γ %.1e", μ, minimum(real.(λ)) / maximum(abs.(λ)), im, γ)
        end
        logp(s)
    end
end
write(joinpath(@__DIR__, "..", "..", "output", "ts_data", "lift_vs_p2.txt"), join(OUT, "\n") * "\n")
