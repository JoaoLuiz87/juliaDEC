# ============================================================================
#  m0_barrier_check.jl -- the analogue of the skew for the 0-form star M0 (point values -> cell integrals), and the spectrum of the
#  linear shallow-water schemes of sw_linear_three.jl with the three stars.
#
#      julia -t 4 scripts/shallow_water/m0_barrier_check.jl        -> output/ts_data/m0_barrier.txt
#
#  (1) s0(p,q) = Σ_v [ p(x_v) ∫_ṽ q − q(x_v) ∫_ṽ p ].  If M0 = M0ᵀ and M0 is exact on p and q (M0 R⁰p = cell integrals of p),
#      then p̂ᵀ M0 q̂ − q̂ᵀ M0 p̂ = 0 = s0(p,q) with p̂ = point values: so a symmetric M0 exact on p, q requires s0(p,q) = 0, and
#      for a symmetric M0 the error on p or q is at least of the order of s0 (the argument of Theorem 3.5 for 0-forms).
#      Computed with a degree-5 rule on the Voronoi fan triangles (cell integrals exact to ~1e-10).
#  (2) max Re λ of the linear operators on meshes 3 and 4 (dense eigenvalues): TRSK and consistent.
# ============================================================================
include(joinpath(@__DIR__, "sw_linear_three.jl"))
OUTM = String[]; sm(s) = (println(s); flush(stdout); push!(OUTM, s))
tr5 = tri_rule(5)
"cell integrals of a function (unit torus) by the degree-5 rule on the fan triangles, applied twice (4 sub-triangles) for accuracy"
function cell_integrals(mesh, f)
    bq, wq = tr5
    I = zeros(mesh.cells.n)
    for c in 1:mesh.cells.n, (a, b, d) in voronoi_cell_triangles(mesh, c)
        m1, m2, m3 = (a .+ b) ./ 2, (b .+ d) ./ 2, (a .+ d) ./ 2
        for (x, y, z) in ((a, m1, m3), (m1, b, m2), (m3, m2, d), (m1, m2, m3))
            Ar = _tri_area(x, y, z)
            for (bb, ww) in zip(bq, wq); I[c] += ww * Ar * f(bb[1] .* x .+ bb[2] .* y .+ bb[3] .* z); end
        end
    end
    return I
end
pf(x) = sin(2π * x[1]) * cos(2π * x[2])
qf(x) = cos(4π * x[1]) + 0.5 * sin(2π * (x[1] + x[2]))
rf(x) = sin(2π * (2x[1] - x[2]))
sm("M0 skew: s0(p,q) = Σ p(x_v)∫q − q(x_v)∫p   (pairs (p,q), (p,r), (q,r));  in brackets s0/h², s0/h⁴")
for mn in ("mesh2", "mesh3", "mesh4", "mesh5", "mesh6", "mesh7", "mesh3_regular", "mesh5_regular", "mesh6_regular")
    mesh = VoronoiMesh("meshes/$mn.nc"); N = mesh.cells.n; h = 1 / sqrt(N)
    X = [_pt(p) for p in mesh.cells.position]
    Ip, Iq, Ir = cell_integrals(mesh, pf), cell_integrals(mesh, qf), cell_integrals(mesh, rf)
    vp, vq, vr = [pf(x) for x in X], [qf(x) for x in X], [rf(x) for x in X]
    s(a, Ia, b, Ib) = dot(a, Ib) - dot(b, Ia)
    ss = (s(vp, Ip, vq, Iq), s(vp, Ip, vr, Ir), s(vq, Iq, vr, Ir))
    sm(@sprintf("%-14s N %6d  ", mn, N) * join([@sprintf("%+.3e [%+.2e, %+.2e]", s0, s0 / h^2, s0 / h^4) for s0 in ss], "  "))
end
sm("\nspectrum of the linear schemes (dense eigenvalues, mesh 3 and mesh 4): max Re λ, e-folding time, max |Im λ|")
for mn in ("mesh3", "mesh4")
    mesh = VoronoiMesh("meshes/$mn.nc"); S = cached_stars(mn; mesh)
    O0 = sw_ops(mesh, S["H_DEC"]); P = perp_p2(mesh, O0.W * O0.H); M0c = mass0_corrected(mesh) .* O0.L^2
    for st in STARS3
        O = sw_ops(mesh, S[st])
        _, Lt = scheme_ops(O, :trsk, nothing, nothing)
        Lc = [spzeros(O.nc, O.nc) H0J .* sparse(Matrix(M0c) \ Matrix(O.D0t * O.H)); -GJ .* O.D0 -FJ .* P]
        for (lab, L) in (("trsk", Lt), ("consistent", Lc))
            λ = eigvals(Matrix(L)); r = maximum(real, λ)
            sm(@sprintf("%s %-5s %-10s max Re λ = %+.3e s⁻¹  (e-folding %s)   max|Im λ| = %.2e s⁻¹   max Re λ/max|Im λ| = %.1e", mn, st, lab, r,
                        r > 1e-12 ? @sprintf("%.1f days", 1 / r / 86400) : "none", maximum(abs ∘ imag, λ), r / maximum(abs ∘ imag, λ)))
        end
    end
end
write(joinpath(TSDIR, "m0_barrier.txt"), join(OUTM, "\n") * "\n")
