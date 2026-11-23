# ============================================================================
#  three_stars_tests.jl -- checks that the code behind three_stars_report.tex has no gaps.
#
#      julia -t 4 scripts/stars/three_stars_tests.jl      (writes output/ts_data/tests.txt)
#
#  Every check prints PASS/FAIL against a stated tolerance.  The exactness check (T3) does NOT reuse the construction
#  code: it has its own monomials, its own 5-point Gauss rule and its own tangents, so a wrong basis, scaling, frame or
#  orientation in symmetric_correction / rowwise_correction would show up there.
# ============================================================================
include(joinpath(@__DIR__, "three_stars.jl"))
empty!(lines)

npass = Ref(0); nfail = Ref(0)
function check(name, val, tol)
    ok = val <= tol
    ok ? (npass[] += 1) : (nfail[] += 1)
    logp(@sprintf("   %-4s %-62s %.2e  (tol %.0e)", ok ? "PASS" : "FAIL", name, val, tol))
end

# ------------------------------------------------------------- independent row-exactness check (T3)
const GL5X = [-0.9061798459386640, -0.5384693101056831, 0.0, 0.5384693101056831, 0.9061798459386640]
const GL5W = [0.2369268850561891, 0.4786286704993665, 0.5688888888888889, 0.4786286704993665, 0.2369268850561891]
"∫ F·τ ds along the segment a→b for a fixed unit vector τ (5-point Gauss, exact for polynomial F of degree ≤ 9)."
segint(F, a, b, τ) = norm(b .- a) * sum(w / 2 * dot(F(a .+ (ξ + 1) / 2 .* (b .- a)), τ) for (ξ, w) in zip(GL5X, GL5W))
"Vector monomials of exact degree k about x0, scaled by s: (X^i Y^j, 0) and (0, X^i Y^j), i + j = k."
vmonos(k, x0, s) = vcat([x -> [((x[1] - x0[1]) / s)^i * ((x[2] - x0[2]) / s)^(k - i), 0.0] for i in k:-1:0],
                        [x -> [0.0, ((x[1] - x0[1]) / s)^i * ((x[2] - x0[2]) / s)^(k - i)] for i in k:-1:0])

"max over the sampled rows e and the monomials of exact degree k of |(H R_h f)_e - (R~ *f)_e| / |e~|."
function row_exactness(mesh, H, k, rows)
    Ht = sparse(H'); worst = 0.0
    for e in rows
        xe = mesh.edges.position[e]; x0 = _pt(xe)
        a, b = edge_endpoints(mesh, e, true; about = xe); s = norm(b .- a); te = (b .- a) ./ s
        v1, v2 = edge_endpoints(mesh, e, false; about = xe); le = norm(v2 .- v1)
        for F in vmonos(k, x0, s)
            lhs = 0.0
            for p in nzrange(Ht, e)
                j = rowvals(Ht)[p]
                q1, q2 = edge_endpoints(mesh, j, true; about = xe)
                lhs += nonzeros(Ht)[p] * segint(F, q1, q2, (q2 .- q1) ./ norm(q2 .- q1))
            end
            worst = max(worst, abs(lhs - segint(F, v1, v2, te)) / le)
        end
    end
    return worst
end

for mn in MESHES
    mesh = VoronoiMesh("meshes/$mn.nc")
    logp(@sprintf("%s  (h = %.4f, %d cells, %d edges, %d triangles)", mn, meshsize(mesh), mesh.cells.n, mesh.edges.n, mesh.vertices.n))
    tris, D1, nflip = triangles_ccw(mesh)
    d0 = d0_matrix_primal(mesh); A = mesh.cells.area
    lp = edge_lengths(mesh, true)

    logp("  T1  geometry and orientation")
    check("dual edges oriented by J t_e (count of wrong edges)", check_dual_orientation(mesh), 0)
    tot, Ω, _ = check_diamond_partition(mesh)
    check("diamonds tile the torus: |Σ|D_e| - |Ω||", abs(tot - Ω), 1e-12)
    check("Voronoi cells tile the torus: |Σ A_c - |Ω||", abs(sum(A) - Ω), 1e-12)
    check("triangles tile the torus: |Σ |T| - |Ω||", abs(sum(T.area for T in tris) - Ω), 1e-12)
    logp(@sprintf("        (%d of %d triangles are listed clockwise by mesh.vertices.cells; D1 corrects them)", nflip, length(tris)))
    # Stokes identities.  A sign or orientation error makes these O(1); what remains is the error of the 4-point edge
    # rule and the 7-point triangle rule (largest on mesh 1, about 1e-5), so the tolerance is 1e-3.
    pe = [pfun(_pt(x)) for x in mesh.cells.position]
    αg = reduce_primal_edges(mesh, gradp).values
    check("d0 p = R_h grad p  (rel. max; quadrature error only)", maximum(abs.(d0 * pe .- αg)) / maximum(abs.(αg)), 1e-3)
    stok = d0' * reduce_dual_edges(mesh, star_field(gradp)).values
    lapc = discretize_scalar_field_cells(mesh, lapp).values
    check("Stokes on Voronoi cells: d0' R~*grad p = -∫Δp  (rel. max)", maximum(abs.(stok .+ lapc)) / maximum(abs.(lapc)), 1e-3)
    check("D1 d0 = 0 (exact integers)", maximum(abs.(D1 * d0)), 0)
    fv = reduce_primal_edges(mesh, x -> _ROT90 * VD(x)).values
    Φ = [triangle_integral_ho(PHID, T.pts...; qdeg = 5) for T in tris]
    check("D1 R_h(Jv) = ∫_T div v  (rel. max)", maximum(abs.(D1 * fv .- Φ)) / maximum(abs.(Φ)), 1e-3)
    pT = [PD(T.cc) for T in tris]
    circ = reduce_dual_edges(mesh, x -> -VD(x)).values               # circulation of ∇p along e~ (direction J t_e)
    check("D1' p(c_T) = circulation of grad p along e~  (rel. max)", maximum(abs.(D1' * pT .- circ)) / maximum(abs.(circ)), 1e-3)

    logp("  T2  construction")
    t0 = time(); stars, info = three_stars(mesh); tb = time() - t0
    H1, H2, Hd = stars["H_P1"], stars["H_P2"], stars["H_DEC"]
    check("H_P1: relative residual of the [P1]^2 constraints", info.res_p1, 1e-9)
    check("H_P1 symmetric: ||H - H'||_F / ||H||_F", norm(H1 - H1') / norm(H1), 0)
    check("H_DEC, H_P1 diagonal = |e~|/|e| + C_ee > 0 (min, negated)", -minimum(diag(H1)), 0)
    Smin = Inf; Cmax = 0.0
    for e in 1:mesh.edges.n                                            # unisolvence of every [P2]^2 row problem
        x0 = _pt(mesh.edges.position[e]); s = lp[e]
        Ae = reduce(hcat, [(q = edge_endpoints(mesh, j, true; about = mesh.edges.position[e]); seg_moments(q[1], q[2], x0, s, 2))
                           for j in ring(mesh, e, 3)])'
        sv = svdvals(Ae); Smin = min(Smin, sv[end]); Cmax = max(Cmax, sv[1] / sv[end])
    end
    check("H_P2: every A_e has full column rank (-min σ_min, scaled basis)", -Smin, -1e-3)
    logp(@sprintf("        (min σ_min(A_e) = %.3e, max cond(A_e) = %.1f; build time of the three stars %.1f s)", Smin, Cmax, tb))
    for (name, H) in stars
        check("$name: Gershgorin certificate of sym(H) is positive (-g)", -gersh_sym(mesh, H), 0)
    end

    logp("  T3  exactness, independent quadrature and basis (all rows on meshes 1-4, every 10th on 5-6)")
    rows = mesh.edges.n <= 3072 ? (1:mesh.edges.n) : (1:10:mesh.edges.n)
    for (name, k) in (("H_DEC", 0), ("H_P1", 1), ("H_P2", 2))
        for kk in 0:k
            check("$name exact on degree $kk  (max |row residual| / |e~|)", row_exactness(mesh, stars[name], kk, rows), 1e-9)
        end
        logp(@sprintf("        %s on degree %d (should NOT be exact): %.2e", name, k + 1, row_exactness(mesh, stars[name], k + 1, rows)))
    end

    logp("  T4  Darcy (Hirani et al. eq. 4.6)")
    ex = darcy_patch(mesh, tris)
    for name in STARS
        fh, ph = solve_darcy(stars[name], D1, ex.G, ex.Φ, [T.area for T in tris])
        check("$name patch test v = (1,0), g = (1,0): flux error (max)", maximum(abs.(fh .- ex.f)), 1e-10)
        check("$name patch test: pressure is constant (max - min)", maximum(ph) - minimum(ph), 1e-10)
    end
    # as in T1, sign errors would make these O(1); the values are quadrature errors (largest on mesh 1)
    ex = darcy_smooth(mesh, tris)
    check("exact smooth solution satisfies D1 f = phi  (rel. max)", maximum(abs.(D1 * ex.f .- ex.Φ)) / maximum(abs.(ex.Φ)), 1e-3)
    for name in STARS
        H = stars[name]
        εw = H * ex.f .- reduce_dual_edges(mesh, star_field(x -> _ROT90 * VD(x))).values
        r1 = -H * ex.f .+ D1' * ex.p .- ex.G
        check("$name: Darcy-law residual of the exact solution = -ε(Jv)  (rel.)", maximum(abs.(r1 .+ εw)) / maximum(abs.(D1' * ex.p)), 1e-4)
    end

    logp("  T5  Poisson solver")
    for name in STARS
        H = stars[name]; L = sparse(d0' * H * d0)
        b = discretize_scalar_field_cells(mesh, x -> -lapp(x)).values; b .-= sum(b) / length(b)
        ph = solve_poisson_primal(mesh, H, x -> -lapp(x))
        check("$name: ||L p_h - b|| / ||b||", norm(L * ph .- b) / norm(b), 1e-10)
    end
end
logp(@sprintf("\n%d checks passed, %d failed", npass[], nfail[]))
write(joinpath(TSDIR, "tests.txt"), join(lines, "\n") * "\n")
