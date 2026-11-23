# ============================================================================
#  p2_symmetric_check.jl -- can H_P2 be built like H_P1, i.e. as the smallest SYMMETRIC correction of DEC exact on
#  [P2]^2, if the stencil is made wide enough?  (Section 2.5 of three_stars_report.tex)
#
#      julia -t 4 scripts/stars/p2_symmetric_check.jl      (writes output/ts_data/p2_symmetric.txt and .jls)
#
#  Stencils of an edge e, built with the juliaDEC.jl neighbour functions:
#      :t2  e, its two triangles, and the triangles across their 4 outer edges        13 edges  (get_neighbors_2level_primal)
#      3    every edge at the 4 vertices of the two triangles (used by H_P1, H_P2)    17-21     (get_neighbors_3level_primal)
#      :t3  :t2 plus one more layer of triangles                                      ~29
#      4    the union of the 13-edge stencils of the 13 edges of :t2                  ~45       (ring(mesh, e, 4))
#  On each stencil and mesh two stars exact on [P2]^2 row by row:
#      sym  the H_P1 recipe with k = 2: smallest symmetric correction, one global solve (symmetric_correction)
#      row  the H_P2 recipe: smallest correction of each row on its own (rowwise_correction)
#  and their size (Λ, largest entry), positivity (Gershgorin g, λmin of the symmetric part), exactness (independent
#  check), Hodge error, Poisson error.  Mesh 1 is left out: the wide stencils wrap around its torus.
# ============================================================================
include(joinpath(@__DIR__, "three_stars.jl"))         # tri_layers, ring(mesh, e, :t2 | :t3), STENCILS, SNAME
empty!(lines)

# independent row-exactness check (same as T3 of three_stars_tests.jl)
const GL5X = [-0.9061798459386640, -0.5384693101056831, 0.0, 0.5384693101056831, 0.9061798459386640]
const GL5W = [0.2369268850561891, 0.4786286704993665, 0.5688888888888889, 0.4786286704993665, 0.2369268850561891]
segint(F, a, b, τ) = norm(b .- a) * sum(w / 2 * dot(F(a .+ (ξ + 1) / 2 .* (b .- a)), τ) for (ξ, w) in zip(GL5X, GL5W))
vmonos(k, x0, s) = vcat([x -> [((x[1] - x0[1]) / s)^i * ((x[2] - x0[2]) / s)^(k - i), 0.0] for i in k:-1:0],
                        [x -> [0.0, ((x[1] - x0[1]) / s)^i * ((x[2] - x0[2]) / s)^(k - i)] for i in k:-1:0])
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

"λmin of D^-1/2 sym(H) D^-1/2 (D = H_DEC) to 1e-3, by bisection with sparse Cholesky: λmin > t <=> S - tI is SPD."
function lambda_min(mesh, H)
    Dm = Diagonal(1 ./ sqrt.(diag(hdiag(mesh)))); S = sparse(Symmetric(Dm * ((H + H') / 2) * Dm))
    spd(t) = issuccess(cholesky(S - t * I; check = false))
    lo = -1.0; while spd(lo) == false && lo > -1e7; lo *= 4; end
    hi = 1.5
    while hi - lo > 1e-3 * max(1.0, abs(lo))
        t = (lo + hi) / 2; spd(t) ? (lo = t) : (hi = t)
    end
    return lo
end

skew(mesh) = (α(v) = reduce_primal_edges(mesh, v).values; β(v) = reduce_dual_edges(mesh, star_field(v)).values;
              dot(α(U2), β(fields.d)) - dot(α(fields.d), β(U2)))

R = Dict{Any,Any}()
logp("[P2]^2 stars on wider stencils: the symmetric recipe of H_P1 against the row-wise recipe of H_P2")
logp("   unk/edge = unknowns per edge (12 equations per edge); res = relative residual of the constraints;")
logp("   Λ = max_e Σ_j |H_ej||e_j|/|e~_e|; g = Gershgorin bound and μ = λmin of the symmetric part (relative to DEC);")
logp("   exact = max row residual on degree-2 monomials / |e~| (independent check); hodge = r_2(d), r_inf(u2)+r_inf(d);")
logp("   bound = |s_h(u2,d)|/4, the least value of r_inf(u2)+r_inf(d) for a symmetric star (Theorem ceiling)")
for mn in ("mesh2", "mesh3", "mesh4", "mesh5", "mesh6")
    mesh = VoronoiMesh("meshes/$mn.nc"); n = mesh.edges.n; h = meshsize(mesh)
    sh = skew(mesh)
    logp(@sprintf("%s  (h = %.4f, %d edges)  s_h(u2,d) = %+.3e = %+.4f h^2, bound %.2e", mn, h, n, sh, sh / h^2, abs(sh) / 4))
    R[(mn, :h)] = h; R[(mn, :skew)] = sh
    rows = n <= 3072 ? (1:n) : (1:10:n)
    for st in STENCILS
        ssize = sum(length(ring(mesh, e, st)) for e in 1:n) / n
        for recipe in (:sym, :row)
            recipe == :sym && mn == "mesh6" && st in (:t3, 4) && continue        # global solve too large
            t0 = time()
            if recipe == :sym
                H, info = try
                    symmetric_correction(mesh, 2, st)
                catch err
                    logp(@sprintf("   %-28s sym  |S| %.1f  FAILED: %s", SNAME[st], ssize, sprint(showerror, err)[1:min(end, 80)]))
                    continue
                end
                unk, res = info.unknowns / n, info.res
            else
                H, _ = rowwise_correction(mesh, hdiag(mesh), 2, st)
                unk, res = ssize, NaN
            end
            tb = time() - t0
            Λ = bigLambda(mesh, H); g = gersh_sym(mesh, H); μ = lambda_min(mesh, H)
            ex = row_exactness(mesh, H, 2, rows)
            ed = hodge_error(mesh, H, HFIELDS.d...); eu = hodge_error(mesh, H, HFIELDS.u2...)
            pe = try poisson_errors(mesh, H).l2 catch; NaN end
            R[(mn, st, recipe)] = (h = h, unk = unk, res = res, Λ = Λ, g = g, μ = μ, exact = ex, hodge = ed.l2,
                                   hinf = eu.inf + ed.inf, poisson = pe, ssize = ssize, time = tb)
            logp(@sprintf("   %-28s %s  |S| %4.1f  unk/edge %4.1f  res %8.1e  Λ %8.2e  g %+9.2e  μ %+9.2e  exact %7.1e  hodge %.2e  rinf(u2)+rinf(d) %.2e  poisson %.2e  (%.0f s)",
                          SNAME[st], recipe, ssize, unk, res, Λ, g, μ, ex, ed.l2, eu.inf + ed.inf, pe, tb))
        end
    end
    serialize(joinpath(TSDIR, "p2_symmetric.jls"), R)
    write(joinpath(TSDIR, "p2_symmetric.txt"), join(lines, "\n") * "\n")
end
logp("\nobserved orders (last two meshes with data), hodge r_2(d) / Λ / poisson")
for st in STENCILS, recipe in (:sym, :row)
    ms = [mn for mn in ("mesh2", "mesh3", "mesh4", "mesh5", "mesh6") if haskey(R, (mn, st, recipe))]
    length(ms) < 2 && continue
    a, b = R[(ms[end-1], st, recipe)], R[(ms[end], st, recipe)]
    o(x, y) = log(x / y) / log(a.h / b.h)
    logp(@sprintf("   %-28s %s  %s->%s  hodge %5.2f  Λ %+5.2f  poisson %5.2f", SNAME[st], recipe, ms[end-1], ms[end],
                  o(a.hodge, b.hodge), o(a.Λ, b.Λ), o(a.poisson, b.poisson)))
end
serialize(joinpath(TSDIR, "p2_symmetric.jls"), R)
write(joinpath(TSDIR, "p2_symmetric.txt"), join(lines, "\n") * "\n")
