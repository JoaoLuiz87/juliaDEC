# ============================================================================
#  three_stars_run.jl -- all numbers of three_stars_report.tex.
#
#      julia -t 4 scripts/stars/three_stars_run.jl      (writes output/ts_data/results.jls and summary.txt)
#
#  For meshes 1-6 and the stars H_DEC, H_P1, H_P2: construction diagnostics, Hodge errors, Laplacian truncation error,
#  Poisson errors, Darcy errors (Hirani et al. eq. 4.6 on the torus) and both sides of the error bounds of the report
#  (Theorem [bounds]): ||d0 e||_{H_DEC} <= ||ε(∇p)||_{H_DEC^-1}/g  and  ||e_f||_{H_DEC} <= ||ε(Jv)||_{H_DEC^-1}/g,
#  with g the Gershgorin lower bound of λmin(sym part).  Spectra on meshes 3-4; solutions on mesh 4; the worked
#  example of Section 2 on mesh 3, edge 17.
# ============================================================================
include(joinpath(@__DIR__, "three_stars.jl"))
empty!(lines)

R = Dict{Any,Any}()
for mn in MESHES
    mesh = VoronoiMesh("meshes/$mn.nc"); h = meshsize(mesh)
    t0 = time(); stars, info = three_stars(mesh); tb = time() - t0
    tris, D1, _ = triangles_ccw(mesh); At = [T.area for T in tris]
    d0 = d0_matrix_primal(mesh); hd = diag(hdiag(mesh))
    exs = darcy_smooth(mesh, tris); exq = darcy_patch(mesh, tris)
    αgrad = reduce_primal_edges(mesh, gradp).values; βgrad = reduce_dual_edges(mesh, star_field(gradp)).values
    βw = reduce_dual_edges(mesh, star_field(x -> _ROT90 * VD(x))).values
    R[(mn, :h)] = h; R[(mn, :info)] = info; R[(mn, :time)] = tb
    R[(mn, :size)] = (cells = mesh.cells.n, edges = mesh.edges.n, tris = mesh.vertices.n)
    logp(@sprintf("%s  h = %.4f  (%d cells, %d edges)  stars built in %.1f s;  P1 constraint residual %.1e", mn, h,
                  mesh.cells.n, mesh.edges.n, tb, info.res_p1))
    for name in STARS
        H = stars[name]; r = Dict{Symbol,Any}()
        r[:asym] = norm(H - H') / norm(H); r[:Λ] = bigLambda(mesh, H); r[:g] = gersh_sym(mesh, H)
        r[:nnz] = nnz(H) / size(H, 1); r[:corr] = maximum(abs.(H - spdiagm(hd))) / maximum(hd)
        r[:hd] = hodge_error(mesh, H, HFIELDS.d...); r[:hu2] = hodge_error(mesh, H, HFIELDS.u2...)
        r[:tau] = laplacian_truncation(mesh, H)
        # Poisson
        P = poisson_errors(mesh, H)
        e = P.sol .- P.exact; ge = d0 * e
        εp = H * αgrad .- βgrad
        r[:poisson] = (inf = P.inf, l2 = P.l2, grad = P.grad,
                       bound_lhs = sqrt(dot(ge, hd .* ge)), bound_rhs = sqrt(sum(εp .^ 2 ./ hd)) / r[:g])
        # Darcy, smooth solution (Hirani 6.2 on the torus)
        fh, ph = solve_darcy(H, D1, exs.G, exs.Φ, At)
        D = darcy_errors(mesh, tris, fh, ph, exs; pfun_ = PD, vnorm = TP)
        ef = fh .- exs.f; εw = H * exs.f .- βw
        r[:darcy] = merge(D, (bound_lhs = sqrt(sum(hd .* ef .^ 2)), bound_rhs = sqrt(sum(εw .^ 2 ./ hd)) / r[:g]))
        # Darcy patch test (Hirani 6.1 on the torus: constant velocity driven by a constant body force)
        fq, pq = solve_darcy(H, D1, exq.G, exq.Φ, At)
        r[:patch] = (flux = maximum(abs.(fq .- exq.f)), p = maximum(pq) - minimum(pq))
        if mn == "mesh4"
            r[:sol] = (poisson = P.sol, poisson_exact = P.exact, darcy_p = ph .- sum(At .* ph) / sum(At),
                       darcy_v = triangle_velocity(mesh, tris, fh), darcy_f = fh)
        end
        if mn in ("mesh3", "mesh4")
            r[:spec] = star_spectra(mesh, H)
        end
        R[(mn, name)] = r
        logp(@sprintf("   %-5s asym %.1e  Λ %5.2f  g %.3f  Hodge r(d) %.2e/%.2e  tau %.2e  Poisson %.2e  Darcy flux %.2e p %.2e (Hirani %.2e)  patch %.0e",
                      name, r[:asym], r[:Λ], r[:g], r[:hd].inf, r[:hd].l2, r[:tau], P.inf, D.flux, D.p_inf, D.p_hirani,
                      max(r[:patch].flux, r[:patch].p)))
    end
    if mn == "mesh4"
        R[(mn, :exact)] = (darcy_p = exs.p .- sum(At .* exs.p) / sum(At),
                           darcy_v = reduce(vcat, [VD(sum(T.pts) ./ 3)' for T in tris]))
    end
    if mn == "mesh3"                                                      # the worked example of Section 2
        e = 17; x0 = _pt(mesh.edges.position[e]); s = edge_lengths(mesh, true)[e]
        S = ring(mesh, e, 3)
        q1, q2 = edge_endpoints(mesh, e, false; about = mesh.edges.position[e])
        ex = Dict{Symbol,Any}(:e => e, :S => S, :x0 => x0, :s => s, :hd => hd[e], :dual => (q1, q2))
        ex[:primal] = Dict(j => edge_endpoints(mesh, j, true; about = mesh.edges.position[e]) for j in 1:mesh.edges.n
                           if norm(_pt(closest(mesh.edges.position[e], mesh.edges.position[j], 1.0, 1.0)) .- x0) < 5s)
        for k in (1, 2)
            Ae = reduce(hcat, [(q = edge_endpoints(mesh, j, true; about = mesh.edges.position[e]); seg_moments(q[1], q[2], x0, s, k)) for j in S])'
            be = seg_moments(q1, q2, x0, s, k; rotate = true)
            ex[Symbol("A$k")] = Ae; ex[Symbol("b$k")] = be
            ex[Symbol("r$k")] = be .- hd[e] .* Ae[findfirst(==(e), S), :]
        end
        for (name, k) in (("H_P1", 1), ("H_P2", 2))
            H = stars[name]; cols = findall(!iszero, H[e, :])
            ex[Symbol("row_" * name)] = Dict(j => H[e, j] - (j == e ? hd[e] : 0.0) for j in cols)
            Afull = reduce(hcat, [(q = edge_endpoints(mesh, j, true; about = mesh.edges.position[e]); seg_moments(q[1], q[2], x0, s, k)) for j in cols])'
            ex[Symbol("check_" * name)] = maximum(abs.(Afull' * [H[e, j] for j in cols] .- seg_moments(q1, q2, x0, s, k; rotate = true)))
        end
        R[:example] = ex
    end
end
serialize(joinpath(TSDIR, "results.jls"), R)

# ------------------------------------------------------------- summary tables with observed orders (h halves each step)
function table(title, f)
    logp("\n" * title)
    logp("   star   " * join([@sprintf("%10s", m) for m in MESHES], "") * "     orders 1->2->...->6")
    for name in STARS
        v = [f(R[(m, name)]) for m in MESHES]
        logp(@sprintf("   %-6s %s     %s", name, join([@sprintf("%10.2e", x) for x in v], ""),
                      join([@sprintf("%5.2f", log2(v[i] / v[i+1])) for i in 1:length(v)-1], " ")))
    end
end
table("Hodge error r_inf(d)", r -> r[:hd].inf)
table("Hodge error r_2(d)", r -> r[:hd].l2)
table("Hodge error r_inf(u2)", r -> r[:hu2].inf)
table("Hodge error r_2(u2)", r -> r[:hu2].l2)
table("Laplacian truncation tau (max, relative)", r -> r[:tau])
table("Poisson: nodal error, max (relative)", r -> r[:poisson].inf)
table("Poisson: nodal error, L2 (relative)", r -> r[:poisson].l2)
table("Poisson: gradient error ||d0 e||_{H_DEC} (relative)", r -> r[:poisson].grad)
table("Poisson bound: lhs ||d0 e||_{H_DEC}", r -> r[:poisson].bound_lhs)
table("Poisson bound: rhs ||eps||_{H_DEC^-1}/g", r -> r[:poisson].bound_rhs)
table("Darcy: flux error ||e_f||_{H_DEC} (relative)", r -> r[:darcy].flux)
table("Darcy: flux error, max (relative to |v|inf)", r -> r[:darcy].flux_inf)
table("Darcy: pressure at circumcentres, max (relative)", r -> r[:darcy].p_inf)
table("Darcy: pressure at circumcentres, L2 (relative)", r -> r[:darcy].p_l2)
table("Darcy: pressure, Hirani's measure (p_T constant on T), L2 (relative)", r -> r[:darcy].p_hirani)
table("Darcy bound: lhs ||e_f||_{H_DEC}", r -> r[:darcy].bound_lhs)
table("Darcy bound: rhs ||eps(Jv)||_{H_DEC^-1}/g", r -> r[:darcy].bound_rhs)
logp("\nconstruction and stability")
for name in STARS
    logp(@sprintf("   %-6s asym %s", name, join([@sprintf("%9.1e", R[(m, name)][:asym]) for m in MESHES], "")))
    logp(@sprintf("   %-6s Λ    %s", name, join([@sprintf("%9.2f", R[(m, name)][:Λ]) for m in MESHES], "")))
    logp(@sprintf("   %-6s g    %s", name, join([@sprintf("%9.3f", R[(m, name)][:g]) for m in MESHES], "")))
    logp(@sprintf("   %-6s |C|  %s", name, join([@sprintf("%9.3f", R[(m, name)][:corr]) for m in MESHES], "")))
    logp(@sprintf("   %-6s nnz/row %s", name, join([@sprintf("%6.1f", R[(m, name)][:nnz]) for m in MESHES], "")))
    logp(@sprintf("   %-6s patch %s", name, join([@sprintf("%9.0e", max(R[(m, name)][:patch]...)) for m in MESHES], "")))
end
logp("   build time (s): " * join([@sprintf("%8.1f", R[(m, :time)]) for m in MESHES], ""))
for mn in ("mesh3", "mesh4"), name in STARS
    sp = R[(mn, name)][:spec]
    λ = sp.L; ωm = sqrt(maximum(abs.(λ)))
    logp(@sprintf("   spectra %s %-6s  H: Re in [%.3f, %.3f], max|Im| %.1e   L: max|Im|/max|λ| %.1e, min Re %.1e, smallest nonzero |λ| %.2f",
                  mn, name, minimum(real.(sp.H)), maximum(real.(sp.H)), maximum(abs.(imag.(sp.H))),
                  maximum(abs.(imag.(λ))) / maximum(abs.(λ)), minimum(real.(λ)), sort(abs.(λ))[2]))
end
ex = R[:example]
logp(@sprintf("\nworked example: mesh3 edge %d, |S_e| = %d, H_DEC_ee = %.4f;  row checks: P1 %.1e  P2 %.1e", ex[:e], length(ex[:S]),
              ex[:hd], ex[:check_H_P1], ex[:check_H_P2]))
logp("   DEC residual on the [P2]^2 basis: " * join([@sprintf("%+.3e", x) for x in ex[:r2]], " "))
write(joinpath(TSDIR, "summary.txt"), join(lines, "\n") * "\n")
