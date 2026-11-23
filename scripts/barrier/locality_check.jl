# ============================================================================
#  locality_check.jl -- can a symmetric star exact on [P1]^2 be built locally?  (barrier_paper.tex, Section "Locality")
#
#      julia -t 4 scripts/barrier/locality_check.jl      (writes output/ts_data/locality.txt and .jls)
#
#  The patch identity.  Let H be symmetric and exact on [P1]^2 row by row, f, g linear fields on a disc, and P a set of
#  rows (edges with midpoint within ρ of x0).  Then
#      Σ_{j∈P} [(R f)_j (H R g)_j - (R g)_j (H R f)_j]  =  Σ_{j∈P, i∉P} H_ji K_ji,     K_ji = (Rf)_j (Rg)_i - (Rg)_j (Rf)_i,
#  because the pairs inside P cancel (H symmetric, K antisymmetric).  By exactness the left side is the mesh quantity
#      Q(P) = Σ_{j∈P} |e||e~| [ (t·f(m))(t·g(m~)) - (t·g(m))(t·f(m~)) ],
#  and the right side only involves the rows near the boundary of P.  If a rule computes H_ji from the mesh within a
#  bounded distance of e_j, e_i, moving a generator deep inside P leaves the right side unchanged; so a local rule can
#  exist only if Q(P) does not depend on the interior of P.  The derivation in barrier_paper.tex predicts
#      Q(P) = 1/2 Σ_{e∈P} |e|^3 |e~| n_e' Z n_e + (terms near the boundary of P),
#      Z = 1/12 sym( G_g G_f - G_f G_g + G_f tr G_g - G_g tr G_f ),
#  which depends on the interior unless Z = 0.  Part A moves one generator and compares dQ with dD, D = the sum above.
#  Part B measures how far the coefficients of H_P1 (global solve), H_P2 (row by row) and DEC react to that move.
# ============================================================================
include(joinpath(@__DIR__, "..", "stars", "three_stars.jl"))
empty!(lines)

"The mesh with generator v0 moved by dx (connectivity kept; the circumcentres of its triangles recomputed)."
function moved_mesh(mesh, v0, dx)
    pdg = VoronoiMeshes.get_diagram(mesh.diagram)
    g = deepcopy(pdg.generators); vt = deepcopy(pdg.vertices)
    p = g[v0]; g[v0] = typeof(p)(p.x + dx[1], p.y + dx[2])
    lx, ly = pdg.x_period, pdg.y_period
    for t in eachindex(pdg.cellsOnVertex)
        cs = pdg.cellsOnVertex[t]
        v0 in cs || continue
        c = vt[t]
        P = [_pt(closest(c, g[k], lx, ly)) for k in cs]
        a, b, cc = P
        dd = 2 * (a[1] * (b[2] - cc[2]) + b[1] * (cc[2] - a[2]) + cc[1] * (a[2] - b[2]))
        ux = (sum(abs2, a) * (b[2] - cc[2]) + sum(abs2, b) * (cc[2] - a[2]) + sum(abs2, cc) * (a[2] - b[2])) / dd
        uy = (sum(abs2, a) * (cc[1] - b[1]) + sum(abs2, b) * (a[1] - cc[1]) + sum(abs2, cc) * (b[1] - a[1])) / dd
        vt[t] = typeof(c)(ux, uy)
    end
    pd2 = VoronoiMeshes.PlanarVoronoiDiagram(g, vt, pdg.verticesOnCell, pdg.cellsOnVertex, pdg.meshDensity, lx, ly)
    return VoronoiMesh(VoronoiMeshes.VoronoiDiagram(pd2))
end

"Geometry of every edge, unwrapped in the frame of x0: (m, m~, t, n, |e|, |e~|, distance of m from x0)."
function edge_geometry(mesh, x0v)
    x0 = _pt(x0v)
    map(1:mesh.edges.n) do e
        xe = closest(x0v, mesh.edges.position[e], 1.0, 1.0)
        a, b = edge_endpoints(mesh, e, true; about = xe); v1, v2 = edge_endpoints(mesh, e, false; about = xe)
        le = norm(b - a); t = (b - a) / le; n = [-t[2], t[1]]
        (m = (a + b) / 2, mt = (v1 + v2) / 2, t = t, n = n, le = le, ld = norm(v2 - v1), r = norm((a + b) / 2 - x0))
    end
end

const GBASIS = Dict("E11" => [1.0 0; 0 0], "E12" => [0.0 1; 0 0], "E21" => [0.0 0; 1 0], "E22" => [0.0 0; 0 1],
                    "I" => [1.0 0; 0 1], "J" => [0.0 -1; 1 0], "D" => [1.0 0; 0 -1], "S" => [0.0 1; 1 0])
Zof(Gf, Gg) = (Zr = (Gg * Gf - Gf * Gg + Gf * tr(Gg) - Gg * tr(Gf)) / 12; (Zr + Zr') / 2)

"Q(P) for f = af + Gf (x - x0), g = ag + Gg (x - x0) over the edges with r < ρ, and the predicted interior sum D(P)."
function QD(geo, x0, ρ, af, Gf, ag, Gg)
    Z = Zof(Gf, Gg); Q = 0.0; Dsum = 0.0
    f(x) = af + Gf * (x - x0); g(x) = ag + Gg * (x - x0)
    for E in geo
        E.r < ρ || continue
        Q += E.le * E.ld * (dot(E.t, f(E.m)) * dot(E.t, g(E.mt)) - dot(E.t, g(E.m)) * dot(E.t, f(E.mt)))
        Dsum += 0.5 * E.le^3 * E.ld * dot(E.n, Z * E.n)
    end
    return Q, Dsum
end

"Σ_{j∈P, i∉P} H_ji K_ji (the boundary side of the patch identity)."
function crossing(H, geo, x0, ρ, af, Gf, ag, Gg)
    f(x) = af + Gf * (x - x0); g(x) = ag + Gg * (x - x0)
    Rf = [E.le * dot(E.t, f(E.m)) for E in geo]; Rg = [E.le * dot(E.t, g(E.m)) for E in geo]      # exact for linear fields
    Ht = sparse(H'); s = 0.0
    for j in eachindex(geo)
        geo[j].r < ρ || continue
        for p in nzrange(Ht, j)
            i = rowvals(Ht)[p]; geo[i].r < ρ && continue
            s += nonzeros(Ht)[p] * (Rf[j] * Rg[i] - Rg[j] * Rf[i])
        end
    end
    return s
end

"max_{entries in row bins} |dH_ji/dx| · h, binned by the distance r of the row edge from the moved generator (40 bins of width h)."
function influence(build, meshes, geo0, ε, h)
    dH = (build(meshes[1]) - build(meshes[-1])) / (2ε) * h
    bins = zeros(40)
    for (j, v) in zip(findnz(dH)[1], findnz(dH)[3])
        k = Int(floor(geo0[j].r / h)) + 1
        k <= 40 && (bins[k] = max(bins[k], abs(v)))
    end
    return bins
end

if get(ENV, "LOC_MAIN", "1") == "1"                # LOC_MAIN=0 only loads the functions (used by locality_subspace_check.jl)
R = Dict{Any,Any}()
for mn in ("mesh4", "mesh5")
    mesh = VoronoiMesh("meshes/$mn.nc"); h = meshsize(mesh)
    v0 = argmin([norm(_pt(p) .- [0.5, 0.5]) for p in mesh.cells.position]); x0v = mesh.cells.position[v0]; x0 = _pt(x0v)
    logp(@sprintf("%s  (h = %.5f)  moved generator %d at (%.3f, %.3f)", mn, h, v0, x0...))
    ε = 1e-3 * h
    meshes = Dict{Any,Any}(s => moved_mesh(mesh, v0, s .* [ε, 0.0]) for s in (-1, 1))
    merge!(meshes, Dict((:y, s) => moved_mesh(mesh, v0, s .* [0.0, ε]) for s in (-1, 1)))
    geo0 = edge_geometry(mesh, x0v)
    geos = Dict(k => edge_geometry(m, x0v) for (k, m) in meshes)
    nchanged = count(i -> abs(geos[1][i].ld - geo0[i].ld) > 1e-14 || abs(geos[1][i].le - geo0[i].le) > 1e-14, eachindex(geo0))
    logp(@sprintf("   moving it by %.1e changes %d edges, all within %.2f h", ε, nchanged,
                  maximum(geo0[i].r for i in eachindex(geo0) if abs(geos[1][i].ld - geo0[i].ld) > 1e-14 || abs(geos[1][i].le - geo0[i].le) > 1e-14) / h))

    # ---- Part A: does Q(P) depend on the interior?  dQ/dx against the prediction dD/dx
    logp("   Part A: derivative of Q(P) and of the predicted interior sum D(P) with respect to the position of the generator (× 1/h^3)")
    pairs = [("const (1,0)", [1.0, 0], zeros(2, 2), "E11", [0.0, 0], GBASIS["E11"]), ("const (0,1)", [0.0, 1], zeros(2, 2), "S", [0.0, 0], GBASIS["S"]),
             ("const (1,0)", [1.0, 0], zeros(2, 2), "J", [0.0, 0], GBASIS["J"]),
             ("I (dilation)", [0.0, 0], GBASIS["I"], "J (rotation)", [0.0, 0], GBASIS["J"]),
             ("I (dilation)", [0.0, 0], GBASIS["I"], "D (shear)", [0.0, 0], GBASIS["D"]),
             ("I (dilation)", [0.0, 0], GBASIS["I"], "S (shear)", [0.0, 0], GBASIS["S"]),
             ("J (rotation)", [0.0, 0], GBASIS["J"], "D (shear)", [0.0, 0], GBASIS["D"]),
             ("D (shear)", [0.0, 0], GBASIS["D"], "S (shear)", [0.0, 0], GBASIS["S"]),
             ("E11", [0.0, 0], GBASIS["E11"], "E12", [0.0, 0], GBASIS["E12"]),
             ("E12", [0.0, 0], GBASIS["E12"], "E22", [0.0, 0], GBASIS["E22"])]
    for ρ in (0.15, 0.3)
        for (nf, af, Gf, ng, ag, Gg) in pairs
            d(k1, k2) = begin
                q1, d1 = QD(geos[k1], x0, ρ, af, Gf, ag, Gg); q2, d2 = QD(geos[k2], x0, ρ, af, Gf, ag, Gg)
                ((q1 - q2) / (2ε), (d1 - d2) / (2ε))
            end
            (qx, dx_), (qy, dy_) = d(1, -1), d((:y, 1), (:y, -1))
            logp(@sprintf("      ρ = %.2f  f = %-13s g = %-13s  dQ/dx = %+.4e  dD/dx = %+.4e   dQ/dy = %+.4e  dD/dy = %+.4e", ρ, nf, ng,
                          qx / h^3, dx_ / h^3, qy / h^3, dy_ / h^3))
            R[(mn, :A, ρ, nf, ng)] = (qx = qx / h^3, dx = dx_ / h^3, qy = qy / h^3, dy = dy_ / h^3)
        end
    end

    # ---- the identity itself, on the global symmetric H_P1 and on the row-wise H_P2
    stars, _ = three_stars(mesh)
    for (name, H) in stars
        for ρ in (0.15, 0.3)
            q, _ = QD(geo0, x0, ρ, [0.0, 0], GBASIS["I"], [0.0, 0], GBASIS["D"])
            c = crossing(H, geo0, x0, ρ, [0.0, 0], GBASIS["I"], [0.0, 0], GBASIS["D"])
            # the left side computed directly from H (no exactness assumed)
            f(x) = GBASIS["I"] * (x - x0); g(x) = GBASIS["D"] * (x - x0)
            Rf = [E.le * dot(E.t, f(E.m)) for E in geo0]; Rg = [E.le * dot(E.t, g(E.m)) for E in geo0]
            HRf = H * Rf; HRg = H * Rg
            lhs = sum(Rf[j] * HRg[j] - Rg[j] * HRf[j] for j in eachindex(geo0) if geo0[j].r < ρ)
            logp(@sprintf("   identity, %-5s ρ = %.2f, f = I, g = D:  left side %+.6e   Q(P) %+.6e   crossing terms %+.6e", name, ρ, lhs, q, c))
        end
    end

    # ---- Part B: how far does the move reach into the coefficients?
    logp("   Part B: max |dH_ji/dx| · h over the entries whose row edge lies at distance r from the moved generator")
    for (name, build) in (("H_DEC", m -> hdiag(m)), ("H_P1", m -> symmetric_correction(m, 1, 3)[1]),
                          ("H_P2", m -> rowwise_correction(m, hdiag(m), 2, 3)[1]))
        t0 = time(); Hp = build(meshes[1]); Hm = build(meshes[-1])
        dH = (Hp - Hm) / (2ε) * h
        bins = zeros(40)
        rows, cols, vals = findnz(dH)
        for (j, v) in zip(rows, vals)
            k = Int(floor(geo0[j].r / h)) + 1
            k <= 40 && (bins[k] = max(bins[k], abs(v)))
        end
        R[(mn, :B, name)] = (h = h, bins = bins)
        logp(@sprintf("      %-5s (%.0f s): r/h = 0..%d :  ", name, time() - t0, 39) * join([@sprintf("%.1e", b) for b in bins[1:2:end]], " "))
    end
end
serialize(joinpath(TSDIR, "locality.jls"), R)
write(joinpath(TSDIR, "locality.txt"), join(lines, "\n") * "\n")
end
