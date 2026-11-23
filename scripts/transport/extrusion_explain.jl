# ============================================================================
#  extrusion_explain.jl -- figures and a worked example that explain, step by step, the extrusion discretisation of
#  the contraction (lie_derivative_note / hodge_thesis, transport sections)
#
#      julia -t 4 scripts/transport/extrusion_explain.jl
#        fig_ex_density.png     one Voronoi edge of mesh 3 in the cellular flow: (a) geometry and the swept region,
#                               (b) the stencil of the upwind cell, (c) the normal velocity along the edge, (d) the density
#                               along the edge: exact, cubic reconstruction from each side, quadratic
#        fig_ex_signchange.png  an edge where the normal velocity changes sign: two swept regions, two upwind cells
#        fig_ex_oneform.png     the two contractions of the 1-form Lie derivative: extrusion of a Delaunay edge (vorticity from
#                               the upwind triangle) and of a generator (the triangle that contains −u)
#        ts_data/extrusion_example.txt   every number of the worked example
# ============================================================================
include(joinpath(@__DIR__, "extrusion_oneform.jl"))
include(joinpath(@__DIR__, "..", "shared", "thesis_plot_helpers.jl"))
const LOGX = String[]
sx(s) = (println(s); push!(LOGX, s))
const ORL, ORD, BLL, BLD = colorant"#fbd3c0", colorant"#eb6834", colorant"#cde2fb", colorant"#2a78d6"
ufun = cellular                                                           # the flow
const RHOC = Ref((0.42, 0.40))                                           # centre of the density bump, set next to edge A below
ρfun(x) = 0.2 + bump(x; x0 = RHOC[], σ = 0.12)                            # the density

mn = "mesh3"
G = lie_setup(mn; cfv = (), upw = ()); extrusion_setup!(G; p_rho = 3, rings = 2, constrained = true); oneform_setup!(G)
mesh = G["mesh"]; h = G["h"]; ld = G["ld"]; v = reduce_primal_edges(mesh, ufun).values
V2(x) = TensorsLite.Vec(x = x[1], y = x[2])
unwrap(ref, p) = _vec2(closest(V2(ref), p, 1.0, 1.0))

"Constrained (mean-preserving) or plain least-squares polynomial from the cell averages around cell c0, evaluated at xs."
function recon_rows(c0, xs, m, s; p = 3, rings = 2, constrained = true)
    α = monomials(p); nα = length(α); bq, wq = tri_rule(5)
    φ(x) = [((x[1] - m[1]) / s)^a * ((x[2] - m[2]) / s)^c for (a, c) in α]
    S = cell_rings(mesh, [c0], rings); Φ = zeros(length(S), nα)
    for (i, c) in enumerate(S)
        Ac = 0.0
        for (P1, P2, P3) in cell_fan(mesh, c, m)
            At = abs(_cross2(P2 - P1, P3 - P1)) / 2; Ac += At
            for (bb, ww) in zip(bq, wq); Φ[i, :] .+= (ww * At) .* φ(bb[1] .* P1 .+ bb[2] .* P2 .+ bb[3] .* P3); end
        end
        Φ[i, :] ./= Ac
    end
    if constrained
        i0 = findfirst(==(c0), S); oth = [i for i in eachindex(S) if i != i0]; Mp = pinv(Φ[oth, 2:end] .- Φ[i0:i0, 2:end])
        ws = map(xs) do x
            r = Mp' * (φ(x)[2:end] .- Φ[i0, 2:end]); wv = zeros(length(S)); wv[oth] .= r; wv[i0] = 1 - sum(r); wv
        end
        return S, ws
    else
        F = qr(Φ); Qm = Matrix(F.Q)[:, 1:nα]; Rm = UpperTriangular(F.R); return S, [Qm * (Rm' \ φ(x)) for x in xs]
    end
end
"Points along the Voronoi edge e for s ∈ [−|ẽ|/2, |ẽ|/2] (s = 0 at its midpoint, direction n_e)."
function dual_line(e, ns)
    v1, v2, m = dual_edge(mesh, e); n = (v2 .- v1) ./ norm(v2 .- v1); L = norm(v2 .- v1)
    ss = range(-L / 2, L / 2; length = ns); return ss, [m .+ s .* n for s in ss], m, n, L
end
"u♯·t_e along ẽ (quadratic least-squares reconstruction of u from the circulations on the stencil of e)."
function usharp_t(e, xs)
    S = tri_layers(mesh, e, 3); m = dual_edge(mesh, e)[3]; t = _unit_t(mesh, e)
    rows = circulation_lsq_rows(mesh, m, norm(-(edge_endpoints(mesh, e, true)...)), S, 2, [(x, t) for x in xs])
    return [dot(r, v[S]) for r in rows]
end
"Draw the Voronoi cells and Delaunay edges near the point c (window half-width w)."
function draw_background!(ax, c, w; delaunay = true)
    for cc in 1:mesh.cells.n
        g = unwrap(c, mesh.cells.position[cc]); maximum(abs.(g .- c)) > w + 2h && continue
        P = [Point2f(t[2]...) for t in cell_fan(mesh, cc, c)]; push!(P, P[1])
        lines!(ax, P; color = colorant"#bdbcb7", linewidth = 1)
    end
    if delaunay
        for e in 1:mesh.edges.n
            a, b = edge_endpoints(mesh, e, true; about = V2(c)); a = _vec2(a); b = _vec2(b)
            maximum(abs.((a .+ b) ./ 2 .- c)) > w + h && continue
            lines!(ax, [Point2f(a...), Point2f(b...)]; color = colorant"#e4e3df", linewidth = 0.8, linestyle = :dash)
        end
    end
    xlims!(ax, c[1] - w, c[1] + w); ylims!(ax, c[2] - w, c[2] + w)
end
fillcell!(ax, cc, ref, col) = poly!(ax, [Point2f(t[2]...) for t in cell_fan(mesh, cc, ref)]; color = col, strokewidth = 0)

# ---------------------------------------------------------------------------- choose the edges
function wexact(e)
    xs = dual_gauss(mesh, e)[1]; t = _unit_t(mesh, e)
    return [dot(ufun(x), t) for x in xs]
end
cands = [e for e in 1:mesh.edges.n if all(0.3 .< _vec2(mesh.edges.position[e]) .< 0.7)]
eA = argmax(e -> (w = wexact(e); minimum(w) > 0 ? (maximum(w) - minimum(w)) / maximum(w) * min(1, minimum(w) / 0.3) : -1.0), cands)
eB = argmax(e -> (w = wexact(e); minimum(w) < 0 < maximum(w) ? min(-minimum(w), maximum(w)) : -1.0), cands)
let pA = _vec2(mesh.edges.position[eA]), tA = _unit_t(mesh, eA)
    RHOC[] = Tuple(pA .+ 0.09 .* tA .+ 0.03 .* [-tA[2], tA[1]])            # downstream of the edge, so ρ varies across it
end
ρ̄ = cell_averages(mesh, ρfun)

# ---------------------------------------------------------------------------- the worked example (edge A)
let e = eA
    a, b = Int.(mesh.edges.cells[e]); xs, wg_, L = dual_gauss(mesh, e); wg = collect(wg_); t = _unit_t(mesh, e)
    p1, p2 = edge_endpoints(mesh, e, true); le = norm(p2 - p1); δ = dot(dual_edge(mesh, e)[3] .- (_vec2(p1) .+ _vec2(p2)) ./ 2, (dual_edge(mesh, e)[2] .- dual_edge(mesh, e)[1]) ./ L)
    Uex = reduce_dual_edges(mesh, star_field(ufun)).values[e]; Ud = (G["H_DEC"] * v)[e]; U2 = (G["H_P2"] * v)[e]
    q = [(G["Qg"][g] * v)[e] for g in 1:3]; mq = sum(wg .* q) / 2; ŵ = q .- mq .+ U2 / L
    wex = [dot(ufun(x), t) for x in xs]
    side = [w > 0 ? 1 : 2 for w in ŵ]; up = [side[g] == 1 ? a : b for g in 1:3]
    ρa = [(G["R1g"][g] * ρ̄)[e] for g in 1:3]; ρb = [(G["R2g"][g] * ρ̄)[e] for g in 1:3]; ρup = [side[g] == 1 ? ρa[g] : ρb[g] for g in 1:3]
    F = L / 2 * sum(wg .* ŵ .* ρup)
    Fstraight = L / 2 * sum(wg .* (U2 / L) .* ρup); FDEC = L / 2 * sum(wg .* (q .- mq .+ Ud / L) .* ρup)
    _, xs5, _, _, _ = dual_line(e, 2); v1, v2, m = dual_edge(mesh, e)
    Fex = sum((w5 / 2) * L * ρfun(m .+ (ξ / 2) .* (v2 .- v1)) * dot(ufun(m .+ (ξ / 2) .* (v2 .- v1)), t) for (ξ, w5) in _GL5)
    sx("Worked example: mesh 3, edge $e between cells a = $a and b = $b; flow = cellular, density = 0.2 + bump(x0, σ = 0.12), x0 = $(round.(RHOC[], digits = 4))")
    sx(@sprintf("   |e| = %.5f  |ẽ| = %.5f  offset δ = %.5f   circulation v_e = %+.6f", le, L, δ, v[e]))
    sx(@sprintf("   swept area per unit time  ∫_ẽ u·t:  exact %+.6f   H_DEC v %+.6f (error %.1e)   H_P2 v %+.6f (error %.1e)", Uex, Ud, abs(Ud - Uex) / abs(Uex), U2, abs(U2 - Uex) / abs(Uex)))
    for g in 1:3
        sx(@sprintf("   Gauss point %d (s = %+.5f, weight %.4f): u·t exact %+.5f  u♯·t %+.5f  ŵ %+.5f  upwind cell %d (%s)  ρ exact %.5f  ρ̂_a %.5f  ρ̂_b %.5f",
                    g, _GL3[g][1] * L / 2, wg[g], wex[g], q[g], ŵ[g], up[g], side[g] == 1 ? "a" : "b", ρfun(xs[g]), ρa[g], ρb[g]))
    end
    sx(@sprintf("   mean of u♯·t over ẽ %+.5f, replaced by (H_P2 v)/|ẽ| = %+.5f", mq, U2 / L))
    sx(@sprintf("   cell averages: ρ̄_a = %.5f  ρ̄_b = %.5f", ρ̄[a], ρ̄[b]))
    sx(@sprintf("   mass flux F_e: exact %+.7f   extrusion (cubic, H_P2) %+.7f (error %.1e)   straight extrusion %+.7f (%.1e)   with H_DEC %+.7f (%.1e)",
                Fex, F, abs(F - Fex) / abs(Fex), Fstraight, abs(Fstraight - Fex) / abs(Fex), FDEC, abs(FDEC - Fex) / abs(Fex)))
    sx(@sprintf("   two-cell average (standard DEC/TRiSK): ½(ρ̄_a + ρ̄_b)(H_DEC v) = %+.7f (error %.1e)", (ρ̄[a] + ρ̄[b]) / 2 * Ud, abs((ρ̄[a] + ρ̄[b]) / 2 * Ud - Fex) / abs(Fex)))
end

# ---------------------------------------------------------------------------- figure 1: edge A
let e = eA
    a, b = Int.(mesh.edges.cells[e]); ss, qs, m, n, L = dual_line(e, 41); t = _unit_t(mesh, e)
    xa, xb = edge_endpoints(mesh, e, true; about = V2(m)); xa = _vec2(xa); xb = _vec2(xb)
    f = Figure(size = (1500, 1250), fontsize = 17); w = 1.9h
    # (a) geometry + swept region
    ax = Axis(f[1, 1]; aspect = DataAspect(), title = "(a) the Voronoi edge ẽ and the region it sweeps backwards", titlealign = :left); hidedecorations!(ax)
    fillcell!(ax, a, m, colorant"#fdeee6"); fillcell!(ax, b, m, colorant"#eef4fc")
    draw_background!(ax, m, w)
    τ = 0.55L / maximum(norm(ufun(x)) for x in qs)
    reg = vcat([Point2f(x...) for x in qs], reverse([Point2f((x .- τ .* ufun(x))...) for x in qs]))
    poly!(ax, reg; color = (ORD, 0.45), strokecolor = ORD, strokewidth = 1.5)
    lines!(ax, [Point2f(xa...), Point2f(xb...)]; color = INK2, linewidth = 2, linestyle = :dash)
    lines!(ax, [Point2f(qs[1]...), Point2f(qs[end]...)]; color = INK, linewidth = 4)
    xs, _, _ = dual_gauss(mesh, e)
    scatter!(ax, [Point2f(x...) for x in xs]; color = INK, markersize = 11)
    arrows2d!(ax, [Point2f(x...) for x in xs], [Vec2f(ufun(x)...) for x in xs]; lengthscale = 0.6τ, color = ORD, shaftwidth = 2.5, tipwidth = 12, tiplength = 12)
    scatter!(ax, [Point2f(xa...), Point2f(xb...)]; color = INK, markersize = 13)
    text!(ax, xa[1], xa[2]; text = "  x_a", align = (:left, :top), fontsize = 18); text!(ax, xb[1], xb[2]; text = "  x_b", align = (:left, :bottom), fontsize = 18)
    text!(ax, qs[end][1], qs[end][2]; text = " ẽ", align = (:left, :bottom), fontsize = 20, font = :bold)
    text!(ax, (xa .+ xb)[1] / 2, (xa .+ xb)[2] / 2; text = "e ", align = (:right, :bottom), fontsize = 18, color = INK2)
    # (b) stencil of the upwind cell
    ax = Axis(f[1, 2]; aspect = DataAspect(), title = "(b) upwind cell a and its two rings: the cubic stencil", titlealign = :left); hidedecorations!(ax)
    for c in cell_rings(mesh, [a], 2); fillcell!(ax, c, m, ORL); end
    for c in cell_rings(mesh, [a], 1); fillcell!(ax, c, m, colorant"#f5a37c"); end
    fillcell!(ax, a, m, ORD)
    draw_background!(ax, m, 2.6h; delaunay = false)
    lines!(ax, [Point2f(qs[1]...), Point2f(qs[end]...)]; color = INK, linewidth = 4)
    text!(ax, xa[1], xa[2]; text = "a", align = (:center, :center), fontsize = 20, font = :bold, color = :white)
    text!(ax, xb[1], xb[2]; text = "b", align = (:center, :center), fontsize = 20, font = :bold)
    # (c) normal velocity along ẽ
    ax = Axis(f[2, 1]; xlabel = "position s along ẽ", ylabel = "u · t_e", title = "(c) the extrusion speed along ẽ", axstyle_th...)
    wex = [dot(ufun(x), t) for x in qs]; us = usharp_t(e, qs)
    xs3, wg_, _ = dual_gauss(mesh, e); wg = collect(wg_); q3 = [(G["Qg"][g] * v)[e] for g in 1:3]; mq = sum(wg .* q3) / 2
    U2 = (G["H_P2"] * v)[e]; Ud = (G["H_DEC"] * v)[e]
    lines!(ax, collect(ss), wex; color = INK, linewidth = 2.5, label = "exact u·t")
    lines!(ax, collect(ss), us .- mq .+ U2 / L; color = ORD, linewidth = 2.5, linestyle = :dash, label = "ŵ: u♯·t with its mean set to (H_P2 v)/|ẽ|")
    hlines!(ax, [U2 / L]; color = ORD, linewidth = 1.5, linestyle = :dot, label = "(H_P2 v)/|ẽ|: straight extrusion")
    hlines!(ax, [Ud / L]; color = C_DEC, linewidth = 1.5, linestyle = :dot, label = "(H_DEC v)/|ẽ|")
    sg = [_GL3[g][1] * L / 2 for g in 1:3]
    scatter!(ax, sg, q3 .- mq .+ U2 / L; color = ORD, markersize = 13)
    axislegend(ax; position = :rb, framevisible = false, labelsize = 14)
    # (d) density along ẽ
    ax = Axis(f[2, 2]; xlabel = "position s along ẽ", ylabel = "ρ", title = "(d) the density on ẽ, reconstructed from each side", axstyle_th...)
    sc = norm(xb .- xa)
    _, ra = recon_rows(a, qs, m, sc); _, rb = recon_rows(b, qs, m, sc); _, rq = recon_rows(a, qs, m, sc; p = 2, rings = 1, constrained = false)
    Sa = cell_rings(mesh, [a], 2); Sb = cell_rings(mesh, [b], 2); Sq = cell_rings(mesh, [a], 1)
    lines!(ax, collect(ss), [ρfun(x) for x in qs]; color = INK, linewidth = 2.5, label = "exact ρ")
    lines!(ax, collect(ss), [dot(r, ρ̄[Sa]) for r in ra]; color = ORD, linewidth = 2.5, linestyle = :dash, label = "cubic from cell a (upwind)")
    lines!(ax, collect(ss), [dot(r, ρ̄[Sb]) for r in rb]; color = BLD, linewidth = 2, linestyle = :dashdot, label = "cubic from cell b")
    lines!(ax, collect(ss), [dot(r, ρ̄[Sq]) for r in rq]; color = C_DEC, linewidth = 1.8, linestyle = :dot, label = "quadratic from cell a, one ring")
    hlines!(ax, [(ρ̄[a] + ρ̄[b]) / 2]; color = VIOLET, linewidth = 1.5, label = "two-cell average (standard DEC)")
    scatter!(ax, sg, [(G["R1g"][g] * ρ̄)[e] for g in 1:3]; color = ORD, markersize = 13)
    axislegend(ax; position = :lt, framevisible = false, labelsize = 14)
    CairoMakie.save(thesis_out("fig_ex_density.png"), f; px_per_unit = 2)
end

# ---------------------------------------------------------------------------- figure 2: sign change (edge B)
let e = eB
    a, b = Int.(mesh.edges.cells[e]); ss, qs, m, n, L = dual_line(e, 61); t = _unit_t(mesh, e)
    xa, xb = edge_endpoints(mesh, e, true; about = V2(m)); xa = _vec2(xa); xb = _vec2(xb)
    f = Figure(size = (1500, 620), fontsize = 17)
    ax = Axis(f[1, 1]; aspect = DataAspect(), title = "(a) u·t changes sign on ẽ: two swept regions (normal part)", titlealign = :left); hidedecorations!(ax)
    fillcell!(ax, a, m, colorant"#fdeee6"); fillcell!(ax, b, m, colorant"#eef4fc")
    draw_background!(ax, m, 1.9h)
    wex = [dot(ufun(x), t) for x in qs]; τ = 0.8L / maximum(abs.(wex))
    for sgn in (1, -1)                                  # only the displacement across ẽ is drawn: it carries the swept area
        idx = findall(w -> sgn * w > 0, wex); isempty(idx) && continue
        seg = qs[idx]; reg = vcat([Point2f(x...) for x in seg], reverse([Point2f((x .- τ * dot(ufun(x), t) .* t)...) for x in seg]))
        poly!(ax, reg; color = (sgn > 0 ? ORD : BLD, 0.45), strokecolor = sgn > 0 ? ORD : BLD, strokewidth = 1.5)
    end
    lines!(ax, [Point2f(xa...), Point2f(xb...)]; color = INK2, linewidth = 2, linestyle = :dash)
    lines!(ax, [Point2f(qs[1]...), Point2f(qs[end]...)]; color = INK, linewidth = 4)
    xs, _, _ = dual_gauss(mesh, e)
    arrows2d!(ax, [Point2f(x...) for x in xs], [Vec2f(ufun(x)...) for x in xs]; lengthscale = 0.9h / maximum(norm(ufun(x)) for x in xs), color = INK, shaftwidth = 2.5, tipwidth = 12, tiplength = 12)
    scatter!(ax, [Point2f(xa...), Point2f(xb...)]; color = INK, markersize = 13)
    text!(ax, xa[1], xa[2]; text = "  x_a", align = (:left, :top), fontsize = 18); text!(ax, xb[1], xb[2]; text = "  x_b", align = (:left, :bottom), fontsize = 18)
    ax = Axis(f[1, 2]; xlabel = "position s along ẽ", ylabel = "u · t_e", title = "(b) each Gauss point takes the upwind side of its own sign", axstyle_th...)
    q3 = [(G["Qg"][g] * v)[e] for g in 1:3]; wg = collect(dual_gauss(mesh, e)[2]); mq = sum(wg .* q3) / 2; U2 = (G["H_P2"] * v)[e]
    us = usharp_t(e, qs)
    hlines!(ax, [0.0]; color = INK2, linewidth = 1)
    lines!(ax, collect(ss), wex; color = INK, linewidth = 2.5, label = "exact u·t")
    lines!(ax, collect(ss), us .- mq .+ U2 / L; color = INK2, linewidth = 2.5, linestyle = :dash, label = "ŵ")
    sg = [_GL3[g][1] * L / 2 for g in 1:3]; wh = q3 .- mq .+ U2 / L
    scatter!(ax, sg, wh; color = [w > 0 ? ORD : BLD for w in wh], markersize = 16)
    for g in 1:3
        text!(ax, sg[g], wh[g]; text = wh[g] > 0 ? "  from a" : "  from b", align = (:left, :center), fontsize = 15, color = wh[g] > 0 ? ORD : BLD)
    end
    axislegend(ax; position = :lt, framevisible = false, labelsize = 14)
    CairoMakie.save(thesis_out("fig_ex_signchange.png"), f; px_per_unit = 2)
    sx(@sprintf("Sign-change example: edge %d, ŵ at the Gauss points %+.4f %+.4f %+.4f", e, wh...))
end

# ---------------------------------------------------------------------------- figure 3: 1-form contractions
let
    f = Figure(size = (1500, 680), fontsize = 17)
    # (a) Delaunay edge e1: vorticity contraction
    candE = [e for e in cands if (P = primal_edge(mesh, e); minimum(abs(dot(ufun(P[3] .+ (ξ / 2) .* (P[2] .- P[1])), P[5])) for (ξ, _) in _GL3) > 0.25)]
    e1 = candE[1]; P = primal_edge(mesh, e1); a1, b1, m1, t1, n1, l1 = P
    un = dot(ufun(m1), n1); Ts = [Int(T) for T in mesh.edges.vertices[e1]]
    side = [dot(sum(tri_vertices(mesh, T, m1)) ./ 3 .- m1, n1) for T in Ts]; Tup = un > 0 ? Ts[argmin(side)] : Ts[argmax(side)]
    ax = Axis(f[1, 1]; aspect = DataAspect(), title = "(a) i_u dβ on a Delaunay edge: vorticity from the upwind triangle", titlealign = :left); hidedecorations!(ax)
    for T in tri_edge_ring2(mesh, Tup)
        poly!(ax, [Point2f(p...) for p in tri_vertices(mesh, T, m1)]; color = T == Tup ? ORD : ORL, strokecolor = :white, strokewidth = 1)
    end
    for e in 1:mesh.edges.n
        p, q = edge_endpoints(mesh, e, true; about = V2(m1)); p = _vec2(p); q = _vec2(q)
        maximum(abs.((p .+ q) ./ 2 .- m1)) > 2.6h && continue
        lines!(ax, [Point2f(p...), Point2f(q...)]; color = colorant"#bdbcb7", linewidth = 1)
    end
    qs = [m1 .+ s .* (b1 .- a1) for s in range(-0.5, 0.5; length = 31)]; τ = 0.6l1 / maximum(norm(ufun(x)) for x in qs)
    reg = vcat([Point2f(x...) for x in qs], reverse([Point2f((x .- τ .* ufun(x))...) for x in qs]))
    poly!(ax, reg; color = (BLD, 0.35), strokecolor = BLD, strokewidth = 1.5)
    lines!(ax, [Point2f(a1...), Point2f(b1...)]; color = INK, linewidth = 4)
    g3 = [m1 .+ (ξ / 2) .* (b1 .- a1) for (ξ, _) in _GL3]
    arrows2d!(ax, [Point2f(x...) for x in g3], [Vec2f(ufun(x)...) for x in g3]; lengthscale = 0.6τ, color = BLD, shaftwidth = 2.5, tipwidth = 12, tiplength = 12)
    xlims!(ax, m1[1] - 2.3h, m1[1] + 2.3h); ylims!(ax, m1[2] - 2.3h, m1[2] + 2.3h)
    # (b) generator: point contraction
    c = argmin(cc -> norm(_vec2(mesh.cells.position[cc]) .- [0.45, 0.42]), 1:mesh.cells.n); xc = _vec2(mesh.cells.position[c]); d = -ufun(xc)
    ax = Axis(f[1, 2]; aspect = DataAspect(), title = "(b) i_u β at a generator: the triangle that contains −u", titlealign = :left); hidedecorations!(ax)
    Tsel = 0
    for T in mesh.cells.vertices[c]
        Pt = tri_vertices(mesh, Int(T), xc); oth = [p for p in Pt if norm(p .- xc) > 1e-12]
        r1, r2 = oth[1] .- xc, oth[2] .- xc
        inside = _cross2(r1, d) * _cross2(r1, r2) >= 0 && _cross2(r2, d) * _cross2(r2, r1) >= 0
        inside && (Tsel = Int(T))
        poly!(ax, [Point2f(p...) for p in Pt]; color = inside ? ORD : colorant"#f7f6f3", strokecolor = INK2, strokewidth = 1)
    end
    S = sort(collect(union((Set(tri_layers(mesh, Int(e), 2)) for e in mesh.vertices.edges[Tsel])...)))
    for e in S
        p, q = edge_endpoints(mesh, e, true; about = V2(xc)); lines!(ax, [Point2f(_vec2(p)...), Point2f(_vec2(q)...)]; color = ORD, linewidth = 2.2)
    end
    scatter!(ax, [Point2f(xc...)]; color = INK, markersize = 14)
    arrows2d!(ax, [Point2f(xc...)], [Vec2f(ufun(xc)...)]; lengthscale = 1.2h / norm(ufun(xc)), color = INK, shaftwidth = 3, tipwidth = 14, tiplength = 14)
    arrows2d!(ax, [Point2f(xc...)], [Vec2f(d...)]; lengthscale = 1.2h / norm(d), color = BLD, shaftwidth = 3, tipwidth = 14, tiplength = 14)
    text!(ax, (xc .+ 1.25h .* ufun(xc) ./ norm(ufun(xc)))...; text = " u", fontsize = 20, font = :bold)
    text!(ax, (xc .+ 1.25h .* d ./ norm(d))...; text = " −u", fontsize = 20, font = :bold, color = BLD)
    xlims!(ax, xc[1] - 2.6h, xc[1] + 2.6h); ylims!(ax, xc[2] - 2.6h, xc[2] + 2.6h)
    CairoMakie.save(thesis_out("fig_ex_oneform.png"), f; px_per_unit = 2)
    sx("1-form figure: Delaunay edge $e1 (upwind triangle $Tup), generator $c (upwind triangle $Tsel, stencil of $(length(S)) edges)")
end
write(joinpath(TSDIR, "extrusion_example.txt"), join(LOGX, "\n") * "\n")
println("extrusion explanation written")
