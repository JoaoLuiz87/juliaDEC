# ============================================================================
#  extrusion_lie.jl -- the discrete contraction as the dual of a discrete extrusion (Bossavit 2002; Desbrun-Hirani-Leok-
#  Marsden 2005 §10; Mullen et al. 2011; Eldred-Waruszewski-Roberts, SAND LDRD report, ch. 3), and the Lie derivative
#  by Cartan's formula with the exact D's.  Library (no output); builds on lie_derivative.jl.
#
#  Flowed-out definition:   ∫_S i_u α = d/dt|_{t=0} ∫_{E_t(S)} α   (E_t(S): S swept by the flow of u for time t).
#  Upwind (backward) extrusion: the swept region lies upstream of S, so α is reconstructed from the upwind cell.
#  Discretely,  I(u, α)_S = (approximate integral of α over the extrusion of S)  =  ⟨α, E_h(u) S⟩,  i.e. I = E_hᵀ:
#  the contraction is the transpose (dual) of a discrete extrusion operator on chains, and
#        L_h = D I + I D      (Cartan; exact D)        on any form degree.
#
#  WHERE THE HODGE STAR ENTERS.  The velocity is the primal 1-cochain v_e = ∫_e u·t_e (circulation along Delaunay edges).
#  The extrusion of a cell S needs the velocity transverse to S:
#    * S = Voronoi edge ẽ (contraction of a density, a dual 2-form): the swept area per unit time is ∫_ẽ u·t_e, the volume
#      flux across ẽ.  From v this is exactly what a Hodge star gives: U = H v.  H fixes the AREA of the extruded region.
#      A high-order extrusion also needs the DISTRIBUTION of u·t_e along ẽ (its shape); that comes from a reconstruction
#      u♯ of u from v.  The reconstruction itself defines a star, H_rec v = ∫_ẽ u♯·t_e.
#    * S = generator (contraction of a primal 1-form, e.g. dq for a 0-form q): the extrusion of a point needs the vector
#      u(x_c): a reconstruction u♯ from v, no Hodge star.
#    * S = Delaunay edge (contraction of the vorticity 2-form, for the 1-form Lie derivative): needs u·n_e along e, the
#      component that v does NOT carry: again u♯ (TRiSK's W ∘ H does this job), not H itself.
#
#  Functions:
#    circulation_lsq_rows   rows of the least-squares map  (circulations on a stencil) -> u♯(x)·d   (u♯ ∈ [P_p]²)
#    sharp_on_dual_edges    Q_g (g = 1:3):  (u♯·t_e)(x_g) at the Gauss points x_g of every Voronoi edge
#    sharp_at_generators    (Ux, Uy):  u♯(x_c)
#    lsq_edge_points        side-cell density polynomials evaluated at the Gauss points of ẽ (UPW2 stencils)
#    extrusion_flux         I_2(u, ρ̃) on every Voronoi edge (pointwise upwind, mean fixed by H)
#    oneform_slots          per (generator, incident triangle): reconstruction of a primal 1-form at the generator
#    extrusion_scalar_rhs   I_1(u, dq) at the generators with the upwind triangle (the 0-form Lie derivative i_u dq)
# ============================================================================
isdefined(Main, :lie_setup) || include(joinpath(@__DIR__, "lie_derivative.jl"))

_V(x) = TensorsLite.Vec(x = x[1], y = x[2])

"""
Rows r (over the stencil edges) with u♯(x)·d = Σ_f r_f v_f, for u♯ ∈ [P_p]² fitted by least squares to the circulations
v_f = ∫_f u♯·t_f on the stencil (exact for [P_p]² fields).  Monomials in (x - x0)/scale.
"""
function circulation_lsq_rows(mesh, x0, scale, stencil, p, targets)
    α = monomials(p); nα = length(α)
    φ(x) = [((x[1] - x0[1]) / scale)^a * ((x[2] - x0[2]) / scale)^b for (a, b) in α]
    M = zeros(length(stencil), 2nα)
    for (i, f) in enumerate(stencil)
        a, b = edge_endpoints(mesh, f, true; about = _V(x0))
        for (ξ, g) in _GL3
            ph = φ(a .+ ((ξ + 1) / 2) .* (b .- a)) .* (g / 2)
            M[i, 1:nα] .+= ph .* (b[1] - a[1]); M[i, nα+1:end] .+= ph .* (b[2] - a[2])
        end
    end
    F = qr(M); Qm = Matrix(F.Q)[:, 1:2nα]; Rm = UpperTriangular(F.R)
    return [Qm * (Rm' \ vcat(φ(x) .* d[1], φ(x) .* d[2])) for (x, d) in targets]
end

function _unit_t(mesh, e)
    ab = edge_endpoints(mesh, e, true)
    return (ab[2] .- ab[1]) ./ norm(ab[2] .- ab[1])
end
"Gauss points of the Voronoi edge ẽ (unwrapped about the edge position) and their weights (Σw = 2), plus |ẽ|."
function dual_gauss(mesh, e)
    v1, v2, m = dual_edge(mesh, e)
    return [m .+ (ξ / 2) .* (v2 .- v1) for (ξ, _) in _GL3], [g for (_, g) in _GL3], norm(v2 .- v1)
end

"Q_g, g = 1:3 (edges × edges):  (Q_g v)_e = (u♯ · t_e)(x_g) at the Gauss points of ẽ;  P_p reconstruction, stencil tri_layers(e, L)."
function sharp_on_dual_edges(mesh, p; L = 3)
    rows = Vector{Any}(undef, mesh.edges.n)
    Threads.@threads for e in 1:mesh.edges.n
        xs, _, _ = dual_gauss(mesh, e); t = _unit_t(mesh, e)
        S = tri_layers(mesh, e, L)
        rows[e] = (S, circulation_lsq_rows(mesh, dual_edge(mesh, e)[3], norm(-(edge_endpoints(mesh, e, true)...)), S, p, [(x, t) for x in xs]))
    end
    return ntuple(g -> sparse(reduce(vcat, [fill(e, length(rows[e][1])) for e in 1:mesh.edges.n]),
                              reduce(vcat, [rows[e][1] for e in 1:mesh.edges.n]),
                              reduce(vcat, [rows[e][2][g] for e in 1:mesh.edges.n]), mesh.edges.n, mesh.edges.n), 3)
end

"Edges of the triangles around generator c, widened by L-1 triangle layers."
cell_edge_stencil(mesh, c, L) = sort(collect(union((Set(tri_layers(mesh, Int(e), L)) for e in mesh.cells.edges[c])...)))

"(Ux, Uy) (cells × edges):  u♯(x_c) from the circulations, P_p reconstruction."
function sharp_at_generators(mesh, p; L = 2)
    rows = Vector{Any}(undef, mesh.cells.n)
    Threads.@threads for c in 1:mesh.cells.n
        x0 = _vec2(mesh.cells.position[c]); S = cell_edge_stencil(mesh, c, L)
        rows[c] = (S, circulation_lsq_rows(mesh, x0, meshsize(mesh), S, p, [(x0, [1.0, 0.0]), (x0, [0.0, 1.0])]))
    end
    I = reduce(vcat, [fill(c, length(rows[c][1])) for c in 1:mesh.cells.n]); J = reduce(vcat, [rows[c][1] for c in 1:mesh.cells.n])
    return (sparse(I, J, reduce(vcat, [rows[c][2][1] for c in 1:mesh.cells.n]), mesh.cells.n, mesh.edges.n),
            sparse(I, J, reduce(vcat, [rows[c][2][2] for c in 1:mesh.cells.n]), mesh.cells.n, mesh.edges.n))
end

"""
Point values at the three Gauss points of ẽ of the degree-p least-squares polynomial fitted to the cell averages of
cell `side` of e and its neighbours (the UPW2 stencils): three matrices (edges × cells).
`constrained = true`: mean-preserving fit (the polynomial reproduces the mean of cell `side` exactly).  The plain fit is
stable only when it nearly interpolates (quadratic on one ring); on wider stencils it smears the upwind cell's own
value and is violently unstable.  With the constraint, 2-ring quadratics and cubics are stable (cubic: 4th order).
"""
function lsq_edge_points(mesh, p; side, rings = 1, qdeg = 5, constrained = false)
    α = monomials(p); nα = length(α); b, w = tri_rule(qdeg)
    rows = Vector{Any}(undef, mesh.edges.n)
    Threads.@threads for e in 1:mesh.edges.n
        xs, _, _ = dual_gauss(mesh, e); m = dual_edge(mesh, e)[3]; s = norm(-(edge_endpoints(mesh, e, true)...))
        φ(x) = [((x[1] - m[1]) / s)^a * ((x[2] - m[2]) / s)^c for (a, c) in α]
        c0 = Int(mesh.edges.cells[e][side]); S = cell_rings(mesh, [c0], rings)
        Φ = zeros(length(S), nα)
        for (i, c) in enumerate(S)
            Ac = 0.0
            for (P1, P2, P3) in cell_fan(mesh, c, m)
                At = abs(_cross2(P2 - P1, P3 - P1)) / 2; Ac += At
                for (bb, ww) in zip(b, w); Φ[i, :] .+= (ww * At) .* φ(bb[1] .* P1 .+ bb[2] .* P2 .+ bb[3] .* P3); end
            end
            Φ[i, :] ./= Ac
        end
        if constrained          # reproduce the mean of the side (upwind) cell exactly; least squares on the others
            i0 = findfirst(==(c0), S); oth = [i for i in eachindex(S) if i != i0]
            Mp = pinv(Φ[oth, 2:end] .- Φ[i0:i0, 2:end])
            rows[e] = (S, map(xs) do x
                r = Mp' * (φ(x)[2:end] .- Φ[i0, 2:end]); wv = zeros(length(S)); wv[oth] .= r; wv[i0] = 1 - sum(r); wv
            end)
        else
            F = qr(Φ); Qm = Matrix(F.Q)[:, 1:nα]; Rm = UpperTriangular(F.R)
            rows[e] = (S, [Qm * (Rm' \ φ(x)) for x in xs])
        end
    end
    I = reduce(vcat, [fill(e, length(rows[e][1])) for e in 1:mesh.edges.n]); J = reduce(vcat, [rows[e][1] for e in 1:mesh.edges.n])
    return ntuple(g -> sparse(I, J, reduce(vcat, [rows[e][2][g] for e in 1:mesh.edges.n]), mesh.edges.n, mesh.cells.n), 3)
end

"Everything the extrusion contraction needs on one mesh (added to a lie_setup dictionary)."
function extrusion_setup!(G; p_sharp = 2, p_rho = 2, rings = 1, constrained = false)
    mesh = G["mesh"]
    G["Qg"] = sharp_on_dual_edges(mesh, p_sharp)
    G["R1g"] = lsq_edge_points(mesh, p_rho; side = 1, rings, constrained); G["R2g"] = lsq_edge_points(mesh, p_rho; side = 2, rings, constrained)
    G["ld"] = [dual_gauss(mesh, e)[3] for e in 1:mesh.edges.n]
    G["Ux"], G["Uy"] = sharp_at_generators(mesh, p_sharp)
    G["H_rec"] = Diagonal(G["ld"] ./ 2) * (_GL3[1][2] * G["Qg"][1] + _GL3[2][2] * G["Qg"][2] + _GL3[3][2] * G["Qg"][3])   # the star of u♯
    return G
end

"""
I_2(u, ρ̃) on every Voronoi edge:  F_e = (|ẽ|/2) Σ_g w_g ŵ_g ρ̂_up(g)(x_g)   (upwind extrusion of ẽ, pointwise upwinding).
ŵ_g = u♯·t_e at the Gauss points, with its mean replaced by U_e/|ẽ| (U = H v): H fixes the swept area, u♯ its shape.
`shape = false` uses ŵ_g = U_e/|ẽ| (a straight extrusion), which is UPW2.
"""
function extrusion_flux(G, U, Qv, ρ̄; shape = true)
    ld = G["ld"]; w = (_GL3[1][2], _GL3[2][2], _GL3[3][2])
    m = 0.5 .* (w[1] .* Qv[1] .+ w[2] .* Qv[2] .+ w[3] .* Qv[3])
    F = zeros(length(U))
    for g in 1:3
        ŵ = shape ? Qv[g] .- m .+ U ./ ld : U ./ ld
        ρa = G["R1g"][g] * ρ̄; ρb = G["R2g"][g] * ρ̄
        F .+= (ld ./ 2) .* w[g] .* (max.(ŵ, 0) .* ρa .+ min.(ŵ, 0) .* ρb)
    end
    return F
end

"dρ̃/dt = -D̃ I_2(u, ρ̃) for a flow test (fields, coefs, T) and a star ('H_DEC','H_P1','H_P2','H_rec','exact')."
function extrusion_rhs(G, star, test; shape = true)
    mesh = G["mesh"]; Div = G["Div"]; A = G["A"]
    vs = [reduce_primal_edges(mesh, f).values for f in test.fields]
    Us = [star == "exact" ? reduce_dual_edges(mesh, star_field(f)).values : G[star] * v for (f, v) in zip(test.fields, vs)]
    Qvs = if star == "exact"            # exact u·t_e at the Gauss points
        [ntuple(g -> [dot(f(dual_gauss(mesh, e)[1][g]), _unit_t(mesh, e)) for e in 1:mesh.edges.n], 3) for f in test.fields]
    else
        [ntuple(g -> G["Qg"][g] * v, 3) for v in vs]
    end
    return function (ρ, t)
        c = test.coefs(t, test.T)
        U = sum(ci .* Ui for (ci, Ui) in zip(c, Us)); Qv = ntuple(g -> sum(ci .* q[g] for (ci, q) in zip(c, Qvs)), 3)
        return -(Div * extrusion_flux(G, U, Qv, ρ ./ A; shape))
    end
end

# ---------------------------------------------------------------------------- 0-forms:  L_u q = i_u dq  (no Hodge star)
"""
Slots (generator c, incident triangle T): rows (over edges) of the reconstruction of a primal 1-form β ∈ [P_p]² from its
edge values on the stencil of T (tri_layers of T's edges, L layers), evaluated at x_c; plus the two other vertices of T
relative to x_c, for the upwind test.  Returns (Bx, By, slots, rel) with slots[c] the slot range of c.
"""
function oneform_slots(mesh, p; L = 2)
    nsl = [length(mesh.cells.vertices[c]) for c in 1:mesh.cells.n]
    first_ = cumsum([1; nsl[1:end-1]]); slots = [first_[c]:(first_[c] + nsl[c] - 1) for c in 1:mesh.cells.n]
    rows = Vector{Any}(undef, sum(nsl)); rel = Vector{Any}(undef, sum(nsl))
    h = meshsize(mesh)
    Threads.@threads for c in 1:mesh.cells.n
        xc = _vec2(mesh.cells.position[c])
        for (k, T) in enumerate(mesh.cells.vertices[c])
            s = slots[c][k]
            S = sort(collect(union((Set(tri_layers(mesh, Int(e), L)) for e in mesh.vertices.edges[T])...)))
            x0 = _vec2(closest(mesh.cells.position[c], mesh.vertices.position[T], 1.0, 1.0))
            r = circulation_lsq_rows(mesh, x0, h, S, p, [(xc, [1.0, 0.0]), (xc, [0.0, 1.0])])
            rows[s] = (S, r)
            others = [Int(cc) for cc in mesh.vertices.cells[T] if cc != c]
            rel[s] = [_vec2(closest(mesh.cells.position[c], mesh.cells.position[cc], 1.0, 1.0)) .- xc for cc in others]
        end
    end
    n = sum(nsl)
    I = reduce(vcat, [fill(s, length(rows[s][1])) for s in 1:n]); J = reduce(vcat, [rows[s][1] for s in 1:n])
    Bx = sparse(I, J, reduce(vcat, [rows[s][2][1] for s in 1:n]), n, mesh.edges.n)
    By = sparse(I, J, reduce(vcat, [rows[s][2][2] for s in 1:n]), n, mesh.edges.n)
    return Bx, By, slots, rel
end

"The slot of the triangle incident to c that contains the backward direction -u (upwind extrusion of the point x_c)."
function upwind_slot(slots, rel, c, ux, uy)
    d = (-ux, -uy); best = first(slots[c]); bestscore = -Inf
    for s in slots[c]
        p1, p2 = rel[s]
        a1 = _cross2(p1, d) * _cross2(p1, p2); a2 = _cross2(p2, d) * _cross2(p2, p1)
        score = min(a1, a2)                                # ≥ 0 iff d lies in the sector of T
        score > bestscore && (bestscore = score; best = s)
    end
    return best
end

"dq/dt = -i_u dq at the generators (advective form of the 0-form Lie derivative), for a flow test; velocity: 'rec' or 'exact'."
function extrusion_scalar_rhs(G, test; velocity = "rec")
    mesh = G["mesh"]; D0 = sparse(Float64.(d0_matrix_primal(mesh)))
    Bx, By, slots, rel = G["slots"]
    vs = [reduce_primal_edges(mesh, f).values for f in test.fields]
    Us = velocity == "exact" ? [([f(_vec2(x))[1] for x in mesh.cells.position], [f(_vec2(x))[2] for x in mesh.cells.position]) for f in test.fields] :
                               [(G["Ux"] * v, G["Uy"] * v) for v in vs]
    N = mesh.cells.n
    return function (q, t)
        c = test.coefs(t, test.T)
        ux = sum(ci .* U[1] for (ci, U) in zip(c, Us)); uy = sum(ci .* U[2] for (ci, U) in zip(c, Us))
        β = D0 * q; bx = Bx * β; by = By * β
        out = zeros(N)
        @inbounds for k in 1:N
            s = upwind_slot(slots, rel, k, ux[k], uy[k])
            out[k] = -(ux[k] * bx[s] + uy[k] * by[s])
        end
        return out
    end
end
