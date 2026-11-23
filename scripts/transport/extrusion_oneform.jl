# ============================================================================
#  extrusion_oneform.jl -- the Lie derivative of a primal 1-form by Cartan's formula with extrusion-based contractions.
#  Library (no output); builds on extrusion_lie.jl.
#
#  The 1-form β (e.g. the velocity v♭) is the primal 1-cochain b_e = ∫_e β·t_e on the Delaunay edges.
#        L_u β = i_u dβ + d i_u β          →          L_1 b = I_2(u, D1 b) + D0 I_1(u, b)
#  * D1 b (triangles): the circulation around each Delaunay triangle = the integral of the vorticity ζ = curl β.
#  * I_2(u, ·): primal 2-form -> primal 1-form.  Extrude the Delaunay edge e backwards along the flow; the swept area per
#    unit time is u·n_e (n_e = t_e turned clockwise) and the vorticity is taken from the upwind triangle:
#        I_2(u, ζ)_e = ∫_e ζ_up u·n_e  ≈ (|e|/2) Σ_g w_g ŵ_g ζ̂_up(g)(x_g),   ŵ_g = (u♯·n_e)(x_g),
#    ζ̂_T = quadratic least-squares fit to the triangle averages of ζ around the upwind triangle T.
#    (continuous check: (i_u ζ dx∧dy)(t) = ζ (u_x t_y − u_y t_x) = ζ u·n.)
#  * I_1(u, ·): primal 1-form -> 0-form at the generators, the backward extrusion of the point x_c (as EXT0):
#        I_1(u, b)_c = u♯(x_c) · b♯_T(x_c),  T = the triangle at x_c containing −u♯(x_c).
#  No Hodge star anywhere: both contractions need the velocity at points (u♯, from the circulations v), and v does
#  not carry u·n_e on Delaunay edges -- a star maps v to Voronoi edges, which is not where it is needed.
#
#  Structure (any I_1, I_2):  D1 L_1 = L_2 D1  with L_2 = D1 I_2 (vorticity is transported in flux form on the
#  triangles), and L_1 D0 = D0 L_0 with L_0 = I_1(u, D0 ·) (EXT0).  So the spectrum of L_1 is that of L_0 (on exact
#  forms) together with that of L_2 restricted to exact 2-cochains (plus the 2 harmonic directions of the torus).
# ============================================================================
isdefined(Main, :extrusion_setup!) || include(joinpath(@__DIR__, "extrusion_lie.jl"))

const _GL5 = ((-0.9061798459386640, 0.2369268850561891), (-0.5384693101056831, 0.4786286704993665), (0.0, 0.5688888888888889),
              (0.5384693101056831, 0.4786286704993665), (0.9061798459386640, 0.2369268850561891))

"The three generators of triangle T, unwrapped about the point ref."
tri_vertices(mesh, T, ref) = [_vec2(closest(_V(ref), mesh.cells.position[c], mesh.x_period, mesh.y_period)) for c in mesh.vertices.cells[T]]
"Triangles sharing a generator with T (T's vertex ring)."
tri_vertex_ring(mesh, T) = sort(unique(Int(t) for c in mesh.vertices.cells[T] for t in mesh.cells.vertices[c]))
"Triangles sharing an edge with T."
tri_edge_neighbors(mesh, T) = [Int(t) for e in mesh.vertices.edges[T] for t in mesh.edges.vertices[e] if t != T]
"T, its edge neighbours and theirs (10 triangles)."
tri_edge_ring2(mesh, T) = sort(unique(vcat(T, tri_edge_neighbors(mesh, T), (tri_edge_neighbors(mesh, s) for s in tri_edge_neighbors(mesh, T))...)))

"Delaunay edge e: endpoints (unwrapped about its position), midpoint, unit tangent, normal n = t turned clockwise, length."
function primal_edge(mesh, e)
    a, b = edge_endpoints(mesh, e, true); a = _vec2(a); b = _vec2(b); l = norm(b .- a); t = (b .- a) ./ l
    return a, b, (a .+ b) ./ 2, t, [t[2], -t[1]], l
end

"""
Triangle geometry: areas A_T and the sign σ_T that turns the rows of D1 = d1_matrix_primal into the counter-clockwise
circulation (ζ̄_T = σ_T (D1 b)_T / A_T is the mean vorticity of T).
"""
function triangle_geometry(mesh, D1)
    nT = mesh.vertices.n; A = zeros(nT); σ = zeros(nT)
    for T in 1:nT
        ref = _vec2(mesh.vertices.position[T]); P = tri_vertices(mesh, T, ref); C = (P[1] .+ P[2] .+ P[3]) ./ 3
        A[T] = abs(_cross2(P[2] .- P[1], P[3] .- P[1])) / 2
        e = Int(mesh.vertices.edges[T][1]); a, b = edge_endpoints(mesh, e, true; about = _V(C)); a = _vec2(a); b = _vec2(b)
        σ[T] = sign(D1[T, e] * _cross2(b .- a, C .- a))          # interior on the left of the oriented edge ⇔ CCW
    end
    return A, σ
end

"""
(Zm_g, Zp_g), g = 1:3 (edges × triangles): the degree-p least-squares polynomial fitted to the triangle averages on the
stencil of the triangle of e on the −n_e side (Zm) or the +n_e side (Zp), evaluated at the Gauss points of e.
`constrained = true`: the fit reproduces the mean of that (upwind) triangle exactly, and least squares acts on the
others only.  The plain least-squares fit (constrained = false) smears the upwind triangle's own value over the stencil
and gives an anti-diffusive, violently unstable scheme (Re λ ∝ 1/h) for every stencil and degree ≥ 1.
"""
function tri_lsq_edge_points(mesh, p; stencil = :vertex, qdeg = 5, constrained = true)
    α = monomials(p); nα = length(α); bq, wq = tri_rule(qdeg); E = mesh.edges.n
    rows = Vector{Any}(undef, E)
    Threads.@threads for e in 1:E
        a, b, m, t, n, l = primal_edge(mesh, e)
        xs = [m .+ (ξ / 2) .* (b .- a) for (ξ, _) in _GL3]
        φ(x) = [((x[1] - m[1]) / l)^i * ((x[2] - m[2]) / l)^j for (i, j) in α]
        out = Any[]
        Ts = [Int(T) for T in mesh.edges.vertices[e]]
        side = [dot(sum(tri_vertices(mesh, T, m)) ./ 3 .- m, n) for T in Ts]      # centroid side of each triangle
        order = side[1] < side[2] ? (1, 2) : (2, 1)                                  # (−n side, +n side)
        for k in order
            T0 = Ts[k]
            S = stencil == :vertex ? tri_vertex_ring(mesh, T0) : stencil == :edge2 ? tri_edge_ring2(mesh, T0) :
                stencil == :edge1 ? sort(vcat(T0, tri_edge_neighbors(mesh, T0))) : stencil == :self ? [T0] : error("stencil")
            Φ = zeros(length(S), nα)
            for (i, T) in enumerate(S)
                P = tri_vertices(mesh, T, m)
                for (bb, ww) in zip(bq, wq); Φ[i, :] .+= ww .* φ(bb[1] .* P[1] .+ bb[2] .* P[2] .+ bb[3] .* P[3]); end
            end
            if !constrained || p == 0
                F = qr(Φ); Qm = Matrix(F.Q)[:, 1:nα]; Rm = UpperTriangular(F.R)
                push!(out, (S, [Qm * (Rm' \ φ(x)) for x in xs]))
            else
                # mean-preserving fit: p̂ = ζ̄_T0 + Σ_{α≠0} c_α (φ_α − φ̄_α(T0)), c by least squares on the other triangles
                i0 = findfirst(==(T0), S); oth = [i for i in eachindex(S) if i != i0]
                M = Φ[oth, 2:end] .- Φ[i0:i0, 2:end]; Mp = pinv(M)
                ws = map(xs) do x
                    r = Mp' * (φ(x)[2:end] .- Φ[i0, 2:end]); wv = zeros(length(S)); wv[oth] .= r; wv[i0] = 1 - sum(r); wv
                end
                push!(out, (S, ws))
            end
        end
        rows[e] = out
    end
    mk(k, g) = sparse(reduce(vcat, [fill(e, length(rows[e][k][1])) for e in 1:E]), reduce(vcat, [rows[e][k][1] for e in 1:E]),
                      reduce(vcat, [rows[e][k][2][g] for e in 1:E]), E, mesh.vertices.n)
    return ntuple(g -> mk(1, g), 3), ntuple(g -> mk(2, g), 3)
end

"QN_g, g = 1:3 (edges × edges): (u♯·n_e)(x_g) at the Gauss points of the Delaunay edge e (P_p fit, stencil tri_layers(e, L))."
function sharp_normal_on_primal_edges(mesh, p; L = 3)
    E = mesh.edges.n; rows = Vector{Any}(undef, E)
    Threads.@threads for e in 1:E
        a, b, m, t, n, l = primal_edge(mesh, e); S = tri_layers(mesh, e, L)
        rows[e] = (S, circulation_lsq_rows(mesh, m, l, S, p, [(m .+ (ξ / 2) .* (b .- a), n) for (ξ, _) in _GL3]))
    end
    return ntuple(g -> sparse(reduce(vcat, [fill(e, length(rows[e][1])) for e in 1:E]), reduce(vcat, [rows[e][1] for e in 1:E]),
                              reduce(vcat, [rows[e][2][g] for e in 1:E]), E, E), 3)
end

"Type-stable copies of the slot data of oneform_slots (the Vector{Any} version makes the upwind search slow)."
function typed_slots(G)
    Bx, By, slots, rel = G["slots"]
    return Bx, By, UnitRange{Int}[first(s):last(s) for s in slots],
           NTuple{2,NTuple{2,Float64}}[((r[1][1], r[1][2]), (r[2][1], r[2][2])) for r in rel]
end
@inline function upwind_slot_t(sl::Vector{UnitRange{Int}}, rl::Vector{NTuple{2,NTuple{2,Float64}}}, c, ux, uy)
    d = (-ux, -uy); best = first(sl[c]); bestscore = -Inf
    @inbounds for s in sl[c]
        p1, p2 = rl[s]
        score = min(_cross2(p1, d) * _cross2(p1, p2), _cross2(p2, d) * _cross2(p2, p1))
        score > bestscore && (bestscore = score; best = s)
    end
    return best
end

"Everything L_1 needs on one mesh (added to a lie_setup + extrusion_setup! dictionary)."
function oneform_setup!(G; p_sharp = 2, p_zeta = 2, stencil = :edge2)
    mesh = G["mesh"]
    haskey(G, "slots") || (G["slots"] = oneform_slots(mesh, 2))
    G["D1p"] = sparse(Float64.(d1_matrix_primal(mesh))); G["D0p"] = sparse(Float64.(d0_matrix_primal(mesh)))
    G["AT"], G["σT"] = triangle_geometry(mesh, G["D1p"])
    G["QN"] = sharp_normal_on_primal_edges(mesh, p_sharp)
    G["Zm"], G["Zp"] = tri_lsq_edge_points(mesh, p_zeta; stencil = stencil)
    G["le"] = [primal_edge(mesh, e)[6] for e in 1:mesh.edges.n]
    G["tslots"] = typed_slots(G)
    return G
end

"I_2(u, ζ̄) on the Delaunay edges (ζ̄: mean CCW vorticity of each triangle; ŵ: the three vectors (u·n_e)(x_g))."
function vorticity_flux(G, ŵ, ζ̄)
    le = G["le"]; F = zeros(length(le))
    for g in 1:3
        zm = G["Zm"][g] * ζ̄; zp = G["Zp"][g] * ζ̄; wg = _GL3[g][2]
        @. F += (le / 2) * wg * (max(ŵ[g], 0) * zm + min(ŵ[g], 0) * zp)        # ŵ > 0: flow towards +n, upwind = −n side
    end
    return F
end
"I_1(u, b) at the generators (u♯ components ux, uy at the generators)."
function point_contraction(G, ux, uy, b)
    Bx, By, sl, rl = G["tslots"]; bx = Bx * b; by = By * b; N = length(ux); s = zeros(N)
    @inbounds for c in 1:N
        k = upwind_slot_t(sl, rl, c, ux[c], uy[c]); s[c] = ux[c] * bx[k] + uy[c] * by[k]
    end
    return s
end
"L_1 b = I_2(u, D1 b) + D0 I_1(u, b)."
oneform_lie(G, ŵ, ux, uy, b) = vorticity_flux(G, ŵ, G["σT"] .* (G["D1p"] * b) ./ G["AT"]) .+ G["D0p"] * point_contraction(G, ux, uy, b)

"Velocity data of a field f: ŵ (u·n_e at the Gauss points of e) and u at the generators, reconstructed from v or exact."
function oneform_velocity(G, f; velocity = "rec")
    mesh = G["mesh"]
    if velocity == "rec"
        v = reduce_primal_edges(mesh, f).values
        return ntuple(g -> G["QN"][g] * v, 3), G["Ux"] * v, G["Uy"] * v
    else
        P = [primal_edge(mesh, e) for e in 1:mesh.edges.n]
        ŵ = ntuple(g -> [dot(f(p[3] .+ (_GL3[g][1] / 2) .* (p[2] .- p[1])), p[5]) for p in P], 3)
        X = [_vec2(x) for x in mesh.cells.position]
        return ŵ, [f(x)[1] for x in X], [f(x)[2] for x in X]
    end
end

"db/dt = -L_1 b for a flow test (fields, coefs, T)."
function oneform_rhs(G, test; velocity = "rec")
    vel = [oneform_velocity(G, f; velocity) for f in test.fields]
    return function (b, t)
        c = test.coefs(t, test.T)
        ŵ = ntuple(g -> sum(ci .* V[1][g] for (ci, V) in zip(c, vel)), 3)
        ux = sum(ci .* V[2] for (ci, V) in zip(c, vel)); uy = sum(ci .* V[3] for (ci, V) in zip(c, vel))
        return -oneform_lie(G, ŵ, ux, uy, b)
    end
end

# ---------------------------------------------------------------------------- matrices (spectra and structure checks)
"Sparse K (edges × triangles) with I_2(u, ζ̄) = K ζ̄ for a steady ŵ."
function vorticity_matrix(G, ŵ)
    le = G["le"]; K = spzeros(length(le), size(G["Zm"][1], 2))
    for g in 1:3
        K = K .+ Diagonal((le ./ 2) .* _GL3[g][2] .* max.(ŵ[g], 0)) * G["Zm"][g] .+ Diagonal((le ./ 2) .* _GL3[g][2] .* min.(ŵ[g], 0)) * G["Zp"][g]
    end
    return K
end
"Sparse S (cells × edges) with I_1(u, b) = S b for steady u at the generators."
function point_matrix(G, ux, uy)
    Bx, By, sl, rl = G["tslots"]; sel = [upwind_slot_t(sl, rl, c, ux[c], uy[c]) for c in 1:length(ux)]
    return Diagonal(ux) * Bx[sel, :] .+ Diagonal(uy) * By[sel, :]
end
"(L1, L2, L0): the 1-form operator, the vorticity operator on triangles (L2 = D1 K diag(σ/A)), and the 0-form operator."
function oneform_matrices(G, ŵ, ux, uy)
    K = vorticity_matrix(G, ŵ) * Diagonal(G["σT"] ./ G["AT"]); S = point_matrix(G, ux, uy)
    return K * G["D1p"] .+ G["D0p"] * S, G["D1p"] * K, S * G["D0p"]
end

# ---------------------------------------------------------------------------- exact solutions
"X(0) for the characteristic through (x, t) (RK4 backwards, nsteps steps)."
function backtrace(test, x, t; nsteps = 400)
    X = copy(x); s = t; ds = t / nsteps
    g(X, s) = flow_u(test, X, s)
    for _ in 1:nsteps
        k1 = g(X, s); k2 = g(X .- ds / 2 .* k1, s - ds / 2); k3 = g(X .- ds / 2 .* k2, s - ds / 2); k4 = g(X .- ds .* k3, s - ds)
        X = X .- ds / 6 .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4); s -= ds
    end
    return X
end
"""
Exact edge integrals of the transported 1-form at time t: ∫_e β(t) = ∫_{X(e)} β0 = ∫_e β0(X(x))·(DX(x) t_e) dl
(X: backward characteristic map; DX t_e by central differences; 5-point Gauss on e).
"""
function exact_oneform(mesh, test, β0, t; δ = 1e-5)
    E = mesh.edges.n; out = zeros(E)
    Threads.@threads for e in 1:E
        a, b, m, te, n, l = primal_edge(mesh, e); s = 0.0
        for (ξ, w) in _GL5
            x = m .+ (ξ / 2) .* (b .- a)
            X = backtrace(test, x, t); DXt = (backtrace(test, x .+ δ .* te, t) .- backtrace(test, x .- δ .* te, t)) ./ (2δ)
            s += (w / 2) * l * dot(β0(mod.(X, 1.0)), DXt)
        end
        out[e] = s
    end
    return out
end
"Edge integrals with the 5-point Gauss rule (initial data and exact values of the same accuracy)."
function edge_integrals(mesh, β)
    E = mesh.edges.n; out = zeros(E)
    Threads.@threads for e in 1:E
        a, b, m, t, n, l = primal_edge(mesh, e)
        out[e] = sum((w / 2) * l * dot(β(m .+ (ξ / 2) .* (b .- a)), t) for (ξ, w) in _GL5)
    end
    return out
end
"""
Exact (L_u β)_e = ∫_e ζ_β u·n_e + (u·β)(x_b) − (u·β)(x_a)   (Cartan, i_u dβ + d i_u β), ζ_β = curl β given.
"""
function exact_lie_oneform(mesh, u, β, ζβ)
    E = mesh.edges.n; out = zeros(E)
    Threads.@threads for e in 1:E
        a, b, m, t, n, l = primal_edge(mesh, e)
        s = sum((w / 2) * l * ζβ(m .+ (ξ / 2) .* (b .- a)) * dot(u(m .+ (ξ / 2) .* (b .- a)), n) for (ξ, w) in _GL5)
        out[e] = s + dot(u(b), β(b)) - dot(u(a), β(a))
    end
    return out
end
"Diamond areas |e||ẽ|/2 (weights of the edge norms)."
diamond_areas(G) = G["le"] .* G["ld"] ./ 2
