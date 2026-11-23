# ============================================================================
#  lie_derivative.jl -- discrete Lie derivative of densities (twisted 2-forms) and of scalars (0-forms) on the
#  periodic Delaunay–Voronoi meshes, with the three Hodge stars of three_stars.jl.  Library (no output).
#
#  Continuous.  A density ρ̃ = ρ dA lives on the Voronoi cells (dual 2-cochain ρ̃_c = ∫_c ρ).  Cartan's formula and
#  dρ̃ = 0 give
#        L_u ρ̃ = d i_u ρ̃ ,        i_u ρ̃ = ρ ⋆u♭   (the mass flux, a dual 1-form: F̃_e = ∫_ẽ ρ u·t_e).
#  So  ∂t ρ̃ + L_u ρ̃ = 0  is  ∂t ρ̃_c + Σ_e (D̃ F̃)_c = 0  with D̃ = d1_matrix_dual (cells × edges) exact, and all the
#  approximation is in the discrete interior product F̃ ≈ i_u ρ̃.  The velocity is the primal 1-cochain
#  v_e = ∫_e u·t_e (circulation along the Delaunay edge, as in TRiSK / DEC); a Hodge star H turns it into fluxes.
#  For a scalar q (0-form), L_u q = i_u dq = u·∇q, discretised consistently with the mass flux (Eldred & Bauer 2022,
#  eq. 44:  L_u q = ⋆̃(1/h ∧ F̃ ∧ dq)) as
#        (L_u q)_c = [ D̃(q^e F̃) − q_c D̃ F̃ ]_c / ρ̃_c ,
#  which vanishes for constant q on every mesh (mass–tracer consistency).
#
#  Discrete interior products F̃ = K(u) ρ̄  (ρ̄_c = ρ̃_c / A_c, the cell value; K: cells -> edges):
#    "DEC"     wedge, then Hodge:  F̃ = H (ρ ∧ v),  (ρ ∧ v)_e = ½(ρ̄_a + ρ̄_b) v_e        (Hirani)
#    "TRiSK"   Hodge, then wedge:  F̃ = ρ^e ⊙ (H v),  ρ^e = ½(ρ̄_a + ρ̄_b)                  (Ringler et al. 2010)
#    "Ham"     F̃ = ½[ρ^e ⊙ H v + H(ρ ∧ v)]   the Hamiltonian mass flux of Eldred & Bauer (2022), eq. 99
#    "Peixoto" F̃ = ρ^e ⊙ (H v),  ρ^e = barycentric interpolation at the Voronoi-edge midpoint m̃_e   (Peixoto 2016, §3.1)
#    "CFVp"    F̃ = ρ^e ⊙ (H v),  ρ^e = mean over ẽ of the degree-p least-squares polynomial fitted to the cell
#              averages around e: a centred finite-volume reconstruction (Eldred, Norman & Taylor 2021)
#  For H = H_DEC (diagonal) "DEC" = "TRiSK" = "Ham".  H v is the volume flux ("transport velocity") Ũ; with the exact
#  volume flux R̃⋆u instead of H v, the variants show the error of the density reconstruction alone.
#  (Korn's ICON-O flux P^T(h P u), with Perot's reconstruction P, puts the density between a reconstruction and its
#   adjoint; it needs a factorisation H = P^T P and is not one of the stars here.)
# ============================================================================
isdefined(Main, :cached_stars) || include(joinpath(@__DIR__, "..", "stars", "stars_cache.jl"))
using SparseArrays, LinearAlgebra

# ---------------------------------------------------------------------------- test fields on the unit torus
"LeVeque's swirl, divergence free, stream function ψ = sin²(πx) sin²(πy)/π."
swirl(x) = [-sin(π * x[1])^2 * sin(2π * x[2]), sin(2π * x[1]) * sin(π * x[2])^2]
"A divergent field: swirl + ½∇φ, φ = cos2πx cos2πy / 2π."
swirl_div(x) = swirl(x) .+ 0.5 .* [-sin(2π * x[1]) * cos(2π * x[2]), -cos(2π * x[1]) * sin(2π * x[2])]
divu_div(x) = -0.5 * 2 * TP * cos(TP * x[1]) * cos(TP * x[2])          # div of swirl_div (the swirl part is 0)
"Smooth periodic bump (von Mises), width σ, centre x0."
bump(x; x0 = (0.5, 0.75), σ = 0.1) = exp((cos(TP * (x[1] - x0[1])) + cos(TP * (x[2] - x0[2])) - 2) / (TP^2 * σ^2))
∇bump(x; x0 = (0.5, 0.75), σ = 0.1) = bump(x; x0, σ) .* [-sin(TP * (x[1] - x0[1])), -sin(TP * (x[2] - x0[2]))] ./ (TP * σ^2)

# ---------------------------------------------------------------------------- geometry
_vec2(p) = [p[1], p[2]]
"Voronoi-edge endpoints and midpoint, unwrapped about the edge position."
function dual_edge(mesh, e)
    v1, v2 = edge_endpoints(mesh, e, false); return v1, v2, (v1 + v2) / 2
end
"Fan triangles of Voronoi cell c, unwrapped about the point ref."
function cell_fan(mesh, c, ref)
    g = closest(TensorsLite.Vec(x = ref[1], y = ref[2]), mesh.cells.position[c], mesh.x_period, mesh.y_period)
    vs = [_vec2(closest(g, mesh.vertices.position[v], mesh.x_period, mesh.y_period)) for v in mesh.cells.vertices[c]]
    gp = _vec2(g); n = length(vs)
    return [(gp, vs[i], vs[mod1(i + 1, n)]) for i in 1:n]
end

"½|d0|: the two cell values averaged onto every Delaunay edge (the combinatorial / TRiSK edge value)."
edge_average(mesh) = (D = d0_matrix_primal(mesh); 0.5 .* abs.(sparse(Float64.(D))))

"Barycentric (linear) interpolation of cell values at the Voronoi-edge midpoint m̃_e (Peixoto 2016, eq. 6-7)."
function barycentric_to_dual_midpoint(mesh)
    I, J, V = Int[], Int[], Float64[]
    for e in 1:mesh.edges.n
        _, _, m = dual_edge(mesh, e)
        best = (-Inf, Int[], Float64[])
        for T in mesh.edges.vertices[e]                                # the two triangles of e; m̃_e lies in one of them
            cs = collect(mesh.vertices.cells[T])
            P = [_vec2(closest(TensorsLite.Vec(x = m[1], y = m[2]), mesh.cells.position[c], mesh.x_period, mesh.y_period)) for c in cs]
            Atot = _cross2(P[2] - P[1], P[3] - P[1])
            λ = [_cross2(P[2] - m, P[3] - m), _cross2(P[3] - m, P[1] - m), _cross2(P[1] - m, P[2] - m)] ./ Atot
            minimum(λ) > best[1] && (best = (minimum(λ), cs, λ))
        end
        best[1] < -1e-8 && @warn "edge $e: Voronoi-edge midpoint outside both triangles (min λ = $(best[1]))"
        append!(I, fill(e, 3)); append!(J, best[2]); append!(V, best[3])
    end
    return sparse(I, J, V, mesh.edges.n, mesh.cells.n)
end

"Cells within r neighbour rings of the set S."
function cell_rings(mesh, S, r)
    S = Set{Int}(S)
    for _ in 1:r
        S = union(S, (Int.(collect(mesh.cells.cells[c])) for c in S)...)
    end
    return sort(collect(S))
end

const _GL3 = ((-sqrt(3 / 5), 5 / 9), (0.0, 8 / 9), (sqrt(3 / 5), 5 / 9))
monomials(p) = [(i, k - i) for k in 0:p for i in k:-1:0]

"""
Degree-p least-squares reconstruction from cell averages, evaluated as the mean over the Voronoi edge ẽ:
row e of R maps the cell averages of the stencil (cells within `rings` rings of the two cells of e) to
(1/|ẽ|)∫_ẽ p̂.  Exact for polynomials of degree ≤ p.  Monomials in (x - m̃_e)/|e|.
"""
function lsq_edge_reconstruction(mesh, p; rings = p <= 2 ? 1 : 2, qdeg = 5, side = 0)
    α = monomials(p); nα = length(α)
    b, w = tri_rule(qdeg)
    rows = Vector{Tuple{Vector{Int},Vector{Float64}}}(undef, mesh.edges.n)
    Threads.@threads for e in 1:mesh.edges.n
        v1, v2, m = dual_edge(mesh, e); s = edge_lengths_e(mesh, e)
        φ(x) = [((x[1] - m[1]) / s)^a * ((x[2] - m[2]) / s)^c for (a, c) in α]
        seed = side == 0 ? Int.(collect(mesh.edges.cells[e])) : [Int(mesh.edges.cells[e][side])]   # side 1|2: around one cell
        S = cell_rings(mesh, seed, rings)
        Φ = zeros(length(S), nα)
        for (i, c) in enumerate(S)
            Ac = 0.0
            for (P1, P2, P3) in cell_fan(mesh, c, m)
                At = abs(_cross2(P2 - P1, P3 - P1)) / 2; Ac += At
                for (bb, ww) in zip(b, w)
                    Φ[i, :] .+= (ww * At) .* φ(bb[1] .* P1 .+ bb[2] .* P2 .+ bb[3] .* P3)
                end
            end
            Φ[i, :] ./= Ac
        end
        ε = sum(g * φ(m .+ (ξ / 2) .* (v2 .- v1)) for (ξ, g) in _GL3) ./ 2
        F = qr(Φ); r = Matrix(F.Q)[:, 1:nα] * (UpperTriangular(F.R)' \ ε)
        rows[e] = (S, r)
    end
    I = reduce(vcat, [fill(e, length(rows[e][1])) for e in 1:mesh.edges.n])
    return sparse(I, reduce(vcat, first.(rows)), reduce(vcat, last.(rows)), mesh.edges.n, mesh.cells.n)
end
edge_lengths_e(mesh, e) = norm(-(edge_endpoints(mesh, e, true)...))

"Everything the Lie derivative needs on one mesh: areas, D̃, edge-value operators, and the three stars."
function lie_setup(mn; cfv = (2, 3), upw = (2,))
    mesh = VoronoiMesh("meshes/$mn.nc")
    S = cached_stars(mn; mesh)
    G = Dict{String,Any}("mesh" => mesh, "A" => collect(mesh.cells.area), "Div" => sparse(Float64.(d1_matrix_dual(mesh))),
                         "avg" => edge_average(mesh), "bary" => barycentric_to_dual_midpoint(mesh),
                         "H_DEC" => S["H_DEC"], "H_P1" => S["H_P1"], "H_P2" => S["H_P2"], "h" => meshsize(mesh))
    for p in cfv
        G["cfv$p"] = lsq_edge_reconstruction(mesh, p)
    end
    for p in upw                                   # upwind k-exact: the polynomial of each side cell (rings around it)
        G["up$p"] = (lsq_edge_reconstruction(mesh, p; side = 1), lsq_edge_reconstruction(mesh, p; side = 2))
    end
    return G
end

# ---------------------------------------------------------------------------- the discrete interior product
#  "UPWp" (upwind k-exact, in the spirit of the upwind-biased MPAS fluxes of Skamarock & Gassmann 2011 and the WENO
#  fluxes of Eldred, Norman & Taylor):  F̃_e = U_e⁺ ρ^e_(1) + U_e⁻ ρ^e_(2),  ρ^e_(j) = mean over ẽ of the degree-p
#  least-squares polynomial fitted on cell j of e and its neighbours (1 ring; U > 0 flows from cell 1 to cell 2).
#  UPW2 (quadratic, 1 ring) is a 3rd-order upwind flux and spectrally stable (max Re λ = 0 to round-off, every flow).
#  Wider upwind stencils are NOT: on mesh5, degree 2 on 2 rings, degree 3 or 4 on 2-3 rings all give max Re λ ≈ +20..+100,
#  growing like 1/h (the stencil reaches past the downwind cell).  The centred CFV2/CFV3 are unstable too (CFV2: +1.9).

#  It is not linear in u (it depends on sign U), so it is applied through rhs_function, not through L.
const VARIANTS = ("DEC", "TRiSK", "Ham", "Peixoto", "CFV2", "CFV3")
const UPWINDS = ("UPW2",)
edge_value_op(G, var) = var in ("DEC", "TRiSK", "Ham") ? G["avg"] : var == "Peixoto" ? G["bary"] : G["cfv$(var[end])"]

"Upwind flux F̃ = U⁺ R1 ρ̄ + U⁻ R2 ρ̄ for the volume fluxes U and the upwind reconstruction (R1, R2)."
upwind_flux(U, Rs, ρ̄) = max.(U, 0) .* (Rs[1] * ρ̄) .+ min.(U, 0) .* (Rs[2] * ρ̄)
"The (linear, steady-flow) upwind Lie derivative matrix, for spectra."
lie_density_upwind(G, var, U) = (Rs = G["up$(var[end])"]; G["Div"] * (Diagonal(max.(U, 0)) * Rs[1] .+ Diagonal(min.(U, 0)) * Rs[2]) * Diagonal(1 ./ G["A"]))

"""
dρ̃/dt = f(ρ̃, t) for the scheme (var, star) and the flow test = (fields, coefs, T): linear variants through the
precomputed L_i (u = Σ c_i a_i), upwind variants through the volume fluxes U(t) = Σ c_i U_i.
"""
function rhs_function(G, var, star, test)
    mesh = G["mesh"]
    if var in UPWINDS
        Us = [star == "exact" ? reduce_dual_edges(mesh, star_field(f)).values : G[star] * reduce_primal_edges(mesh, f).values for f in test.fields]
        Rs = G["up$(var[end])"]; Div = G["Div"]; A = G["A"]
        return (ρ, t) -> -(Div * upwind_flux(sum(c .* U for (c, U) in zip(test.coefs(t, test.T), Us)), Rs, ρ ./ A))
    end
    Ls = flow_operators(G, var, star, test.fields)
    return (ρ, t) -> -sum(c .* (L * ρ) for (c, L) in zip(test.coefs(t, test.T), Ls))
end

"Classical RK4 for dρ̃/dt = f(ρ̃, t) on [0, T]."
function rk4(f, ρ0, T, dt; save_every = 0)
    n = round(Int, T / dt); dt = T / n; ρ = copy(ρ0); t = 0.0; frames = Vector{Vector{Float64}}()
    save_every > 0 && push!(frames, copy(ρ))
    for k in 1:n
        k1 = f(ρ, t); k2 = f(ρ .+ (dt / 2) .* k1, t + dt / 2); k3 = f(ρ .+ (dt / 2) .* k2, t + dt / 2); k4 = f(ρ .+ dt .* k3, t + dt)
        ρ .+= (dt / 6) .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4); t += dt
        save_every > 0 && k % save_every == 0 && push!(frames, copy(ρ))
    end
    return save_every > 0 ? (ρ, frames) : ρ
end

"""
K such that F̃ = K ρ̄ (cell values -> fluxes across the Voronoi edges), for the velocity circulations v and the star
`star` ("H_DEC", "H_P1", "H_P2", or "exact" = use the exact volume flux Uex = R̃⋆u instead of H v).
"""
function flux_operator(G, var, star, v; Uex = nothing)
    U = star == "exact" ? Uex : G[star] * v
    R = edge_value_op(G, var)
    var == "DEC"  && return star == "exact" ? error("DEC needs a star") : G[star] * (Diagonal(v) * R)
    var == "Ham"  && return star == "exact" ? error("Ham needs a star") : 0.5 .* (G[star] * (Diagonal(v) * R) .+ Diagonal(U) * R)
    return Diagonal(U) * R
end

"The discrete Lie derivative of densities, L (cells × cells) acting on ρ̃ (cell integrals):  L_u ρ̃ ≈ L ρ̃."
lie_density(G, K) = G["Div"] * K * Diagonal(1 ./ G["A"])

"The consistent discrete Lie derivative of a scalar q given a density ρ̃:  (L_u q)_c = [D̃(q^e F̃) − q_c D̃F̃]_c / ρ̃_c."
function lie_scalar(G, var, star, v, ρ̃, q; Uex = nothing)
    K = flux_operator(G, var, star, v; Uex)
    F = K * (ρ̃ ./ G["A"])                          # mass flux
    qe = edge_value_op(G, var) * q                   # the same edge reconstruction for q
    return (G["Div"] * (qe .* F) .- q .* (G["Div"] * F)) ./ ρ̃
end

# ---------------------------------------------------------------------------- the deformational test flow
#  u(x,t) = (1/T, 0) + cos(πt/T) swirl(x - t/T, y)   (Nair & Lauritzen 2010, case 4, on the unit torus).
#  The swirl reverses at T/2 and the translation completes one period, so the exact solution at T is the initial
#  one.  Unlike u = cos(πt/T) swirl(x), it is not separable in time: with a separable flow, the semi-discrete
#  solution of ∂t ρ̃ = -c(t) L ρ̃ at T is exp(-L ∫c) ρ̃0 = ρ̃0 for every L, and only the time-stepping error is left.
#  Expanding swirl(x - s, y) in s gives u = Σ_i c_i(t) a_i(x) with four fixed divergence-free fields:
const DEFORM_FIELDS = (x -> [1.0, 0.0],                                                           # translation
                       x -> [-0.5 * sin(TP * x[2]), 0.0],                                         # a0
                       x -> [0.5 * cos(TP * x[1]) * sin(TP * x[2]), 0.5 * sin(TP * x[1]) * (1 - cos(TP * x[2]))],    # a1
                       x -> [0.5 * sin(TP * x[1]) * sin(TP * x[2]), -0.5 * cos(TP * x[1]) * (1 - cos(TP * x[2]))])   # a2
deform_coefs(t, T) = (1 / T, cos(π * t / T), cos(π * t / T) * cos(TP * t / T), cos(π * t / T) * sin(TP * t / T))
deform_u(x, t, T) = sum(c .* f(x) for (c, f) in zip(deform_coefs(t, T), DEFORM_FIELDS))

"The operators L_i of a flow u = Σ c_i(t) a_i(x) (every flux variant is linear in the velocity)."
function flow_operators(G, var, star, fields)
    mesh = G["mesh"]
    return [lie_density(G, flux_operator(G, var, star, reduce_primal_edges(mesh, f).values;
                                         Uex = reduce_dual_edges(mesh, star_field(f)).values)) for f in fields]
end
deform_operators(G, var, star) = flow_operators(G, var, star, DEFORM_FIELDS)

# ---------------------------------------------------------------------------- the test suite (unit torus)
#  Planar analogues of the standard tracer tests (Williamson et al. 1992; Nair & Lauritzen 2010; Lauritzen et al. 2012).
#  Each flow is u(x,t) = Σ_i c_i(t) a_i(x).  "initial" means the exact solution at T is the initial state.
"Steady cellular flow, ψ = sin2πx sin2πy / 2π: closed streamlines around (¼,¼), angular speed 2π at the centre."
cellular(x) = [-sin(TP * x[1]) * cos(TP * x[2]), cos(TP * x[1]) * sin(TP * x[2])]
"∇φ(x - s, y) = cos2πs b1 + sin2πs b2 for φ = sin2πξ sin2πy / 4π (a purely divergent field, |∇φ| ≤ ½)."
div_b1(x) = 0.5 .* [cos(TP * x[1]) * sin(TP * x[2]), sin(TP * x[1]) * cos(TP * x[2])]
div_b2(x) = 0.5 .* [sin(TP * x[1]) * sin(TP * x[2]), -cos(TP * x[1]) * cos(TP * x[2])]
pdist(x, x0) = (d = abs.(x .- x0); d = min.(d, 1 .- d); sqrt(sum(d .^ 2)))
"Cosine bell of radius R (C¹, not C²), as in Lauritzen et al. (2012)."
cosbell(x; x0, R = 0.15) = (r = pdist(x, x0); r < R ? 0.5 * (1 + cos(π * r / R)) : 0.0)
hills(x) = bump(x; x0 = (0.35, 0.5)) + bump(x; x0 = (0.65, 0.5))
const DIV_KAPPA = 0.5
const TESTS = (
    translation   = (fields = (x -> [1.0, 1.0],), coefs = (t, T) -> (1.0,), T = 1.0, ρ0 = x -> bump(x; x0 = (0.5, 0.5)), exact = :initial,
                     what = "uniform translation u = (1,1), Gaussian hill, one period"),
    cellular      = (fields = (cellular,), coefs = (t, T) -> (1.0,), T = 1.0, ρ0 = x -> bump(x; x0 = (0.25, 0.32), σ = 0.035), exact = :characteristics,
                     what = "steady cellular (Taylor–Green) flow, sheared hill inside one cell, exact by characteristics"),
    deformational = (fields = DEFORM_FIELDS, coefs = deform_coefs, T = 1.0, ρ0 = hills, exact = :initial,
                     what = "translation + reversing swirl (Nair–Lauritzen case 4), two Gaussian hills"),
    divergent     = (fields = (x -> [1.0, 0.0], x -> DIV_KAPPA .* div_b1(x), x -> DIV_KAPPA .* div_b2(x)),
                     coefs = (t, T) -> (1 / T, cos(π * t / T) * cos(TP * t / T), cos(π * t / T) * sin(TP * t / T)), T = 1.0, ρ0 = hills,
                     exact = :initial, what = "translation + reversing divergent flow (compression), two Gaussian hills"),
    cosine_bells  = (fields = DEFORM_FIELDS, coefs = deform_coefs, T = 1.0,
                     ρ0 = x -> 0.1 + 0.9 * (cosbell(x; x0 = (0.35, 0.5)) + cosbell(x; x0 = (0.65, 0.5))), exact = :initial,
                     what = "deformational flow, two cosine bells (C¹) on a 0.1 background"))
flow_u(test, x, t) = sum(c .* f(x) for (c, f) in zip(test.coefs(t, test.T), test.fields))

"Cell averages of f over every Voronoi cell (fan triangles, degree-5 rule), threaded."
function cell_averages(mesh, f; qdeg = 5)
    b, w = tri_rule(qdeg); out = zeros(mesh.cells.n)
    Threads.@threads for c in 1:mesh.cells.n
        s = 0.0; Ac = 0.0
        for (P1, P2, P3) in cell_fan(mesh, c, _vec2(mesh.cells.position[c]))
            At = abs(_cross2(P2 - P1, P3 - P1)) / 2; Ac += At
            s += At * sum(ww * f(mod.(bb[1] .* P1 .+ bb[2] .* P2 .+ bb[3] .* P3, 1.0)) for (bb, ww) in zip(b, w))
        end
        out[c] = s / Ac
    end
    return out
end

"""
Exact density at time t for a DIVERGENCE-FREE flow: ρ(x,t) = ρ0(X(0)), where X solves dX/ds = u(X,s) backwards from
X(t) = x (RK4, nsteps steps).  (For div u ≠ 0 a factor exp(-∫ div u ds) would multiply; the divergent test does not
need it because its exact solution at T is the initial state.)
"""
function exact_density(test, x, t; nsteps = 400)
    X = copy(x); s = t; ds = t / nsteps
    g(X, s) = flow_u(test, X, s)
    for _ in 1:nsteps
        k1 = g(X, s); k2 = g(X .- ds / 2 .* k1, s - ds / 2); k3 = g(X .- ds / 2 .* k2, s - ds / 2); k4 = g(X .- ds .* k3, s - ds)
        X = X .- ds / 6 .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4); s -= ds
    end
    return test.ρ0(mod.(X, 1.0))
end

"Exact cell averages at T (initial state, or by characteristics)."
exact_at_T(mesh, test) = test.exact == :initial ? cell_averages(mesh, test.ρ0) : cell_averages(mesh, x -> exact_density(test, x, test.T))

# ---------------------------------------------------------------------------- time stepping
"RK4 for dρ̃/dt = -Σ_i c_i(t) L_i ρ̃ from 0 to T (c = deform_coefs, or any function returning the coefficients)."
function rk4_lie(Ls, ρ0, T, dt; coefs = deform_coefs, save_every = 0)
    n = round(Int, T / dt); dt = T / n; ρ = copy(ρ0); t = 0.0; frames = Vector{Vector{Float64}}()
    f(ρ, t) = -sum(c .* (L * ρ) for (c, L) in zip(coefs(t, T), Ls))
    save_every > 0 && push!(frames, copy(ρ))
    for k in 1:n
        k1 = f(ρ, t); k2 = f(ρ .+ (dt / 2) .* k1, t + dt / 2); k3 = f(ρ .+ (dt / 2) .* k2, t + dt / 2); k4 = f(ρ .+ dt .* k3, t + dt)
        ρ .+= (dt / 6) .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4); t += dt
        save_every > 0 && k % save_every == 0 && push!(frames, copy(ρ))
    end
    return save_every > 0 ? (ρ, frames) : ρ
end
