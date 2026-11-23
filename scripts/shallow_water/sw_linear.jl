# ============================================================================
#  sw_linear.jl -- the linear rotating shallow-water tests of Peixoto, Alves, Fumagalli & Antonietti (2026, §5.2) with the
#  DEC / TRiSK C-grid on the SCVT meshes, for the three Hodge stars H_DEC, H_P1, H_P2.  Library + tests.
#
#      julia -t 4 scripts/shallow_water/sw_linear.jl [check | conv | energy]
#
#  Equations (f-plane, flat bottom, about rest):   u_t + f k×u + g∇η = 0,   η_t + H0 ∇·u = 0,   on [0,L]², L = 2πa.
#  Unknowns: f_e = ∫_e u·t_e (circulation along the Delaunay edge e, as in TRiSK), η_i = η(x_i) at the generators.
#      ḟ = −g D0 η + f0 W (H f)          (W: TRiSK operator, Voronoi-edge fluxes -> fluxes ∫_e u·n_e across Delaunay edges)
#      A η̇ = H0 D0ᵀ H f                   (A = cell areas;  D0ᵀ R̃⋆u = −∫_cell ∇·u)
#  Energy E = ½ H0 fᵀ Hs f + ½ g ηᵀ A η is conserved for symmetric H because W is antisymmetric (Thuburn & Cotter 2012).
#  For H = H_DEC this is exactly TRiSK/TRSK (the flux |ẽ| u_e is H_DEC f).
# ============================================================================
include(joinpath(@__DIR__, "..", "stars", "stars_cache.jl"))
using CairoMakie
empty!(lines)

# ---------------------------------------------------------------- physical parameters (Peixoto et al. 2026, §5.1)
const AEARTH = 6371.22e3; const LDOM = 2π * AEARTH; const GRAV = 9.80616; const H0 = 1.0e4; const F0 = 2 * 7.292e-5
const CW = sqrt(GRAV * H0)

# ---------------------------------------------------------------- TRiSK weights, flux form, MPAS orientation (ported from VoronoiOperators.jl)
"W_mpas[e, e'] such that ∫_e u·(k×n_e) ds ≈ Σ W[e,e'] (dv_{e'} u_{e'}), u_e = normal velocity from cell 1 to cell 2 of e"
function trisk_W(mesh)
    E = mesh.edges; C = mesh.cells; Vt = mesh.vertices
    I_, J_, V_ = Int[], Int[], Float64[]
    for e in 1:E.n
        c1, c2 = E.cells[e]
        for (c, sgn) in ((c1, 1.0), (c2, -1.0))
            ed = collect(C.edges[c]); vs = collect(C.vertices[c]); ne = C.nEdges[c]
            k = findfirst(isequal(e), ed)
            eds = circshift(ed, -k); vss = circshift(vs, -k)
            R = 0.0
            for i in 1:(ne-1)
                nx = eds[i]; v = vss[i]
                R += VoronoiMeshes.select_kite_area(Vt.kiteAreas, Vt.cells, v, c) / C.area[c]
                push!(I_, e); push!(J_, nx); push!(V_, sgn * VoronoiMeshes.sign_edge(E.cells[nx], c) * (0.5 - R))
            end
        end
    end
    return sparse(I_, J_, V_, E.n, E.n)
end

"sign s_e = +1 if the DEC orientation of e (d0) goes from cell 1 to cell 2 of the mesh edge, −1 otherwise"
function dec_orientation(mesh, D0)
    s = zeros(mesh.edges.n)
    for e in 1:mesh.edges.n
        c1, c2 = mesh.edges.cells[e]
        s[e] = D0[e, c2] > 0 ? 1.0 : -1.0
        @assert D0[e, c2] == -D0[e, c1]
    end
    return s
end

"everything the solver needs on one mesh (unit-torus geometry, scaled by LDOM)"
function sw_setup(mn)
    mesh = VoronoiMesh("meshes/$mn.nc"); S = cached_stars(mn; mesh = mesh)
    D0 = sparse(Float64.(d0_matrix_primal(mesh)))
    s = dec_orientation(mesh, D0)
    W = Diagonal(s) * trisk_W(mesh) * Diagonal(s)                      # to the DEC orientation
    A = mesh.cells.area .* LDOM^2
    X = [_pt(p) .* LDOM for p in mesh.cells.position]
    return (mesh = mesh, S = S, D0 = D0, W = W, A = A, X = X, n = mesh.edges.n, N = mesh.cells.n, h = LDOM / sqrt(mesh.cells.n))
end

"circulations ∫_e u·t_e (physical units) of a field given in physical coordinates"
circ(g, u) = reduce_primal_edges(g.mesh, x -> u(_pt(x) .* LDOM)).values .* LDOM
"exact Voronoi-edge fluxes ∫_ẽ u·t_e"
vflux(g, u) = reduce_dual_edges(g.mesh, star_field(x -> u(_pt(x) .* LDOM))).values .* LDOM
Jrot(u) = x -> (v = u(x); [-v[2], v[1]])

function check(mn = "mesh4")
    g = sw_setup(mn)
    logp("CHECK on $mn: $(g.N) cells, $(g.n) edges, h = $(round(g.h / 1e3, digits = 1)) km")
    logp(@sprintf("   W antisymmetric: ‖W + Wᵀ‖/‖W‖ = %.1e", norm(g.W + g.W', 1) / norm(g.W, 1)))
    k = 3 / AEARTH
    p(x) = sin(k * x[1]) * cos(2k * x[2]); gp(x) = [k * cos(k * x[1]) * cos(2k * x[2]), -2k * sin(k * x[1]) * sin(2k * x[2])]
    pv = [p(x) for x in g.X]
    logp(@sprintf("   D0 p = ∫_e ∇p·t (exact):  %.1e", maximum(abs, g.D0 * pv .- circ(g, gp)) / maximum(abs, circ(g, gp))))
    u(x) = [sin(k * x[1]) * cos(k * x[2]) + 0.3, cos(2k * x[1]) * sin(k * x[2]) - 0.2]
    F = vflux(g, u); Φ = -circ(g, Jrot(u))                              # ∫_e u·n_e = −∫_e (Ju)·t_e
    for (lab, Fl) in (("exact fluxes", F), ("H_DEC f", g.S["H_DEC"] * circ(g, u)))
        Φh = g.W * Fl
        logp(@sprintf("   W·(%s) vs ∫_e u·n_e:  max rel %.2e, rms rel %.2e, correlation %+.4f", lab,
                      maximum(abs, Φh .- Φ) / maximum(abs, Φ), norm(Φh .- Φ) / norm(Φ), dot(Φh, Φ) / (norm(Φh) * norm(Φ))))
    end
    logp(@sprintf("   divergence check: ‖A⁻¹ D0ᵀ F + ∇·u‖ rel = %.1e (cell means vs point values)",
                  begin dv(x) = k * cos(k * x[1]) * cos(k * x[2]) + k * cos(2k * x[1]) * cos(k * x[2])
                      d = -(g.D0' * F) ./ g.A; dex = [dv(x) for x in g.X]; maximum(abs, d .- dex) / maximum(abs, dex) end))
end

isempty(ARGS) || ARGS[1] != "check" || check(length(ARGS) > 1 ? ARGS[2] : "mesh4")
