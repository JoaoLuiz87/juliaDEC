# ============================================================================
#  sw_jet.jl -- nonlinear rotating shallow water (TRiSK form in DEC variables) and the barotropically unstable jet
#               of Peixoto, Alves, Fumagalli & Antonietti (PolyDG paper, §5.1), the planar analogue of Galewsky et al. (2004).
#
#      julia -t 4 scripts/shallow_water/sw_jet.jl mesh5 mesh6      ->  output/ts_data/sw_jet_<mesh>_<star>.jls, sw_jet.txt
#
#  Unknowns: depth h_i at the generators (point values, M0 = diag(A) as in TRiSK) and circulations v_e = ∫_e u·t_e on the
#  Delaunay edges (orientation cells[e][1] -> cells[e][2], the orientation of d0_matrix_primal).
#      U  = H v                        volume flux across the Voronoi edges (the only place where the star enters, with K)
#      F  = ĥ ⊙ U,  ĥ_e = ½(h_a + h_b)   mass flux
#      A ḣ = D0ᵀ F                      (exact: inflow − outflow)
#      v̇  = −Q(F) − D0 (K + g h)       Q(F)_e = Σ_e' W_ee' F_e' ½(q_e + q_e')   (energy-conserving PV flux, TRiSK weights W)
#      K_i = (1/A_i) Σ_{e∈∂i} ¼ v_e (Hv)_e   (Ringler's kinetic energy when H = H_DEC)
#      q_v = (f + ζ_v)/ĥ_v,  ζ_v = (D1 v)_v / |T_v|,  ĥ_v = Σ_i kite(i,v) h_i / |T_v|,  q_e = ½(q_v1 + q_v2)
#  W depends only on the mesh (Thuburn et al. 2009; Thuburn & Cotter 2012), not on H, so H_DEC and H_P2 differ only in the star.
#  With H_DEC this is TRiSK (Ringler et al. 2010): mass and energy are conserved (energy up to the time stepping).
#
#  Test (Peixoto et al., §5.1): Ω = [0,L]², L = 2π·6371.22 km, f = 2·7.292e-5 s⁻¹, g = 9.80616, H = 10⁴ m, flat bottom;
#  u = (u0 sin⁸¹(2πy/L), 0), u0 = 50 m/s; h in geostrophic balance, integrated from h = H at y = 0; perturbation
#  0.01 H exp(−(r/r_p)²), r_p = L/√1000, centred at (0.85L, 0.75L) and (0.15L, 0.25L); 20 days, RK4.
# ============================================================================
isdefined(Main, :cached_stars) || include(joinpath(@__DIR__, "..", "stars", "stars_cache.jl"))
using SparseArrays, LinearAlgebra, Printf, Serialization, QuadGK

const AE = 6371.22e3
const LDOM = 2π * AE
const GJ = 9.80616
const H0J = 1.0e4
const FJ = 2 * 7.292e-5
const U0J = 50.0
const DAY = 86400.0
_p2(p) = [p[1], p[2]]

# ---------------------------------------------------------------------------- geometry and operators
"Unwrapped positions (unit torus) of the three generators of triangle t about its circumcentre, in counter-clockwise order, with their cells."
function tri_cells_ccw(mesh, t)
    xp, yp = mesh.x_period, mesh.y_period; cc = mesh.vertices.position[t]
    cs = collect(Int.(mesh.vertices.cells[t])); P = [_p2(closest(cc, mesh.cells.position[c], xp, yp)) for c in cs]
    o = (P[2][1] - P[1][1]) * (P[3][2] - P[1][2]) - (P[2][2] - P[1][2]) * (P[3][1] - P[1][1])
    o < 0 && (cs = cs[[1, 3, 2]]; P = P[[1, 3, 2]])
    return cs, P, _p2(cc)
end
shoelace(Q) = 0.5 * sum(Q[k][1] * Q[mod1(k + 1, length(Q))][2] - Q[mod1(k + 1, length(Q))][1] * Q[k][2] for k in eachindex(Q))

"Everything the model needs, on the mesh scaled to [0, L]² (H is dimensionless and carries over from the unit torus)."
function sw_ops(mesh, H; L = LDOM, fcor = FJ)
    nc, ne, nt = mesh.cells.n, mesh.edges.n, mesh.vertices.n
    D0 = sparse(Float64.(d0_matrix_primal(mesh)))                  # edges × cells, c1 -> c2
    tris, D1, _ = triangles_ccw(mesh); D1 = sparse(Float64.(D1))    # triangles × edges, counter-clockwise circulation
    A = Float64.(mesh.cells.area) .* L^2
    AT = [T.area for T in tris] .* L^2
    le = [norm(-(edge_endpoints(mesh, e, true)...)) for e in 1:ne] .* L
    # kites: kite(i, t) = area of (Voronoi cell i) ∩ (triangle t), signed
    KI, KJ, KV = Int[], Int[], Float64[]
    for t in 1:nt
        cs, P, C = tri_cells_ccw(mesh, t)
        for k in 1:3
            Pk, Pn, Pp = P[k], P[mod1(k + 1, 3)], P[mod1(k + 2, 3)]
            push!(KI, t); push!(KJ, cs[k]); push!(KV, shoelace([Pk, (Pk + Pn) / 2, C, (Pp + Pk) / 2]) * L^2)
        end
    end
    Kite = sparse(KI, KJ, KV, nt, nc)                                # triangles × cells
    # TRiSK weights: for edges e, e' of cell i, going counter-clockwise from e to e' past the vertices v,
    #   W[e, e'] += s_{e,i} s_{e',i} (½ − Σ_v R_{i,v}),   R_{i,v} = kite(i,v)/A_i,   s_{e,i} = +1 if t_e points out of i
    WI, WJ, WV = Int[], Int[], Float64[]
    xp, yp = mesh.x_period, mesh.y_period
    for i in 1:nc
        x = mesh.cells.position[i]
        vs = collect(Int.(mesh.cells.vertices[i]))
        ang = [(q = _p2(closest(x, mesh.vertices.position[v], xp, yp)) .- _p2(x); atan(q[2], q[1])) for v in vs]
        vs = vs[sortperm(ang)]; n = length(vs)
        es = Vector{Int}(undef, n)                                   # es[k] = edge between vs[k] and vs[k+1]
        for k in 1:n
            pair = Set((vs[k], vs[mod1(k + 1, n)]))
            es[k] = only(e for e in Int.(mesh.cells.edges[i]) if Set(Int.(mesh.edges.vertices[e])) == pair)
        end
        R = [Kite[v, i] / A[i] for v in vs]
        s = [Int(mesh.edges.cells[e][1]) == i ? 1.0 : -1.0 for e in es]
        for a in 1:n, b in 1:n
            a == b && continue
            S = 0.0; k = a
            while k != b; k = mod1(k + 1, n); S += R[k]; end            # vertices vs[a+1], ..., vs[b]
            push!(WI, es[a]); push!(WJ, es[b]); push!(WV, s[a] * s[b] * (0.5 - S))
        end
    end
    W = sparse(WI, WJ, WV, ne, ne)
    # orientation of W: W (H_DEC v) must approximate the circulation of J u (u constant); fix the sign by that test
    u = [1.0, 0.3]; tv = [(q = -(edge_endpoints(mesh, e, true)...); -q) for e in 1:ne]   # p2 − p1 (unit torus)
    vC = [dot(u, tv[e]) for e in 1:ne] .* L; Ju = [-u[2], u[1]]; vJ = [dot(Ju, tv[e]) for e in 1:ne] .* L
    Hd = spdiagm(0 => [norm(-(edge_endpoints(mesh, e, false)...)) / norm(-(edge_endpoints(mesh, e, true)...)) for e in 1:ne])
    WU = W * (Hd * vC); sgn = dot(WU, vJ) > 0 ? 1.0 : -1.0; W .*= sgn
    werr = norm(W * (Hd * vC) - vJ) / norm(vJ)
    Phe = abs.(D0) .* 0.5                                            # ĥ_e = ½(h_a + h_b)
    Pqe = sparse(vcat(1:ne, 1:ne), vcat([Int(mesh.edges.vertices[e][1]) for e in 1:ne], [Int(mesh.edges.vertices[e][2]) for e in 1:ne]),
                 fill(0.5, 2ne), ne, nt)                             # q_e = ½(q_v1 + q_v2)
    Kc = spdiagm(0 => 1 ./ A) * (0.25 .* abs.(D0)')                  # K_i = (1/A_i) Σ ¼ (v ⊙ Hv)_e
    # APVM (anticipated potential vorticity, Sadourny & Basdevant 1985; as in MPAS):  q_e ← q_e − ½Δt (u·∇q)_e, with
    #   u·t_e = v_e/|e|,  ∂q/∂t_e = (q_c2 − q_c1)/|e|  (q at the generators from the kites),
    #   u·n_e = −(W U)_e/|e|  (n_e = J t_e),  ∂q/∂n_e = s_e (q_v2 − q_v1)/|ẽ|,  s_e = sign((c_v2 − c_v1)·n_e)
    ld = [norm(-(edge_endpoints(mesh, e, false)...)) for e in 1:ne] .* L
    GtI, GtJ, GtV = Int[], Int[], Float64[]
    for e in 1:ne
        p1, p2 = edge_endpoints(mesh, e, true); c1, c2 = edge_endpoints(mesh, e, false); t = _p2(p2) .- _p2(p1); n = [-t[2], t[1]]
        s = sign(dot(_p2(c2) .- _p2(c1), n)); v1, v2 = Int.(mesh.edges.vertices[e])
        push!(GtI, e, e); push!(GtJ, v2, v1); push!(GtV, s / ld[e], -s / ld[e])
    end
    Gt = sparse(GtI, GtJ, GtV, ne, nt)
    Qc = spdiagm(0 => 1 ./ A) * sparse(Kite')                         # q at the generators: Σ_v kite(i,v) q_v / A_i
    Gn = spdiagm(0 => 1 ./ le) * D0
    return (mesh = mesh, H = sparse(H), D0 = D0, D0t = sparse(D0'), D1 = D1, W = W, Wt_asym = norm(W + W', 1) / norm(W, 1), werr = werr,
            Kite = Kite, A = A, AT = AT, le = le, Phe = Phe, Pqe = Pqe, Kc = Kc, nc = nc, ne = ne, nt = nt, L = L,
            Gt = Gt, Qc = Qc, Gn = Gn, apvm = Ref(0.0), dt = Ref(0.0), fc = Ref(Float64(fcor)),
            xc = [_p2(mesh.cells.position[i]) .* L for i in 1:nc], tris = tris)
end

# ---------------------------------------------------------------------------- the model
function sw_rhs!(dx, x, O, buf)
    nc = O.nc; h = view(x, 1:nc); v = view(x, nc+1:nc+O.ne); dh = view(dx, 1:nc); dv = view(dx, nc+1:nc+O.ne)
    U, F, he, qe, ζ, hv, q, K, Φ, t1, t2 = buf
    mul!(U, O.H, v); mul!(he, O.Phe, h); F .= he .* U
    mul!(t1, O.D0t, F); dh .= t1 ./ O.A                                         # A ḣ = D0ᵀ F
    mul!(ζ, O.D1, v); ζ ./= O.AT; mul!(hv, O.Kite, h); hv ./= O.AT
    q .= (O.fc[] .+ ζ) ./ hv; mul!(qe, O.Pqe, q)
    t1e = buf[12]; t2e = buf[13]
    if O.apvm[] > 0                                                               # APVM: q_e ← q_e − apvm Δt (u·∇q)_e
        mul!(t1e, O.W, U); mul!(t1, O.Qc, q); mul!(t2e, O.Gn, t1)                # W U;  q at cells;  ∂q/∂t_e
        qe .-= O.apvm[] * O.dt[] .* ((v ./ O.le) .* t2e .- (t1e ./ O.le) .* (O.Gt * q))
    end
    mul!(t1e, O.W, F); t2e .= qe .* F; mul!(dv, O.W, t2e)
    dv .= -0.5 .* (qe .* t1e .+ dv)                                               # −Q(F), energy-conserving form
    t1e .= v .* U; mul!(K, O.Kc, t1e); Φ .= K .+ GJ .* h
    mul!(t1e, O.D0, Φ); dv .-= t1e                                                # − D0 (K + g h)
    return dx
end
sw_buffers(O) = (zeros(O.ne), zeros(O.ne), zeros(O.ne), zeros(O.ne), zeros(O.nt), zeros(O.nt), zeros(O.nt), zeros(O.nc), zeros(O.nc),
                 zeros(O.nc), zeros(O.nc), zeros(O.ne), zeros(O.ne))
function rk4_step!(x, dt, O, buf, k1, k2, k3, k4, xt)
    sw_rhs!(k1, x, O, buf); xt .= x .+ (dt / 2) .* k1
    sw_rhs!(k2, xt, O, buf); xt .= x .+ (dt / 2) .* k2
    sw_rhs!(k3, xt, O, buf); xt .= x .+ dt .* k3
    sw_rhs!(k4, xt, O, buf)
    x .+= (dt / 6) .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4)
    return x
end
"Mass, total energy and potential enstrophy."
function invariants(x, O)
    h = x[1:O.nc]; v = x[O.nc+1:end]; U = O.H * v; he = O.Phe * h
    ζ = (O.D1 * v) ./ O.AT; hv = (O.Kite * h) ./ O.AT; q = (O.fc[] .+ ζ) ./ hv
    return (mass = sum(O.A .* h), energy = sum(O.A .* GJ .* h .^ 2 ./ 2) + 0.5 * sum(he .* v .* U), enstrophy = 0.5 * sum(O.AT .* hv .* q .^ 2))
end
vorticity(x, O) = (O.D1 * x[O.nc+1:end]) ./ O.AT

# ---------------------------------------------------------------------------- the unstable jet
ujet(y) = U0J * sin(2π * y / LDOM)^81
pdist(a, b, L) = (d = abs.(a .- b); d = min.(d, L .- d); sqrt(sum(abs2, d)))
function jet_initial(O)
    L = O.L; rp = L / sqrt(1000.0); mesh = O.mesh
    # balanced depth h(y) = H − (f/g) ∫_0^y u dy': composite Simpson on 2·10^5 intervals, then linear interpolation
    M = 200_000; ys = range(0.0, L; length = M + 1); Δ = step(ys); uy = ujet.(ys); cum = zeros(M + 1)
    for k in 2:2:M; cum[k+1] = cum[k-1] + Δ / 3 * (uy[k-1] + 4uy[k] + uy[k+1]); cum[k] = cum[k-1] + Δ / 2 * (uy[k-1] + uy[k]); end
    function hbal(y)
        s = mod(y, L) / Δ; k = clamp(floor(Int, s), 0, M - 1); θ = s - k
        H0J - (FJ / GJ) * ((1 - θ) * cum[k+1] + θ * cum[k+2])
    end
    h = [hbal(x[2]) for x in O.xc]
    for c in ([0.85L, 0.75L], [0.15L, 0.25L]); h .+= 0.01H0J .* exp.(-([pdist(x, c, L) for x in O.xc] ./ rp) .^ 2); end
    GL8 = (( -0.9602898564975363, 0.1012285362903763), (-0.7966664774136267, 0.2223810344533745), (-0.5255324099163290, 0.3137066458778873),
           (-0.1834346424956498, 0.3626837833783620), (0.1834346424956498, 0.3626837833783620), (0.5255324099163290, 0.3137066458778873),
           (0.7966664774136267, 0.2223810344533745), (0.9602898564975363, 0.1012285362903763))
    v = zeros(O.ne)
    for e in 1:O.ne
        p1, p2 = edge_endpoints(mesh, e, true); p1 = _p2(p1) .* L; p2 = _p2(p2) .* L
        v[e] = sum(w / 2 * ujet(p1[2] + (ξ + 1) / 2 * (p2[2] - p1[2])) for (ξ, w) in GL8) * (p2[1] - p1[1])
    end
    return vcat(h, v)
end

"Zonal geostrophic balance (Peixoto et al. §5.1): u = u0 sin(2πy/L), h = H + h0 cos(2πy/L), h0 = f u0 L/(2πg); an exact steady state."
function steady_initial(O)
    L = O.L; h0 = FJ * U0J * L / (2π * GJ)
    h = [H0J + h0 * cos(2π * x[2] / L) for x in O.xc]
    v = zeros(O.ne)
    for e in 1:O.ne
        p1, p2 = edge_endpoints(O.mesh, e, true); p1 = _p2(p1) .* L; p2 = _p2(p2) .* L; dy = p2[2] - p1[2]
        # ∫_e u0 sin(2πy/L) dx along the straight edge, exactly
        v[e] = abs(dy) < 1e-9L ? U0J * sin(2π * p1[2] / L) * (p2[1] - p1[1]) :
               U0J * (p2[1] - p1[1]) / dy * (L / 2π) * (cos(2π * p1[2] / L) - cos(2π * p2[2] / L))
    end
    return vcat(h, v)
end
"Build the star and the operators for mesh mn."
function star_matrix(mn, star, mesh)
    star == "H_P2_swept" || return cached_stars(mn; mesh)[star]
    isdefined(Main, :p1_rows) || (ENV["LS_MAIN"] = "0"; include(joinpath(@__DIR__, "..", "stars", "local_symmetric_check.jl")))
    Hd = cached_stars(mn; mesh)["H_DEC"]; P = p1_rows(mesh; k = 2)
    r = project_rows(P, [zeros(length(P.nbv[e])) for e in 1:P.n]); r = project_rows(P, symmetrize(P, r))
    return Hd + tosparse(P, r)
end
"Steady test: relative rms errors of h and v after `days`, as in Peixoto et al. (depth error normalised by the depth perturbation)."
function run_steady(mn, star; days = 1, cfl = 0.3, log = println)
    mesh = VoronoiMesh("meshes/$mn.nc"); O = sw_ops(mesh, star_matrix(mn, star, mesh))
    c = sqrt(GJ * H0J); nsd = ceil(Int, DAY / (cfl * minimum(O.le) / (c + 2U0J))); dt = DAY / nsd
    x0 = steady_initial(O); x = copy(x0); buf = sw_buffers(O); k1, k2, k3, k4, xt = (similar(x) for _ in 1:5)
    for n in 1:days*nsd; rk4_step!(x, dt, O, buf, k1, k2, k3, k4, xt); end
    eh = x[1:O.nc] .- x0[1:O.nc]; ev = x[O.nc+1:end] .- x0[O.nc+1:end]
    rh = sqrt(sum(O.A .* eh .^ 2) / sum(O.A .* (x0[1:O.nc] .- H0J) .^ 2))
    w = O.le .* (O.H isa Diagonal ? 1 : 1)                                      # edge weights |e||ẽ| ~ area per edge
    rv = sqrt(sum(abs2, ev) / sum(abs2, x0[O.nc+1:end])); mh = maximum(abs, eh) / maximum(abs, x0[1:O.nc] .- H0J)
    log(@sprintf("steady %s %-10s h=L/%d: depth error rms %.3e max %.3e, circulation error rms %.3e", mn, star, round(Int, sqrt(O.nc)), rh, mh, rv))
    return (rh = rh, mh = mh, rv = rv, h = sqrt(1 / O.nc))
end

function run_jet(mn, star; days = 20, save_days = (4, 6, 8, 10, 14, 20), cfl = 0.3, apvm = 0.0, log = println)
    mesh = VoronoiMesh("meshes/$mn.nc")                                         # relative path: NetCDF fails on non-ASCII paths
    O = sw_ops(mesh, star_matrix(mn, star, mesh))
    c = sqrt(GJ * H0J); dt0 = cfl * minimum(O.le) / (c + 2U0J)
    nsteps_day = ceil(Int, DAY / dt0); dt = DAY / nsteps_day
    O.apvm[] = apvm; O.dt[] = dt
    log(@sprintf("%s %s APVM %.2f: %d cells, %d edges; |W+Wᵀ|/|W| = %.1e, W error on a constant field %.2e; dt = %.1f s (%d steps/day)",
                 mn, star, apvm, O.nc, O.ne, O.Wt_asym, O.werr, dt, nsteps_day))
    x = jet_initial(O); buf = sw_buffers(O); k1, k2, k3, k4, xt = (similar(x) for _ in 1:5)
    I0 = invariants(x, O); inv = [(t = 0.0, I0...)]; snaps = Dict{Int,Any}(0 => (ζ = vorticity(x, O), h = x[1:O.nc]))
    steps_out = max(1, nsteps_day ÷ 4)                                          # invariants every 6 h
    t = 0.0; tw = @elapsed for n in 1:days*nsteps_day
        rk4_step!(x, dt, O, buf, k1, k2, k3, k4, xt); t += dt
        if n % steps_out == 0
            I = invariants(x, O); push!(inv, (t = t, I...))
            isfinite(I.energy) || (log("   blew up at day $(t/DAY)"); break)
        end
        if n % nsteps_day == 0
            d = n ÷ nsteps_day
            d in save_days && (snaps[d] = (ζ = vorticity(x, O), h = x[1:O.nc]))
            I = inv[end]
            log(@sprintf("   day %2d: mass %+.2e  energy %+.2e  enstrophy %+.2e  max|ζ| %.2e", d, I.mass / I0.mass - 1, I.energy / I0.energy - 1,
                         I.enstrophy / I0.enstrophy - 1, maximum(abs, vorticity(x, O))))
        end
    end
    log(@sprintf("   %s %s: %.0f s wall time", mn, star, tw))
    R = (mn = mn, star = star, dt = dt, inv = inv, snaps = snaps, nc = O.nc, werr = O.werr, apvm = apvm)
    serialize(joinpath(TSDIR, "sw_jet_$(mn)_$(star)" * (apvm > 0 ? "_apvm" : "") * ".jls"), R)
    return R
end

if abspath(PROGRAM_FILE) == @__FILE__
    OUTJ = String[]; logj(s) = (println(s); flush(stdout); push!(OUTJ, s))
    stars = split(get(ENV, "JET_STARS", "H_DEC,H_P2"), ",")
    for mn in (isempty(ARGS) ? ["mesh5", "mesh6"] : ARGS), st in stars
        run_jet(mn, String(st); log = logj)
        open(joinpath(TSDIR, "sw_jet.txt"), "a") do io; foreach(l -> println(io, l), OUTJ); end; empty!(OUTJ)
    end
end
