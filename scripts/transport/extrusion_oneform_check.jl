# ============================================================================
#  extrusion_oneform_check.jl -- the 1-form Lie derivative L_1 b = I_2(u, D1 b) + D0 I_1(u, b) of extrusion_oneform.jl
#
#      julia -t 4 scripts/transport/extrusion_oneform_check.jl         ->  output/ts_data/extrusion_oneform.{txt,jls}
#      (ENV: OF_MESHES=mesh2,...  OF_KEEP=1 keeps the results already saved)
#
#  A. truncation, meshes 4-7: β = bump·(1, ½), u = swirl + ½∇φ; total, vorticity part i_u dβ, gradient part d i_u β;
#     velocity reconstructed (u♯) or exact.  Also the mean-preserving vs plain least-squares vorticity fit (mesh 4-5).
#  B. structure (mesh 4): D1 L_1 = L_2 D1 and L_1 D0 = D0 L_0 through a whole RK4 run; spectra of L_2 (vorticity on
#     triangles) and L_0 (exact part) for steady translation / swirl / cellular, meshes 4-5 (plain fit too); on mesh 3 the
#     spectrum of L_1 itself against the union.
#  C. the five tests of lie_derivative.jl with the 1-form β0 = ρ0·(1, ½): errors of b/|e| and of the triangle vorticity at T.
# ============================================================================
include(joinpath(@__DIR__, "extrusion_oneform.jl"))
using Printf, Serialization
empty!(lines)
const OMESHES = haskey(ENV, "OF_MESHES") ? Tuple(String.(split(ENV["OF_MESHES"], ","))) : ("mesh2", "mesh3", "mesh4", "mesh5", "mesh6", "mesh7")
const OFILE = joinpath(TSDIR, "extrusion_oneform.jls")
R = isfile(OFILE) && get(ENV, "OF_KEEP", "0") == "1" ? deserialize(OFILE) : Dict{Any,Any}()
βtr(x) = bump(x) .* [1.0, 0.5]
ζtr(x) = (g = ∇bump(x); 0.5 * g[1] - g[2])
oneform0(test) = x -> test.ρ0(x) .* [1.0, 0.5]
relerr(e, ex, W) = (l2 = sqrt(sum(W .* e .^ 2) / sum(W .* ex .^ 2)), linf = maximum(abs.(e)) / maximum(abs.(ex)))

for mn in OMESHES
    t0 = time()
    G = lie_setup(mn; cfv = (), upw = ()); extrusion_setup!(G); oneform_setup!(G)
    mesh = G["mesh"]; h = G["h"]; le = G["le"]; W = diamond_areas(G); AT = G["AT"]; R[(mn, :h)] = h
    logp(@sprintf("\n%s: %d cells, %d edges, %d triangles, h = 1/%d (setup %.0f s)", mn, mesh.cells.n, mesh.edges.n, mesh.vertices.n, round(Int, 1 / h), time() - t0))
    P = [primal_edge(mesh, e) for e in 1:mesh.edges.n]
    # ---------------- A. truncation
    if mn in ("mesh4", "mesh5", "mesh6", "mesh7")
        b = edge_integrals(mesh, βtr); u = swirl_div
        exV = [sum((w / 2) * p[6] * ζtr(p[3] .+ (ξ / 2) .* (p[2] .- p[1])) * dot(u(p[3] .+ (ξ / 2) .* (p[2] .- p[1])), p[5]) for (ξ, w) in _GL5) for p in P]
        exG = [dot(u(p[2]), βtr(p[2])) - dot(u(p[1]), βtr(p[1])) for p in P]
        ex = (exV .+ exG) ./ le; out = String[]
        for vel in ("rec", "exact")
            ŵ, ux, uy = oneform_velocity(G, u; velocity = vel)
            V = vorticity_flux(G, ŵ, G["σT"] .* (G["D1p"] * b) ./ AT); Gp = G["D0p"] * point_contraction(G, ux, uy, b)
            for (part, τ) in (("total", (V .+ Gp) ./ le .- ex), ("vort", (V .- exV) ./ le), ("grad", (Gp .- exG) ./ le))
                R[(mn, vel, part, :trunc)] = (inf = maximum(abs.(τ)) / maximum(abs.(ex)), l2 = sqrt(sum(W .* τ .^ 2) / sum(W .* ex .^ 2)))
            end
            push!(out, @sprintf("[%s] total %.2e/%.2e vort %.2e grad %.2e", vel, R[(mn, vel, "total", :trunc)].inf, R[(mn, vel, "total", :trunc)].l2,
                                R[(mn, vel, "vort", :trunc)].inf, R[(mn, vel, "grad", :trunc)].inf))
        end
        logp("   A. trunc L1 (max/L2; parts max): " * join(out, "   "))
    end
    # ---------------- B. structure and spectra
    if mn == "mesh4"
        test = TESTS.deformational; b0 = edge_integrals(mesh, oneform0(test)); dt = h / 8
        bT = rk4(oneform_rhs(G, test), b0, test.T, dt)
        vel = [oneform_velocity(G, f) for f in test.fields]
        L2rhs = (c, t) -> (cc = test.coefs(t, test.T); ŵ = ntuple(g -> sum(ci .* V[1][g] for (ci, V) in zip(cc, vel)), 3);
                           -(G["D1p"] * vorticity_flux(G, ŵ, G["σT"] .* c ./ AT)))
        cT = rk4(L2rhs, G["D1p"] * b0, test.T, dt)
        q0 = [bump(_vec2(x)) for x in mesh.cells.position]
        L0rhs = (q, t) -> (cc = test.coefs(t, test.T); ux = sum(ci .* V[2] for (ci, V) in zip(cc, vel)); uy = sum(ci .* V[3] for (ci, V) in zip(cc, vel));
                           -point_contraction(G, ux, uy, G["D0p"] * q))
        qT = rk4(L0rhs, q0, test.T, dt); bqT = rk4(oneform_rhs(G, test), G["D0p"] * q0, test.T, dt)
        R[(mn, :commute)] = (vort = norm(G["D1p"] * bT .- cT) / norm(cT), exact = norm(bqT .- G["D0p"] * qT) / norm(G["D0p"] * qT))
        logp(@sprintf("   B. deformational run: |D1 b(T) − (vorticity run)|/|·| = %.1e,  |b(T) − D0 q(T)|/|·| (b0 = D0 q0) = %.1e",
                      R[(mn, :commute)].vort, R[(mn, :commute)].exact))
    end
    if mn in ("mesh3", "mesh4", "mesh5")
        for (fname, f) in (("translation", x -> [1.0, 1.0]), ("swirl", swirl), ("cellular", cellular))
            ŵ, ux, uy = oneform_velocity(G, f)
            L1, L2, L0 = oneform_matrices(G, ŵ, ux, uy)
            if mn == "mesh3"
                λ1 = eigvals(Matrix(-L1)); U = vcat(eigvals(Matrix(-L0)), eigvals(Matrix(-L2)), 0.0)
                R[(mn, fname, :union)] = (maxre = maximum(real.(λ1)), dist = maximum(minimum(abs.(U .- z)) for z in λ1))
                logp(@sprintf("   B. %s: spectrum of L1: max Re λ %+.1e, distance to spec(L0) ∪ spec(L2) %.1e", fname, R[(mn, fname, :union)].maxre, R[(mn, fname, :union)].dist))
                continue
            end
            R[(mn, fname, "L2", :spec)] = maximum(real.(eigvals(Matrix(-L2))))
            R[(mn, fname, "L0", :spec)] = maximum(real.(eigvals(Matrix(-L0))))
            if mn == "mesh4"; Zs = (G["Zm"], G["Zp"]); G["Zm"], G["Zp"] = tri_lsq_edge_points(mesh, 2; stencil = :edge2, constrained = false)
            _, L2p, _ = oneform_matrices(G, ŵ, ux, uy); G["Zm"], G["Zp"] = Zs
            R[(mn, fname, "L2plain", :spec)] = maximum(real.(eigvals(Matrix(-L2p)))); else R[(mn, fname, "L2plain", :spec)] = NaN; end
            logp(@sprintf("   B. max Re λ, %s: L2 (vorticity, mean-preserving) %+.1e   L2 (plain LSQ) %+.1e   L0 %+.1e", fname,
                          R[(mn, fname, "L2", :spec)], R[(mn, fname, "L2plain", :spec)], R[(mn, fname, "L0", :spec)]))
        end
    end
    # ---------------- C. the five tests
    for (name, test) in pairs(TESTS)
        β0 = oneform0(test); b0 = edge_integrals(mesh, β0)
        bex = test.exact == :initial ? b0 : exact_oneform(mesh, test, β0, test.T)
        ζex = G["σT"] .* (G["D1p"] * bex) ./ AT; out = String[]
        for vel in (mn == "mesh7" ? ("rec",) : ("rec", "exact"))
            t = @elapsed (bT = rk4(oneform_rhs(G, test; velocity = vel), b0, test.T, h / 8))
            eb = relerr(bT ./ le .- bex ./ le, bex ./ le, W); eζ = relerr(G["σT"] .* (G["D1p"] * bT) ./ AT .- ζex, ζex, AT)
            R[(mn, String(name), vel, :oneform)] = (l2 = eb.l2, linf = eb.linf, zl2 = eζ.l2, zlinf = eζ.linf, time = t)
            push!(out, @sprintf("[%s] b %.2e/%.2e  ζ %.2e/%.2e (%.0f s)", vel, eb.l2, eb.linf, eζ.l2, eζ.linf, t))
        end
        logp("   C. $name (ℓ2/ℓ∞): " * join(out, "   "))
    end
    serialize(OFILE, R)
end
write(joinpath(TSDIR, "extrusion_oneform.txt"), join(lines, "\n") * "\n")
