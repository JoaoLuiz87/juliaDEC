# ============================================================================
#  extrusion_check.jl -- the extrusion-based (Mullen / Eldred) Lie derivatives of extrusion_lie.jl
#
#      julia -t 4 scripts/transport/extrusion_check.jl         ->  output/ts_data/extrusion.{txt,jls}
#
#  Densities (dual 2-forms):  L_h ρ̃ = D̃ I_2(u, ρ̃), I_2 = upwind extrusion of the Voronoi edges with the swept area
#      U = H v (H_DEC, H_P1, H_P2, H_rec = the star of the velocity reconstruction, exact) and the shape of the
#      extrusion from u♯ (P2 reconstruction of u from v); density: quadratic least squares on the upwind cell's ring.
#  0-forms (scalars at the generators):  L_h q = I_1(u, D0 q) = u♯(x_c)·β♯_up(x_c), β♯ the P2 reconstruction of the
#      1-form dq from the upwind triangle; no Hodge star (velocity: u♯, or exact for reference).
#  A. truncation (u = swirl + ½∇φ; ρ = 1 + bump; q = bump), meshes 4-7
#  B. spectra (steady translation, swirl, cellular), meshes 4-5
#  C. the five tests of lie_derivative.jl (TESTS), meshes 2-7, errors at T
# ============================================================================
include(joinpath(@__DIR__, "extrusion_lie.jl"))
using Printf, Serialization
empty!(lines)
const XMESHES = haskey(ENV, "EXT_MESHES") ? Tuple(String.(split(ENV["EXT_MESHES"], ","))) : ("mesh2", "mesh3", "mesh4", "mesh5", "mesh6", "mesh7")
const XSTARS = ("H_DEC", "H_P1", "H_P2", "H_rec", "exact")
R = isfile(joinpath(TSDIR, "extrusion.jls")) && get(ENV, "EXT_KEEP", "0") == "1" ? deserialize(joinpath(TSDIR, "extrusion.jls")) : Dict{Any,Any}()

"Sparse matrix of the steady density extrusion operator (dρ̃/dt = -L ρ̃)."
function extrusion_matrix(G, U, Qv)
    ld = G["ld"]; w = (_GL3[1][2], _GL3[2][2], _GL3[3][2]); m = 0.5 .* (w[1] .* Qv[1] .+ w[2] .* Qv[2] .+ w[3] .* Qv[3])
    K = spzeros(length(U), G["mesh"].cells.n)
    for g in 1:3
        ŵ = Qv[g] .- m .+ U ./ ld
        K = K .+ Diagonal((ld ./ 2) .* w[g] .* max.(ŵ, 0)) * G["R1g"][g] .+ Diagonal((ld ./ 2) .* w[g] .* min.(ŵ, 0)) * G["R2g"][g]
    end
    return G["Div"] * K * Diagonal(1 ./ G["A"])
end
"Sparse matrix of the steady 0-form operator (dq/dt = -L q)."
function scalar_matrix(G, ux, uy)
    Bx, By, slots, rel = G["slots"]
    sel = [upwind_slot(slots, rel, c, ux[c], uy[c]) for c in 1:length(ux)]
    return (Diagonal(ux) * Bx[sel, :] .+ Diagonal(uy) * By[sel, :]) * sparse(Float64.(d0_matrix_primal(G["mesh"])))
end

for mn in XMESHES
    t0 = time()
    G = lie_setup(mn; cfv = (), upw = ()); extrusion_setup!(G); G["slots"] = oneform_slots(G["mesh"], 2)
    mesh = G["mesh"]; A = G["A"]; h = G["h"]; R[(mn, :h)] = h; pos = [_vec2(p) for p in mesh.cells.position]
    logp(@sprintf("\n%s: %d cells, h = 1/%d (setup %.0f s)", mn, mesh.cells.n, round(Int, 1 / h), time() - t0))
    # ---------------- A. truncation
    if mn in ("mesh4", "mesh5", "mesh6", "mesh7")
        ρf(x) = 1 + bump(x); ρ̄ = discretize_scalar_field_cells(mesh, ρf; qdeg = 5).values ./ A
        divex = G["Div"] * reduce_dual_edges(mesh, star_field(x -> ρf(x) .* swirl_div(x))).values; sc = maximum(abs.(divex ./ A))
        vd = reduce_primal_edges(mesh, swirl_div).values; Ud = reduce_dual_edges(mesh, star_field(swirl_div)).values
        Qd = ntuple(g -> G["Qg"][g] * vd, 3)
        Qdex = ntuple(g -> [dot(swirl_div(dual_gauss(mesh, e)[1][g]), _unit_t(mesh, e)) for e in 1:mesh.edges.n], 3)
        out = String[]
        for s in XSTARS
            U = s == "exact" ? Ud : G[s] * vd
            τ = (G["Div"] * extrusion_flux(G, U, s == "exact" ? Qdex : Qd, ρ̄) .- divex) ./ A
            R[(mn, s, :trunc)] = (inf = maximum(abs.(τ)) / sc, l2 = sqrt(sum(A .* τ .^ 2)) / sc)
            push!(out, @sprintf("%s %.2e/%.2e", s, R[(mn, s, :trunc)].inf, R[(mn, s, :trunc)].l2))
        end
        logp("   A. trunc density EXT (max/L2): " * join(out, "  "))
        q = [bump(p) for p in pos]; ex = [dot(swirl_div(p), ∇bump(p)) for p in pos]
        for vel in ("rec", "exact")
            f = extrusion_scalar_rhs(G, (fields = (swirl_div,), coefs = (t, T) -> (1.0,), T = 1.0); velocity = vel)
            e = -f(q, 0.0) .- ex
            R[(mn, "EXT0", vel, :trunc)] = (inf = maximum(abs.(e)) / maximum(abs.(ex)), l2 = sqrt(sum(A .* e .^ 2) / sum(A .* ex .^ 2)))
        end
        logp(@sprintf("   A. trunc 0-form EXT0 (max/L2): u♯ %.2e/%.2e   exact u %.2e/%.2e", R[(mn, "EXT0", "rec", :trunc)].inf,
                      R[(mn, "EXT0", "rec", :trunc)].l2, R[(mn, "EXT0", "exact", :trunc)].inf, R[(mn, "EXT0", "exact", :trunc)].l2))
    end
    # ---------------- B. spectra
    if mn in ("mesh4", "mesh5")
        for (fname, f) in (("translation", x -> [1.0, 1.0]), ("swirl", swirl), ("cellular", cellular))
            v = reduce_primal_edges(mesh, f).values; Qv = ntuple(g -> G["Qg"][g] * v, 3)
            out = String[]
            for s in ("H_DEC", "H_P2", "H_rec")
                λ = eigvals(Matrix(-extrusion_matrix(G, G[s] * v, Qv)))
                R[(mn, fname, s, :spec)] = maximum(real.(λ)); push!(out, @sprintf("EXT %s %+.1e", s, R[(mn, fname, s, :spec)]))
            end
            λ = eigvals(Matrix(-scalar_matrix(G, G["Ux"] * v, G["Uy"] * v)))
            R[(mn, fname, "EXT0", :spec)] = maximum(real.(λ)); push!(out, @sprintf("EXT0 %+.1e", R[(mn, fname, "EXT0", :spec)]))
            logp("   B. max Re λ, $fname: " * join(out, "  "))
        end
    end
    # ---------------- C. the five tests
    for (name, test) in pairs(TESTS)
        ρ0 = cell_averages(mesh, test.ρ0) .* A; ex = exact_at_T(mesh, test)
        lo, hi = extrema(ex); out = String[]
        for s in ("H_DEC", "H_P2", "H_rec", "exact")
            t = @elapsed (ρT = rk4(extrusion_rhs(G, s, test), ρ0, test.T, h / 8))
            e = ρT ./ A .- ex
            R[(mn, String(name), s, :ext)] = (l2 = sqrt(sum(A .* e .^ 2) / sum(A .* ex .^ 2)), linf = maximum(abs.(e)) / maximum(abs.(ex)),
                                              under = (minimum(ρT ./ A) - lo) / (hi - lo), mass = abs(sum(ρT) - sum(ρ0)) / sum(ρ0), time = t)
            push!(out, @sprintf("%s %.2e/%.2e", s, R[(mn, String(name), s, :ext)].l2, R[(mn, String(name), s, :ext)].linf))
        end
        # 0-form: point values at the generators; exact: initial state, or characteristics (cellular)
        q0 = [test.ρ0(p) for p in pos]
        qex = test.exact == :initial ? q0 : [exact_density(test, p, test.T) for p in pos]
        for vel in (mn == "mesh7" ? ("rec",) : ("rec", "exact"))
            t = @elapsed (qT = rk4(extrusion_scalar_rhs(G, test; velocity = vel), q0, test.T, h / 8))
            e = qT .- qex
            R[(mn, String(name), "EXT0", vel)] = (l2 = sqrt(sum(A .* e .^ 2) / sum(A .* qex .^ 2)), linf = maximum(abs.(e)) / maximum(abs.(qex)), time = t)
            push!(out, @sprintf("EXT0[%s] %.2e/%.2e", vel, R[(mn, String(name), "EXT0", vel)].l2, R[(mn, String(name), "EXT0", vel)].linf))
        end
        logp("   C. $name (ℓ2/ℓ∞): " * join(out, "  "))
    end
    serialize(joinpath(TSDIR, "extrusion.jls"), R)
end
write(joinpath(TSDIR, "extrusion.txt"), join(lines, "\n") * "\n")
