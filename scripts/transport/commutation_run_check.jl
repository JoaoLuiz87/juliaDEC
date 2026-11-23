# ============================================================================
#  commutation_run_check.jl -- D L = L D through the fully discrete runs (not only on matrices), with the right-hand sides of the paper:
#    (a) b evolved with L_1 (oneform_rhs), then D1 b(T)    against   ζ = D1 b(0) evolved on its own with L_2 = D1 I_2 (flux form on triangles)
#    (b) q evolved with L_0 (extrusion_scalar_rhs), then D0 q(T)  against  D0 q(0) evolved with L_1
#    (c) a closed form that is not exact (b = constant harmonic form dx + ½ dy): max |D1 b(T)|
#  Time-dependent flow (deformational test), RK4, Δt = h/8.  L_2 is applied by its own right-hand side (vorticity_flux of the evolved ζ).
#      julia -t 4 scripts/transport/commutation_run_check.jl [meshes]     -> ts_data/commutation_run.txt
# ============================================================================
include(joinpath(@__DIR__, "extrusion_oneform.jl"))
OUTR = String[]; sr(s) = (println(s); flush(stdout); push!(OUTR, s))
function rk4t(f, y, T, dt)
    n = ceil(Int, T / dt); dt = T / n
    for k in 1:n
        t = (k - 1) * dt; k1 = f(y, t); k2 = f(y .+ dt / 2 .* k1, t + dt / 2); k3 = f(y .+ dt / 2 .* k2, t + dt / 2); k4 = f(y .+ dt .* k3, t + dt)
        y = y .+ dt / 6 .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4)
    end
    return y
end
for mn in (isempty(ARGS) ? ["mesh4", "mesh5"] : ARGS)
    G = lie_setup(mn; cfv = (), upw = ()); extrusion_setup!(G); oneform_setup!(G); mesh = G["mesh"]; h = 1 / sqrt(mesh.cells.n); test = TESTS.deformational
    D0, D1 = G["D0p"], G["D1p"]; vel = [oneform_velocity(G, f) for f in test.fields]
    f1 = oneform_rhs(G, test); f0 = extrusion_scalar_rhs(G, test)
    f2(z, t) = (c = test.coefs(t, test.T); ŵ = ntuple(g -> sum(ci .* V[1][g] for (ci, V) in zip(c, vel)), 3); -(D1 * vorticity_flux(G, ŵ, G["σT"] .* z ./ G["AT"])))
    β0(x) = test.ρ0(x) .* [1.0, 0.5]; b0 = edge_integrals(mesh, β0); q0 = [test.ρ0(_vec2(x)) for x in mesh.cells.position]
    for T in (test.T,)
        bT = rk4t(f1, b0, T, h / 8); zT = rk4t(f2, D1 * b0, T, h / 8); dbT = D1 * bT
        qT = rk4t(f0, q0, T, h / 8); gT = rk4t(f1, D0 * q0, T, h / 8)
        hT = rk4t(f1, edge_integrals(mesh, x -> [1.0, 0.5]), T, h / 8)
        sr(@sprintf("%s  (a) max|D1 b(T) − ζ(T)|/max|ζ| %.1e   (b) max|D0 q(T) − b_q(T)|/max %.1e   (c) closed form: max|D1 b(T)|/(max|b|/h) %.1e   mean ζ drift %.1e",
                    mn, maximum(abs, dbT .- zT) / maximum(abs, zT), maximum(abs, D0 * qT .- gT) / maximum(abs, gT), maximum(abs, D1 * hT) * h / maximum(abs, hT) / h,
                    abs(sum(G["σT"] .* zT) - sum(G["σT"] .* (D1 * b0)))))
    end
end
write(joinpath(@__DIR__, "..", "..", "output", "ts_data", "commutation_run.txt"), join(OUTR, "\n") * "\n")
