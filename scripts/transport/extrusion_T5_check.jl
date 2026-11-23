# The deformational test with the period T = 5 of the planar test quoted by Nair & Lauritzen (2010), eqs. (7)-(8): u = (1/T,0) + cos(πt/T) s(x−t/T,y),
# two smooth hills; the paper uses T = 1.  Cubic mean-preserving extrusion, RK4 Δt = h/8, relative max error at T; also max at T/2 shows how thin the filaments get.
#     julia -t 4 scripts/transport/extrusion_T5_check.jl [T] [meshes]   -> ts_data/extrusion_T5.txt
include(joinpath(@__DIR__, "extrusion_lie.jl")); OUT5 = String[]; s5(s) = (println(s); flush(stdout); push!(OUT5, s))
T = isempty(ARGS) ? 5.0 : parse(Float64, ARGS[1]); ms = length(ARGS) > 1 ? ARGS[2:end] : ["mesh4", "mesh5", "mesh6"]
E = Dict{Any,Float64}()
for mn in ms
    G = lie_setup(mn; cfv = (), upw = ()); extrusion_setup!(G; p_rho = 3, rings = 2, constrained = true); mesh = G["mesh"]; A = G["A"]; h = 1 / sqrt(mesh.cells.n)
    test = merge(TESTS.deformational, (T = T,)); ρ0 = A .* cell_averages(mesh, test.ρ0)
    for st in ("H_DEC", "H_P1", "H_P2"), (nm, ic) in (("hills", ρ0), ("uniform", copy(A)))
        f = extrusion_rhs(G, st, test); ρ = copy(ic); n = ceil(Int, T / (h / 8)); dt = T / n; half = 0.0; m0 = maximum(ic ./ A)
        for k in 1:n
            t = (k - 1) * dt; k1 = f(ρ, t); k2 = f(ρ .+ dt / 2 .* k1, t + dt / 2); k3 = f(ρ .+ dt / 2 .* k2, t + dt / 2); k4 = f(ρ .+ dt .* k3, t + dt)
            ρ = ρ .+ dt / 6 .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4); k == n ÷ 2 && (half = maximum(ρ ./ A))
        end
        e = (ρ .- ic) ./ A; E[(mn, st, nm)] = maximum(abs, e) / m0
        s5(@sprintf("T = %g  %s %-5s %-8s  rel. max error %.2e   rel. L2 %.2e   max at T/2 %.3f (initial %.3f)   min at T %.4f", T, mn, st, nm, maximum(abs, e) / m0,
                    sqrt(sum(A .* e .^ 2) / sum(A .* (ic ./ A) .^ 2)), half, m0, minimum(ρ ./ A)))
    end
end
for st in ("H_DEC", "H_P1", "H_P2"), nm in ("hills", "uniform"), (a, b) in zip(ms[1:end-1], ms[2:end])
    s5(@sprintf("order %s→%s %-5s %-8s %.2f", a, b, st, nm, log2(E[(a, st, nm)] / E[(b, st, nm)])))
end
write(joinpath(@__DIR__, "..", "..", "output", "ts_data", "extrusion_T$(Int(T)).txt"), join(OUT5, "\n") * "\n")
