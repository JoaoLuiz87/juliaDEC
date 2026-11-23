# Undershoots of the (linear, unlimited) cubic extrusion scheme: minimum and maximum of the cell means over the run, and at T.
#   julia -t 4 scripts/transport/extrusion_bounds_check.jl [meshes]   -> ts_data/extrusion_bounds.txt
include(joinpath(@__DIR__, "extrusion_lie.jl")); OUTB = String[]; sb(s) = (println(s); flush(stdout); push!(OUTB, s))
for mn in (isempty(ARGS) ? ["mesh4", "mesh5"] : ARGS)
    G = lie_setup(mn; cfv = (), upw = ()); extrusion_setup!(G; p_rho = 3, rings = 2, constrained = true); mesh = G["mesh"]; A = G["A"]; h = 1 / sqrt(mesh.cells.n)
    for tn in (:deformational, :cosine_bells), st in ("H_DEC", "H_P2")
        test = getfield(TESTS, tn); f = extrusion_rhs(G, st, test); ρ = A .* cell_averages(mesh, test.ρ0); ρ0 = copy(ρ)
        n = ceil(Int, test.T / (h / 8)); dt = test.T / n; lo = minimum(ρ ./ A); hi = maximum(ρ ./ A); lo0, hi0 = lo, hi
        for k in 1:n
            t = (k - 1) * dt; k1 = f(ρ, t); k2 = f(ρ .+ dt / 2 .* k1, t + dt / 2); k3 = f(ρ .+ dt / 2 .* k2, t + dt / 2); k4 = f(ρ .+ dt .* k3, t + dt)
            ρ = ρ .+ dt / 6 .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4); lo = min(lo, minimum(ρ ./ A)); hi = max(hi, maximum(ρ ./ A))
        end
        sb(@sprintf("%s %-14s %-5s initial [%.4f, %.4f]   over the run [%.4f, %.4f]   at T [%.4f, %.4f]   mass change %.1e", mn, tn, st, lo0, hi0, lo, hi,
                    minimum(ρ ./ A), maximum(ρ ./ A), abs(sum(ρ) - sum(ρ0)) / sum(ρ0)))
    end
end
write(joinpath(@__DIR__, "..", "..", "output", "ts_data", "extrusion_bounds.txt"), join(OUTB, "\n") * "\n")
