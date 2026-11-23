# sw_vortex_apvm.jl -- the stationary vortex on mesh 7 with APVM 0.5 (as in the jet runs): does it control the growth seen with H_P2?
#   julia -t 4 scripts/shallow_water/sw_vortex_apvm.jl   -> ts_data/sw_vortex_apvm.jls
push!(ARGS, "none")
include(joinpath(@__DIR__, "sw_peixoto_all.jl"))
mn = "mesh7"; mesh = VoronoiMesh("meshes/$mn.nc"); S = cached_stars(mn; mesh); RV = Dict{Any,Any}()
for apvm in (0.0, 0.5), st in ("H_DEC", "H_P1", "H_P2")
    apvm == 0.0 && st == "H_P1" && continue
    O = sw_ops(mesh, S[st]; fcor = FVX); c = sqrt(GJ * H0J); T = 10DAY; n = ceil(Int, T / (0.3 * minimum(O.le) / (c + 2U0J))); dt = T / n
    O.apvm[] = apvm; O.dt[] = dt
    x0 = vortex_initial(O); x = copy(x0); buf = sw_buffers(O); k1, k2, k3, k4, xt = (similar(x) for _ in 1:5); ηr = x0[1:O.nc] .- H0J
    perday = n ÷ 10; series = Float64[]; I0 = invariants(x, O)
    for k in 1:n
        rk4_step!(x, dt, O, buf, k1, k2, k3, k4, xt)
        k % perday == 0 && push!(series, sqrt(sum(O.A .* (x[1:O.nc] .- x0[1:O.nc]) .^ 2) / sum(O.A .* ηr .^ 2)))
    end
    I1 = invariants(x, O)
    RV[(apvm, st)] = (series = series, dE = I1.energy / I0.energy - 1)
    @printf("vortex mesh7 APVM %.1f %-5s depth rms by day: %s   ΔE %+.1e\n", apvm, st, join([@sprintf("%.2e", s) for s in series], " "), RV[(apvm, st)].dE)
    flush(stdout); serialize(joinpath(TSDIR, "sw_vortex_apvm.jls"), RV)
end
