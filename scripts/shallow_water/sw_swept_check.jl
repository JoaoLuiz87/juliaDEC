# ============================================================================
#  sw_swept_check.jl -- the swept stars of local_symmetric_check.jl in the linear shallow-water equations (f = 0)
#
#      julia -t 6 scripts/shallow_water/sw_swept_check.jl        ->  output/ts_data/sw_swept.{txt,jls}
#      (ENV: SWS_PARTS=123 selects the parts; SWS_KEEP=1 keeps results already saved)
#
#      M0 dh/dt = H0 D0ᵀ H v ,   dv/dt = −g D0 h ,   g = H0 = 1          (setting and functions of sw_linear_check.jl)
#
#  Stars:  DEC;  H_P1 (global symmetric);  "P1 loc" = the local star exact on [P1]^2 (row by row, 23-entry pattern);
#          H_P2 (19 entries, the star of the other notes);  "P2 s" = H_P2 on the 23-entry pattern after s sweeps
#          (s = 0, 1, 2, 4, 16), each sweep = symmetrise the correction, then make every row exact again.
#  Part 1  spectrum of the wave operator, M0 = A and M0 = A + C0 (corrected 0-form star): largest growth rate
#          σ/ω_max and where it sits, share of growing modes, max|Im λ|/λmax, RK4 amplification − 1 at ω_max Δt = 0.5, 1.5.
#  Part 2  energy with a neutral time scheme (implicit midpoint): smooth travelling waves up to t = 20, and random noise
#          for 500 periods of the fastest wave (meshes 3-4); the same noise with RK4 at ω_max Δt = 1.5.
#  Part 3  accuracy at t = 1 (RK4, Δt = 10⁻³), M0 = A and M0 = A + C0, meshes 2-6.
# ============================================================================
ENV["LS_MAIN"] = "0"; include(joinpath(@__DIR__, "..", "stars", "local_symmetric_check.jl"))
ENV["SW_MAIN"] = "0"; include(joinpath(@__DIR__, "sw_linear_check.jl"))
using Printf, LinearAlgebra, SparseArrays, Serialization, Random
BLAS.set_num_threads(max(1, Threads.nthreads()))
const LOGS = String[]
say(s) = (println(s); flush(stdout); push!(LOGS, s))
const PARTS_SW = get(ENV, "SWS_PARTS", "123")
const SWF = joinpath(@__DIR__, "..", "..", "output", "ts_data", "sw_swept.jls")
RW = isfile(SWF) && get(ENV, "SWS_KEEP", "0") == "1" ? deserialize(SWF) : Dict{Any,Any}()
const SWEEPS = (0, 1, 2, 4, 16)

"All stars of the comparison on one mesh, as (name, H)."
function star_set(mesh, mn)
    S = cached_stars(mn; mesh); Hd = hdiag(mesh)
    out = Any[("DEC", Hd), ("H_P1", S["H_P1"]), ("P1 loc", nothing), ("H_P2", S["H_P2"])]
    D1 = p1_rows(mesh; k = 1); out[3] = ("P1 loc", Hd + tosparse(D1, project_rows(D1, [zeros(length(D1.nbv[e])) for e in 1:D1.n])))
    D2 = p1_rows(mesh; k = 2); rows = project_rows(D2, [zeros(length(D2.nbv[e])) for e in 1:D2.n])
    for s in 0:maximum(SWEEPS)
        s in SWEEPS && push!(out, ("P2 s=$s", Hd + tosparse(D2, rows)))
        s < maximum(SWEEPS) && (rows = project_rows(D2, symmetrize(D2, rows)))
    end
    return out
end
const NAMES = vcat(["DEC", "H_P1", "P1 loc", "H_P2"], ["P2 s=$s" for s in SWEEPS])

# ---------------------------------------------------------------------------- Part 1: spectrum
if occursin('1', PARTS_SW)
    say("Part 1: spectrum of the wave operator.  σ = |Im √λ| growth rate, ω = |Re √λ| frequency, ω_max = √max|λ|")
    say("   σ/ω_max: largest growth rate (at ω/ω_max);  growing: share of eigenvalues with Im λ ≠ 0;  RK4: largest amplification − 1 at ω_max Δt = 0.5 and 1.5")
    for mn in ("mesh3", "mesh4", "mesh5")
        mesh = VoronoiMesh("meshes/$mn.nc"); M0c = mass0_corrected(mesh); N = mesh.cells.n
        say(@sprintf("%s (%d cells)", mn, N))
        for (name, H) in star_set(mesh, mn), (m0name, M0) in (("A", nothing), ("A+C0", M0c))
            m0name == "A+C0" && !(name in ("DEC", "H_P2", "P2 s=0", "P2 s=1", "P2 s=4")) && continue
            λ, ω, σ, ωm = wave_spectrum(mesh, H; M0 = M0); i = argmax(σ)
            res_ = [σ[j] / ω[j] for j in eachindex(λ) if 1e-8 * ωm < ω[j] < ωm / 4]
            q = (sig = σ[i] / ωm, at = ω[i] / ωm, im = maximum(abs.(imag.(λ))) / ωm^2, frac = count(abs.(imag.(λ)) .> 1e-9 * ωm^2) / N,
                 res = maximum(res_), a05 = rk4_amp(λ, 0.5 / ωm) - 1, a15 = rk4_amp(λ, 1.5 / ωm) - 1, om = ωm, lam = λ)
            RW[(mn, name, m0name, :spec)] = q
            say(@sprintf("   %-8s M0 = %-4s  ω_max %6.1f  σ/ω_max %.1e (at %.2f)  growing %5.2f%%  max|Im λ|/λmax %.1e  resolved σ/ω %.0e  RK4 %+.0e %+.0e",
                         name, m0name, ωm, q.sig, q.at, 100q.frac, q.im, q.res, q.a05, q.a15))
        end
        serialize(SWF, RW)
    end
end

# ---------------------------------------------------------------------------- Part 2: energy
if occursin('2', PARTS_SW)
    say("\nPart 2: energy E = g/2 hᵀAh + H0/2 vᵀ sym(H) v with the implicit midpoint rule (neutral in time), M0 = A")
    say("   smooth: travelling waves, Δt = 0.01 up to t = 20; max_t |E/E0 − 1|")
    say("   noise : random h, v; Δt = 0.25/ω_max for 500 periods of the fastest wave: E(T)/E0;  and RK4 at ω_max Δt = 1.5: E(T)/E0")
    for mn in ("mesh3", "mesh4", "mesh5")
        mesh = VoronoiMesh("meshes/$mn.nc"); nc = mesh.cells.n
        hf, uf = wave(mesh)
        x0 = Vector{Float64}(vcat([hf(_pt(p), 0.0) for p in mesh.cells.position], reduce_primal_edges(mesh, x -> uf(x, 0.0)).values))
        say(mn)
        for (name, H) in star_set(mesh, mn)
            Mb = sw_block(mesh, H); Hs = (H + H') / 2; E(x) = energy(mesh, Hs, x)
            _, dE, dev = midpoint_run(Mb, x0, 0.01, 2000, 20, E)
            line = @sprintf("   %-8s smooth: max|E/E0−1| %.1e (at t = 20: %+.1e)", name, dev, dE); q = (dev = dev, dE = dE, noise = NaN, rk = NaN)
            if mn != "mesh5" && haskey(RW, (mn, name, "A", :spec))
                ωm = RW[(mn, name, "A", :spec)].om; T = 500 * 2π / ωm
                Random.seed!(3); xn = vcat(randn(nc), randn(mesh.edges.n) .* sqrt.(mesh.cells.area[1] ./ diag(Hs)))
                dt = 0.25 / ωm; ns = ceil(Int, T / dt)
                _, dEn, _ = midpoint_run(Mb, xn, T / ns, ns, ns, E)
                nr = ceil(Int, T / (1.5 / ωm)); xr = rk4(x -> Mb * x, xn, T / nr, nr)
                q = (dev = dev, dE = dE, noise = 1 + dEn, rk = E(xr) / E(xn))
                line *= @sprintf("   noise: E(T)/E0 %.4f   RK4 1.5: E(T)/E0 %.4f", q.noise, q.rk)
            end
            RW[(mn, name, :energy)] = q; say(line)
        end
        serialize(SWF, RW)
    end
end

# ---------------------------------------------------------------------------- Part 3: accuracy
if occursin('3', PARTS_SW)
    say("\nPart 3: accuracy at t = 1 (RK4, Δt = 1e-3).  e(h) = max|h − h_exact|/max|h|, e(v) = max_e |v − R_h u|/(|e| max|u|);  dE = E(1)/E(0) − 1")
    for mn in ("mesh2", "mesh3", "mesh4", "mesh5", "mesh6")
        mesh = VoronoiMesh("meshes/$mn.nc"); nc = mesh.cells.n
        hf, uf = wave(mesh); pts = [_pt(p) for p in mesh.cells.position]
        x0 = Vector{Float64}(vcat(hf.(pts, 0.0), reduce_primal_edges(mesh, x -> uf(x, 0.0)).values))
        hT = hf.(pts, 1.0); vT = Vector{Float64}(reduce_primal_edges(mesh, x -> uf(x, 1.0)).values)
        lp = edge_lengths(mesh, true); d0 = 1.0 * d0_matrix_primal(mesh); M0c = mass0_corrected(mesh); F0c = lu(M0c)
        say(@sprintf("%s (h = 1/%d)", mn, round(Int, sqrt(nc))))
        for (name, H) in star_set(mesh, mn)
            nsteps = 1000; dt = 1.0 / nsteps
            Mb = sw_block(mesh, H); E(x) = energy(mesh, (H + H') / 2, x)
            xa = rk4(x -> Mb * x, x0, dt, nsteps)
            fc(x) = vcat(DEPTH .* (F0c \ (d0' * (H * x[nc+1:end]))), -GRAV .* (d0 * x[1:nc]))
            xc = rk4(fc, x0, dt, nsteps)
            err(x) = (maximum(abs.(x[1:nc] .- hT)) / AMP, maximum(abs.(x[nc+1:end] .- vT) ./ lp) / AMP)
            ea, ec = err(xa), err(xc)
            RW[(mn, name, :acc)] = (ha = ea[1], va = ea[2], hc = ec[1], vc = ec[2], dE = E(xa) / E(x0) - 1, h = sqrt(1 / nc))
            say(@sprintf("   %-8s M0 = A: e(h) %.2e  e(v) %.2e  dE %+.1e     M0 = A + C0: e(h) %.2e  e(v) %.2e", name, ea..., E(xa) / E(x0) - 1, ec...))
        end
        serialize(SWF, RW)
    end
    say("\n   observed orders of e(h) with M0 = A + C0 (mesh 2→3→4→5→6):")
    for name in NAMES
        ms = [m for m in ("mesh2", "mesh3", "mesh4", "mesh5", "mesh6") if haskey(RW, (m, name, :acc))]
        say(@sprintf("   %-8s %s", name, join([@sprintf("%5.2f", log2(RW[(ms[i], name, :acc)].hc / RW[(ms[i+1], name, :acc)].hc)) for i in 1:length(ms)-1], " ")))
    end
end
write(joinpath(@__DIR__, "..", "..", "output", "ts_data", "sw_swept" * (PARTS_SW == "123" ? "" : "_part$(PARTS_SW)") * ".txt"), join(LOGS, "\n") * "\n")
