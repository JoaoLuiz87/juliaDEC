# ============================================================================
#  story_transport.jl -- density transport with the cubic extrusion for H_DEC, H_P1 and H_P2
#
#      julia -t 4 scripts/transport/story_transport.jl     ->  output/ts_data/story_transport.{txt,jls}, story_frames_<mesh>.jls
#
#  Tests (unit torus, T = 1): deformational (two hills), divergent (two hills), uniform density ρ = 1 in the deformational flow
#  (stays uniform exactly: the only error is that of the swept areas Hv). Relative maximum error at T, meshes 4-7.
#  On mesh 6, frames every T/80 of the uniform and the deformational runs, for the videos.
# ============================================================================
include(joinpath(@__DIR__, "extrusion_oneform.jl"))
using Serialization, Printf
OUTT = String[]; st_(s) = (println(s); flush(stdout); push!(OUTT, s))
const STARS = ("H_DEC", "H_P1", "H_P2")
RT = Dict{Any,Any}()
function run_frames(rhs, x, T, dt; nframes = 0)
    n = round(Int, T / dt); dt = T / n; frames = Vector{Vector{Float64}}(); every = nframes > 0 ? max(1, n ÷ nframes) : 0
    nframes > 0 && push!(frames, copy(x)); t = 0.0
    for k in 1:n
        k1 = rhs(x, t); k2 = rhs(x .+ (dt / 2) .* k1, t + dt / 2); k3 = rhs(x .+ (dt / 2) .* k2, t + dt / 2); k4 = rhs(x .+ dt .* k3, t + dt)
        x = x .+ (dt / 6) .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4); t += dt
        every > 0 && k % every == 0 && push!(frames, copy(x))
    end
    return x, frames
end
for mn in ("mesh4", "mesh5", "mesh6", "mesh7")
    t0 = time()
    G = lie_setup(mn; cfv = (), upw = ()); extrusion_setup!(G; p_rho = 3, rings = 2, constrained = true)
    mesh = G["mesh"]; A = G["A"]; h = G["h"]; RT[(mn, :h)] = h
    st_(@sprintf("%s: %d cells, setup %.0f s", mn, mesh.cells.n, time() - t0))
    FR = Dict{Any,Any}()
    for test in (:deformational, :divergent, :uniform)
        tc = test == :uniform ? TESTS[:deformational] : TESTS[test]
        ρ0 = test == :uniform ? A .* 1.0 : cell_averages(mesh, tc.ρ0) .* A
        ex = test == :uniform ? ones(mesh.cells.n) : ρ0 ./ A
        for s in STARS
            nf = (mn == "mesh6" && test != :divergent) ? 80 : 0
            t1 = @elapsed ((xT, frames) = run_frames(extrusion_rhs(G, s, tc), ρ0, tc.T, h / 8; nframes = nf))
            e = xT ./ A .- ex
            RT[(mn, test, s)] = (linf = maximum(abs, e) / (test == :uniform ? 1.0 : maximum(abs, ex)), l2 = sqrt(sum(A .* e .^ 2) / sum(A .* ex .^ 2)))
            mn == "mesh6" && (RT[(mn, test, s, :final)] = xT ./ A)
            nf > 0 && (FR[(test, s)] = [f ./ A for f in frames])
            st_(@sprintf("   %-13s %-5s rel. max error %.3e  rms %.3e  (%.0f s)", test, s, RT[(mn, test, s)].linf, RT[(mn, test, s)].l2, t1))
        end
        mn == "mesh6" && (RT[(mn, test, :exact)] = ex)
    end
    isempty(FR) || serialize(joinpath(TSDIR, "story_frames_$(mn).jls"), FR)
    serialize(joinpath(TSDIR, "story_transport.jls"), RT)
    write(joinpath(TSDIR, "story_transport.txt"), join(OUTT, "\n") * "\n")
end
println("done")
