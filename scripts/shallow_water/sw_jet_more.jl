# sw_jet_more.jl -- the steady zonal balance on meshes 3-7 and the jet on mesh 7, both stars  (julia -t 4 scripts/shallow_water/sw_jet_more.jl)
include(joinpath(@__DIR__, "sw_jet.jl"))
OUTS = String[]; logs(s) = (println(s); flush(stdout); push!(OUTS, s))
ST = Dict{Any,Any}()
for mn in ("mesh3", "mesh4", "mesh5", "mesh6", "mesh7"), st in ("H_DEC", "H_P2")
    ST[(mn, st)] = run_steady(mn, st; log = logs)
    serialize(joinpath(TSDIR, "sw_steady.jls"), ST)
end
write(joinpath(TSDIR, "sw_steady.txt"), join(OUTS, "\n") * "\n")
for st in ("H_DEC", "H_P2")
    run_jet("mesh7", st; log = logs)
    write(joinpath(TSDIR, "sw_jet_mesh7.txt"), join(OUTS, "\n") * "\n")
end
println("done")
