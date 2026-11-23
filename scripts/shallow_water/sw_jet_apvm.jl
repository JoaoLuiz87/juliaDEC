# sw_jet_apvm.jl -- the unstable jet with APVM 0.5 (the same for both stars), meshes 5-7, H_DEC and H_P2
#      julia -t 4 scripts/shallow_water/sw_jet_apvm.jl      ->  ts_data/sw_jet_<mesh>_<star>_apvm.jls, sw_jet_apvm.txt
include(joinpath(@__DIR__, "sw_jet.jl"))
OUTA = String[]; loga(s) = (println(s); flush(stdout); push!(OUTA, s))
for mn in ("mesh5", "mesh6", "mesh7"), st in ("H_DEC", "H_P2")
    run_jet(mn, st; apvm = 0.5, log = loga)
    write(joinpath(TSDIR, "sw_jet_apvm.txt"), join(OUTA, "\n") * "\n")
end
println("done")
