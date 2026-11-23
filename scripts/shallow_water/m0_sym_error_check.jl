# errors of the mass matrices A, M0c and M0s = (M0c + M0cᵀ)/2 in the normalisation of the proposition: max_v |(M0 R⁰p)_v − ∫_ṽ p| / (|ṽ| ‖p‖∞), and the
# lower bound |s0(p,q)|/(|Ω| ‖p‖ ‖q‖) for the sum of the two errors of a symmetric M0.     julia scripts/shallow_water/m0_sym_error_check.jl -> ts_data/m0_sym_error.txt
ENV["SKIPMAIN"] = "1"
include(joinpath(@__DIR__, "sw_linear_three.jl")); OUTQ = String[]
src = read(joinpath(@__DIR__, "m0_barrier_check.jl"), String); i = findfirst("tr5 = ", src)[1]; j = findfirst("rf(x) =", src)[1]
include_string(Main, src[i:j-1] * "rf(x) = sin(2π * (2x[1] - x[2]))")
for mn in ("mesh4", "mesh5", "mesh6", "mesh7")
    mesh = VoronoiMesh("meshes/$mn.nc"); A = Float64.(mesh.cells.area); M0c = mass0_corrected(mesh); M0s = (M0c + M0c') ./ 2
    X = [_pt(p) for p in mesh.cells.position]; s = @sprintf("%-6s", mn); V = Dict(); I = Dict()
    for (nm, f, sup) in (("p", pf, 1.0), ("q", qf, 1.5), ("r", rf, 1.0))
        V[nm] = [f(x) for x in X]; I[nm] = cell_integrals(mesh, f)
        s *= @sprintf("  %s: A %.2e M0c %.2e M0s %.2e", nm, (maximum(abs, (M * V[nm] .- I[nm]) ./ A) / sup for M in (spdiagm(0 => A), M0c, M0s))...)
    end
    s0 = dot(V["q"], I["r"]) - dot(V["r"], I["q"]); s *= @sprintf("   bound (q,r): %.2e", abs(s0) / 1.5)
    println(s); flush(stdout); push!(OUTQ, s)
end
write(joinpath(TSDIR, "m0_sym_error.txt"), join(OUTQ, "\n") * "\n")
