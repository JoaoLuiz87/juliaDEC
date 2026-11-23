# errors of W and P on the field u3 of coriolis_barrier_check.jl, in the normalisation of the proposition: max_e |(C)_e - ∫_e Ju·t| / (|e| ‖u‖∞),
# against the lower bound |k(u,u)|/(4|Ω|‖u‖²) for antisymmetric W.    julia scripts/shallow_water/coriolis_p_error_check.jl -> ts_data/coriolis_p_error.txt
include(joinpath(@__DIR__, "sw_linear_three.jl"))
const κ = 2π / LDOM; u3(x, t) = [sin(κ * (2x[1] - x[2])), 0.7 * cos(κ * x[1])]; OUTE = String[]
sup3 = maximum(norm(u3((a, b) .* LDOM, 0.0)) for a in 0:0.005:1, b in 0:0.005:1)
for mn in ("mesh4", "mesh5", "mesh6", "mesh7", "mesh5_regular", "mesh6_regular")
    mesh = VoronoiMesh("meshes/$mn.nc"); Hd = spdiagm(0 => [norm(-(edge_endpoints(mesh, e, false)...)) / norm(-(edge_endpoints(mesh, e, true)...)) for e in 1:mesh.edges.n])
    O = sw_ops(mesh, Hd); P = perp_p2(mesh, O.W * O.H); _, v, fl, cJ = cochains(O, (x, t) -> 0.0, u3, 0.0); h = 1 / sqrt(O.nc)
    k = 2 * dot(fl, cJ) / (O.L^2 * sup3^2)
    s = @sprintf("%-13s rho_W %.3e   rho_P %.3e   bound |k|/4 %.3e (k/h² %.3f)", mn, maximum(abs, (O.W * fl .- cJ) ./ O.le) / sup3, maximum(abs, (P * v .- cJ) ./ O.le) / sup3, abs(k) / 4, k / h^2)
    println(s); flush(stdout); push!(OUTE, s)
end
write(joinpath(TSDIR, "coriolis_p_error.txt"), join(OUTE, "\n") * "\n")
