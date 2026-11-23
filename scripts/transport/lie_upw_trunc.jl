# ============================================================================
#  lie_upw_trunc.jl -- truncation error of the upwind interior product UPW2 (same setting as part A of
#  lie_transport_check.jl: u = swirl + ½∇φ, ρ = 1 + bump), meshes 4-7      ->  output/ts_data/lie_upw_trunc.{txt,jls}
# ============================================================================
include(joinpath(@__DIR__, "lie_derivative.jl"))
using Printf, Serialization
empty!(lines); R = Dict{Any,Any}()
for mn in ("mesh4", "mesh5", "mesh6", "mesh7")
    G = lie_setup(mn; cfv = (), upw = (2,)); mesh = G["mesh"]; A = G["A"]; R[(mn, :h)] = G["h"]
    ρf(x) = 1 + bump(x)
    ρ̄ = discretize_scalar_field_cells(mesh, ρf; qdeg = 5).values ./ A
    vd = reduce_primal_edges(mesh, swirl_div).values; Ud = reduce_dual_edges(mesh, star_field(swirl_div)).values
    divex = G["Div"] * reduce_dual_edges(mesh, star_field(x -> ρf(x) .* swirl_div(x))).values
    scale = maximum(abs.(divex ./ A))
    for s in ("H_DEC", "H_P1", "H_P2", "exact")
        U = s == "exact" ? Ud : G[s] * vd
        τ = (G["Div"] * upwind_flux(U, G["up2"], ρ̄) .- divex) ./ A
        R[(mn, s, "UPW2", :trunc)] = (inf = maximum(abs.(τ)) / scale, l2 = sqrt(sum(A .* τ .^ 2)) / scale)
        logp(@sprintf("%s %-6s UPW2: trunc max %.3e  L2 %.3e", mn, s, R[(mn, s, "UPW2", :trunc)].inf, R[(mn, s, "UPW2", :trunc)].l2))
    end
    serialize(joinpath(TSDIR, "lie_upw_trunc.jls"), R)
end
write(joinpath(TSDIR, "lie_upw_trunc.txt"), join(lines, "\n") * "\n")
