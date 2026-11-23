# ============================================================================
#  p2_symmetric_refs.jl -- reference curves for fig_ts_p2sym.png: H_DEC and H_P1 (exact on [P1]^2, 3-level stencil)
#  measured exactly as p2_symmetric_check.jl measures the [P2]^2 stars (Λ, r_2(d), Poisson nodal L2).
#  H_P2 is recomputed as a check: it must reproduce the "3-level row" entries of p2_symmetric.jls.
#
#      julia -t 4 scripts/stars/p2_symmetric_refs.jl      (writes output/ts_data/p2_symmetric_refs.{jls,txt})
# ============================================================================
include(joinpath(@__DIR__, "stars_cache.jl"))
empty!(lines)
R = Dict{Any,Any}()
P = deserialize(joinpath(TSDIR, "p2_symmetric.jls"))
logp("reference stars for fig_ts_p2sym.png (same measures as p2_symmetric_check.jl)")
for mn in ("mesh2", "mesh3", "mesh4", "mesh5", "mesh6")
    mesh = VoronoiMesh("meshes/$mn.nc"); h = meshsize(mesh)
    S = cached_stars(mn; mesh = mesh)
    for name in ("H_DEC", "H_P1", "H_P2")
        H = S[name]
        Λ = bigLambda(mesh, H); ed = hodge_error(mesh, H, HFIELDS.d...); pe = poisson_errors(mesh, H).l2
        R[(mn, name)] = (h = h, Λ = Λ, hodge = ed.l2, poisson = pe)
        logp(@sprintf("%s  %-6s  Λ %6.3f  r_2(d) %.3e  poisson %.3e", mn, name, Λ, ed.l2, pe))
    end
    q = P[(mn, 3, :row)]; r = R[(mn, "H_P2")]
    logp(@sprintf("%s  check H_P2 against p2_symmetric.jls (3-level, row): Λ %.1e  r_2(d) %.1e  poisson %.1e (relative differences)",
                  mn, abs(r.Λ - q.Λ) / q.Λ, abs(r.hodge - q.hodge) / q.hodge, abs(r.poisson - q.poisson) / q.poisson))
end
serialize(joinpath(TSDIR, "p2_symmetric_refs.jls"), R)
write(joinpath(TSDIR, "p2_symmetric_refs.txt"), join(lines, "\n") * "\n")
