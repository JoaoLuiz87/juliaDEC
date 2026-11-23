# report_figures_row.jl -- fig_rep_row.png from ts_data/report_entries.jls (see report_entries.jl): one row of three stars on mesh 5
include(joinpath(@__DIR__, "stars_cache.jl")); include(joinpath(@__DIR__, "..", "shared", "thesis_plot_helpers.jl")); using Serialization
DD = deserialize(joinpath(TSDIR, "report_entries.jls")); r = DD[:row]; mesh = VoronoiMesh("meshes/mesh5.nc"); e = r.e; lp = edge_lengths(mesh, true); ld = edge_lengths(mesh, false)
xe = mesh.edges.position[e]; x0 = _pt(xe); wr(d) = d .- round.(d); R = 0.052
f = Figure(size = (1560, 640), fontsize = 17)
for (k, nm) in enumerate(("H_P1", "H_P2", "symmetric, exact on [P2]²"))
    q = r.rows[nm]; ax = Axis(f[1, k]; aspect = DataAspect(), title = (k == 3 ? "symmetric star exact on [P2]²" : nm) * @sprintf("   Σ|H_ej||e_j|/|ẽ| = %.2f", sum(abs.(q.v) .* lp[q.j]) / ld[e]), titlealign = :left); hidedecorations!(ax); hidespines!(ax)
    for j in 1:mesh.edges.n
        a, b = edge_endpoints(mesh, j, true; about = xe); a = wr(_pt(a) .- x0); b = wr(_pt(b) .- x0); norm(a) < 1.3R && lines!(ax, [Point2f(a...), Point2f(b...)]; color = (INK2, 0.16), linewidth = 0.9)
    end
    for (j, v) in zip(q.j, q.v)
        a, b = edge_endpoints(mesh, j, true; about = xe); a = wr(_pt(a) .- x0); b = wr(_pt(b) .- x0)
        lines!(ax, [Point2f(a...), Point2f(b...)]; color = v / r.hd, colormap = DIVM, colorrange = (-1, 1), linewidth = j == e ? 11 : 7)
    end
    d = q.v[findfirst(==(e), q.j)] / r.hd; a, b = edge_endpoints(mesh, e, true; about = xe); text!(ax, -0.048, -0.05; text = @sprintf("diagonal entry: %.2f    largest off-diagonal entry: %.2f", d, maximum(abs, q.v[q.j .!= e]) / r.hd), fontsize = 15, color = INK)
    xlims!(ax, -R, R); ylims!(ax, -R, R)
end
Colorbar(f[1, 4]; colormap = DIVM, limits = (-1, 1), width = 12, label = "H_ej / (H_DEC)_ee")
CairoMakie.save(thesis_out("fig_rep_row.png"), f; px_per_unit = 2); println("ok")
