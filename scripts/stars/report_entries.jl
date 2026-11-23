# report_entries.jl -- how the entries of a symmetric star exact on [P2]² grow: least-change star on the 4-ring, meshes 2-5 (largest off-diagonal entry relative to the
# largest entry of H_DEC, largest row sum Λ), next to H_P1 and H_P2; figure of one row on mesh 5.       julia -t 4 scripts/stars/report_entries.jl -> ts_data/report_entries.{txt,jls}, fig_rep_row.png
include(joinpath(@__DIR__, "stars_cache.jl")); include(joinpath(@__DIR__, "..", "shared", "thesis_plot_helpers.jl")); using Serialization
src_sdp = read(joinpath(@__DIR__, "p2_symmetric_sdp.jl"), String)
for nm in ("function sym_system", "assembleH(sys, c)")
    i = findfirst(nm, src_sdp)[1]; j = nm == "assembleH(sys, c)" ? findnext("\n\n", src_sdp, i)[1] : findnext("\nend\n", src_sdp, i)[end]
    include_string(Main, src_sdp[i:j])
end
OUTE2 = String[]; se2(s) = (println(s); flush(stdout); push!(OUTE2, s)); DD = Dict{Any,Any}()
function stats(H, hd)
    off = 0.0; rv = rowvals(H); nz = nonzeros(H); for c in 1:size(H, 2), p in nzrange(H, c); rv[p] == c || (off = max(off, abs(nz[p]))); end
    return off / maximum(hd), maximum(abs.(diag(H))) / maximum(hd)
end
for mn in ("mesh2", "mesh3", "mesh4", "mesh5")
    mesh = VoronoiMesh("meshes/$mn.nc"); sys = sym_system(mesh, 2, 4); H = assembleH(sys, sys.c0); hd = sys.hd; o, d = stats(H, hd); Λ = bigLambda(mesh, H)
    S = cached_stars(mn; mesh); o1, d1 = stats(S["H_P1"], hd); o2, d2 = stats(S["H_P2"], hd)
    se2(@sprintf("%s  symmetric [P2]² least change: largest off-diagonal entry / max H_DEC %.3f, largest diagonal entry %.3f, Λ %.2f   |  H_P1 %.4f %.3f   H_P2 %.4f %.3f", mn, o, d, Λ, o1, d1, o2, d2))
    DD[(mn, :sym)] = (off = o, dia = d, Λ = Λ)
    if mn == "mesh5"
        lp = edge_lengths(mesh, true); ld = edge_lengths(mesh, false); Ht = sparse(H'); rs = [sum(abs(nonzeros(Ht)[p]) * lp[rowvals(Ht)[p]] for p in nzrange(Ht, e)) / ld[e] for e in 1:mesh.edges.n]; e = argmax(rs)
        rows = Dict{String,Any}()
        for (nm, M) in (("symmetric, exact on [P2]²", H), ("H_P1", S["H_P1"]), ("H_P2", S["H_P2"]))
            Mt = sparse(M'); rows[nm] = (j = rowvals(Mt)[nzrange(Mt, e)], v = nonzeros(Mt)[nzrange(Mt, e)])
        end
        DD[:row] = (e = e, rows = rows, hd = hd[e], lam = rs[e])
        se2(@sprintf("mesh5 row %d with the largest Σ|H_ej||e_j|/|ẽ_e| = %.2f of the symmetric star; H_ee(H_DEC) = %.3f", e, rs[e], hd[e]))
        serialize(joinpath(TSDIR, "report_entries.jls"), DD)
        f = Figure(size = (1560, 560), fontsize = 16); xe = mesh.edges.position[e]; x0 = _pt(xe); wr(d) = d .- round.(d)
        for (k, nm) in enumerate(("H_P1", "H_P2", "symmetric, exact on [P2]²"))
            r = rows[nm]; ax = Axis(f[1, k]; aspect = DataAspect(), title = (k == 3 ? "symmetric star exact on [P2]²" : nm) * @sprintf(":  Σ|H_ej||e_j|/|ẽ| = %.2f", sum(abs.(r.v) .* lp[r.j]) / ld[e]), titlealign = :left); hidedecorations!(ax); hidespines!(ax)
            for j in 1:mesh.edges.n
                a, b = edge_endpoints(mesh, j, true; about = xe); a = wr(_pt(a) .- x0); b = wr(_pt(b) .- x0); norm(a) < 0.11 && lines!(ax, [Point2f(a...), Point2f(b...)]; color = (INK2, 0.18), linewidth = 0.8)
            end
            m = 0.4
            for (j, v) in zip(r.j, r.v)
                a, b = edge_endpoints(mesh, j, true; about = xe); a = wr(_pt(a) .- x0); b = wr(_pt(b) .- x0)
                lines!(ax, [Point2f(a...), Point2f(b...)]; color = v / hd[e], colormap = DIVM, colorrange = (-m, m), linewidth = j == e ? 7 : 5)
                abs(v / hd[e]) > 0.15 && text!(ax, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2; text = @sprintf("%.2f", v / hd[e]), fontsize = 13, align = (:center, :bottom))
            end
            xlims!(ax, -0.085, 0.085); ylims!(ax, -0.085, 0.085)
        end
        Colorbar(f[1, 4]; colormap = DIVM, limits = (-0.4, 0.4), width = 12, label = "H_ej / H_ee of H_DEC")
        CairoMakie.save(thesis_out("fig_rep_row.png"), f; px_per_unit = 2)
    end
end
write(joinpath(TSDIR, "report_entries.txt"), join(OUTE2, "\n") * "\n")
