# ============================================================================
#  sw_jet_figures.jl -- figures of the unstable jet (H_DEC against H_P2) and of the steady zonal balance
#
#      julia -t 4 scripts/shallow_water/sw_jet_figures.jl
#        fig_th_jet.png           relative vorticity on mesh 6 with APVM, days 4, 6, 8, 10, 14, 20: H_DEC, H_P2 and H_P2 − H_DEC
#        fig_th_jet_res.png       APVM, days 10 and 20 on meshes 5, 6, 7 (northern jet), H_DEC and H_P2
#        fig_th_jet_inviscid.png  without APVM, mesh 6, days 10 and 20, northern jet: grid-scale noise, H_DEC and H_P2
#        fig_th_jet_diag.png      (a) energy, (b) potential enstrophy against time; (c) ‖ζ(H_P2) − ζ(H_DEC)‖/‖ζ‖ against time;
#                                 (d) steady zonal balance: depth error after one day against h
#      ts_data/sw_jet_diag.txt
# ============================================================================
include(joinpath(@__DIR__, "sw_jet.jl"))
include(joinpath(@__DIR__, "..", "shared", "thesis_plot_helpers.jl"))
const STARS_J = ("H_DEC", "H_P2")
loadjet(mn, st, sfx = "_apvm") = (f = joinpath(TSDIR, "sw_jet_$(mn)_$(st)$(sfx).jls"); isfile(f) ? deserialize(f) : nothing)
OUTD = String[]; sayd(s) = (println(s); flush(stdout); push!(OUTD, s))
VMAX = 8.0e-5                                                             # common vorticity colour range (s⁻¹)
VORT = cgrad([colorant"#1c5cab", colorant"#7fa9d9", colorant"#f0efec", colorant"#e39a8f", colorant"#c23b3a"])
const LKM = LDOM / 1e6                                                    # domain side in 1000 km

"Triangle polygons in units of 1000 km, with periodic copies."
function jet_polys(mesh)
    tris, _, _ = triangles_ccw(mesh)
    P = [Point2f[(p[1] * LKM, p[2] * LKM) for p in T.pts] for T in tris]
    out = Vector{Vector{Point2f}}(); own = Int[]; Ls = Float32(LKM)
    for (i, Q) in enumerate(P)
        push!(out, Q); push!(own, i)
        any(p -> p[1] < 0 || p[1] > Ls || p[2] < 0 || p[2] > Ls, Q) || continue
        for sx in -1:1, sy in -1:1
            (sx, sy) == (0, 0) && continue
            push!(out, [p .+ Point2f(sx * Ls, sy * Ls) for p in Q]); push!(own, i)
        end
    end
    return out, own, tris
end
function vpanel(pos, polys, own, ζ, title; range = (-VMAX, VMAX), cmap = VORT, xl = (0, LKM), yl = (0, LKM), titlesize = 14)
    ax = Axis(pos; aspect = DataAspect(), title = title, titlesize = titlesize, titlealign = :left); hidedecorations!(ax)
    poly!(ax, polys; color = ζ[own], colormap = cmap, colorrange = range, strokewidth = 0)
    xlims!(ax, xl...); ylims!(ax, yl...); return ax
end
cbar(pos) = Colorbar(pos; colormap = VORT, limits = (-VMAX * 1e5, VMAX * 1e5), label = "ζ (10⁻⁵ s⁻¹)")

# ---------------------------------------------------------------------------- vorticity on mesh 6, APVM
let mn = "mesh6", mesh = VoronoiMesh("meshes/mesh6.nc")
    polys, own, tris = jet_polys(mesh); R = Dict(st => loadjet(mn, st) for st in STARS_J)
    days = (4, 6, 8, 10, 14, 20)
    f = Figure(size = (1500, 790), fontsize = 15)
    for (j, d) in enumerate(days)
        for (i, st) in enumerate(STARS_J)
            vpanel(f[i, j], polys, own, R[st].snaps[d].ζ, (j == 1 ? st * ", " : "") * "day $d")
        end
        Δ = R["H_P2"].snaps[d].ζ .- R["H_DEC"].snaps[d].ζ; m = maximum(abs, Δ)
        vpanel(f[3, j], polys, own, Δ, (j == 1 ? "H_P2 − H_DEC, " : "") * @sprintf("max %.1e", m); range = (-m, m), cmap = DIVM)
    end
    cbar(f[1:2, 7])
    Label(f[4, 1:6], "16384 cells (mesh 6), APVM 0.5 for both stars; domain 40 031 km × 40 031 km; difference panels: own symmetric range";
          fontsize = 14, color = INK2)
    CairoMakie.save(thesis_out("fig_th_jet.png"), f; px_per_unit = 2)
end

# ---------------------------------------------------------------------------- resolution: meshes 5, 6, 7, northern jet, APVM
let ms = [m for m in ("mesh5", "mesh6", "mesh7") if loadjet(m, "H_P2") !== nothing]
    f = Figure(size = (1500, 205 * length(ms) + 30), fontsize = 15)
    for (i, mn) in enumerate(ms)
        mesh = VoronoiMesh("meshes/$mn.nc"); polys, own, _ = jet_polys(mesh)
        for (j, (st, d)) in enumerate((("H_DEC", 10), ("H_P2", 10), ("H_DEC", 20), ("H_P2", 20)))
            R = loadjet(mn, st)
            vpanel(f[i, j], polys, own, R.snaps[d].ζ, "$(R.nc) cells, $st, day $d"; yl = (0.55LKM, 0.95LKM), titlesize = 13)
        end
    end
    cbar(f[1:length(ms), 5])
    CairoMakie.save(thesis_out("fig_th_jet_res.png"), f; px_per_unit = 2)
end

# ---------------------------------------------------------------------------- without APVM, mesh 6
let mn = "mesh6", mesh = VoronoiMesh("meshes/mesh6.nc")
    polys, own, _ = jet_polys(mesh)
    f = Figure(size = (1500, 330), fontsize = 15)
    for (j, (st, d)) in enumerate((("H_DEC", 10), ("H_P2", 10), ("H_DEC", 20), ("H_P2", 20)))
        R = loadjet(mn, st, ""); R === nothing && continue
        vpanel(f[1, j], polys, own, R.snaps[d].ζ, "no APVM, $st, day $d"; yl = (0.55LKM, 0.95LKM), titlesize = 13)
    end
    cbar(f[1, 5])
    CairoMakie.save(thesis_out("fig_th_jet_inviscid.png"), f; px_per_unit = 2)
end

# ---------------------------------------------------------------------------- diagnostics
let f = Figure(size = (1500, 470), fontsize = 15)
    ax1 = Axis(f[1, 1]; xlabel = "day", ylabel = "|E(t)/E(0) − 1|", yscale = log10, title = "(a) energy", axstyle_th...)
    ax2 = Axis(f[1, 2]; xlabel = "day", ylabel = "Z(t)/Z(0) − 1", title = "(b) potential enstrophy", axstyle_th...)
    ax3 = Axis(f[1, 3]; xlabel = "day", ylabel = "‖ζ(H_P2) − ζ(H_DEC)‖ / ‖ζ(H_DEC)‖", yscale = log10, title = "(c) difference of the stars", axstyle_th...)
    ax4 = Axis(f[1, 4]; xscale = log10, yscale = log10, xreversed = true, xlabel = "h", ylabel = "rms depth error, 1 day",
               title = "(d) steady zonal balance", axstyle_th...)
    ls = Dict("mesh5" => :dot, "mesh6" => :solid, "mesh7" => :dash)
    for (sfx, tag) in (("_apvm", "APVM"), ("", "no APVM")), mn in ("mesh5", "mesh6", "mesh7")
        for st in STARS_J
            R = loadjet(mn, st, sfx); R === nothing && continue
            t = [p.t / DAY for p in R.inv]; E0 = R.inv[1].energy; Z0 = R.inv[1].enstrophy
            if sfx == "_apvm"
                lines!(ax1, t[2:end], max.(abs.([p.energy / E0 - 1 for p in R.inv[2:end]]), 1e-16); color = STARCOL[st], linestyle = ls[mn],
                       label = "$(R.nc) cells, $st")
                lines!(ax2, t, [p.enstrophy / Z0 - 1 for p in R.inv]; color = STARCOL[st], linestyle = ls[mn], label = "$(R.nc) cells, $st")
            end
            sayd(@sprintf("%s %-5s %-8s energy drift day 20 %+.2e, enstrophy change %+.2e, max|ζ| day 20 %.2e", mn, st, tag,
                          R.inv[end].energy / E0 - 1, R.inv[end].enstrophy / Z0 - 1, maximum(abs, R.snaps[20].ζ)))
        end
        Rd = loadjet(mn, "H_DEC", sfx); Rp = loadjet(mn, "H_P2", sfx); (Rd === nothing || Rp === nothing) && continue
        mesh = VoronoiMesh("meshes/$mn.nc"); tris, _, _ = triangles_ccw(mesh); AT = [T.area for T in tris]
        dd = sort(filter(>(0), collect(keys(Rd.snaps))))
        dif = [sqrt(sum(AT .* (Rp.snaps[d].ζ .- Rd.snaps[d].ζ) .^ 2) / sum(AT .* Rd.snaps[d].ζ .^ 2)) for d in dd]
        sfx == "_apvm" && scatterlines!(ax3, Float64.(dd), dif; color = INK, linestyle = ls[mn], marker = :circle, label = "$(Rd.nc) cells")
        sayd(@sprintf("%s %-8s ‖ζ(H_P2) − ζ(H_DEC)‖/‖ζ‖: ", mn, tag) * join([@sprintf("d%d %.2e", d, x) for (d, x) in zip(dd, dif)], "  "))
    end
    if isfile(joinpath(TSDIR, "sw_steady.jls"))
        ST = deserialize(joinpath(TSDIR, "sw_steady.jls")); ms = [m for m in ("mesh3", "mesh4", "mesh5", "mesh6", "mesh7") if haskey(ST, (m, "H_P2"))]
        hs = [ST[(m, "H_DEC")].h for m in ms]
        for st in STARS_J
            scatterlines!(ax4, hs, [ST[(m, st)].rh for m in ms]; color = STARCOL[st], marker = STARMK[st], markersize = 11, label = st)
        end
        xx = [hs[2], hs[end]]
        lines!(ax4, xx, 0.6 * ST[(ms[2], "H_DEC")].rh .* (xx ./ hs[2]) .^ 2; color = INK2, linestyle = :dot)
        lines!(ax4, xx, 0.6 * ST[(ms[2], "H_DEC")].rh .* (xx ./ hs[2]); color = INK2, linestyle = :dash)
        text!(ax4, xx[2], 0.6 * ST[(ms[2], "H_DEC")].rh * (xx[2] / hs[2])^2; text = " h²", align = (:left, :center), fontsize = 14, color = INK2)
        text!(ax4, xx[2], 0.6 * ST[(ms[2], "H_DEC")].rh * (xx[2] / hs[2]); text = " h", align = (:left, :center), fontsize = 14, color = INK2)
        ax4.xticks = (hs, ["1/" * string(round(Int, 1 / h)) for h in hs])
        for m in ms; sayd(@sprintf("steady %s: depth rms H_DEC %.3e  H_P2 %.3e   max H_DEC %.3e  H_P2 %.3e   circulation rms H_DEC %.3e  H_P2 %.3e", m,
                                   ST[(m, "H_DEC")].rh, ST[(m, "H_P2")].rh, ST[(m, "H_DEC")].mh, ST[(m, "H_P2")].mh, ST[(m, "H_DEC")].rv, ST[(m, "H_P2")].rv)); end
        axislegend(ax4; position = :lb, framevisible = false)
    end
    axislegend(ax3; position = :rb, framevisible = false)
    Legend(f[2, 1:2], ax2; orientation = :horizontal, framevisible = false, nbanks = 2)
    Label(f[2, 3:4], "(a)–(c): APVM 0.5; line style = mesh (dotted 5, solid 6, dashed 7)"; fontsize = 14, color = INK2)
    CairoMakie.save(thesis_out("fig_th_jet_diag.png"), f; px_per_unit = 2)
end
write(joinpath(TSDIR, "sw_jet_diag.txt"), join(OUTD, "\n") * "\n")
println("jet figures written")
