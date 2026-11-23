# ============================================================================
#  sw_linear_rot_figures.jl -- figures for the linear rotating tests (Peixoto et al. §5.2), H_DEC against H_P2
#
#      julia -t 4 scripts/shallow_water/sw_linear_rot_figures.jl
#        fig_th_swlin_conv.png    depth error at T against h: plane wave, Poincaré wave, geostrophic balance; four configurations
#        fig_th_swlin_energy.png  (a) energy with the implicit midpoint rule (M0 = A), meshes 4 and 5; (b) energy change at T with RK4
#        fig_th_swlin_sol.png     plane wave at T on mesh 5: exact η, H_DEC, H_P2 and their errors, for M0 = A and the corrected M0
# ============================================================================
ENV["SW_MAIN"] = "0"
include(joinpath(@__DIR__, "sw_linear_rot.jl"))
include(joinpath(@__DIR__, "..", "shared", "thesis_plot_helpers.jl"))
RL = deserialize(joinpath(TSDIR, "sw_linear_rot.jls"))
MS = [m for m in ("mesh2", "mesh3", "mesh4", "mesh5", "mesh6", "mesh7") if haskey(RL, (m, "H_P2", :balance, :C))]
CONF = (("H_DEC", :A, "H_DEC, M0 = A (TRiSK)", C_DEC, :circle, :solid), ("H_DEC", :C, "H_DEC, corrected M0", C_DEC, :utriangle, :dash),
        ("H_P2", :A, "H_P2, M0 = A", C_P2, :rect, :solid), ("H_P2", :C, "H_P2, corrected M0", C_P2, :diamond, :dash))
TT = Dict(:plane => "plane wave with rotation (1.5 days)", :poincare => "Poincaré wave (one period)", :balance => "geostrophic balance (1 day)")

let f = Figure(size = (1500, 470), fontsize = 15), hs = [RL[(m, "H_DEC", :plane, :A)].h for m in MS]; local ax
    for (j, test) in enumerate((:plane, :poincare, :balance))
        y0 = RL[(MS[min(3, length(MS))], "H_DEC", test, :A)].rh
        ax = conv_axis(f[1, j], hs, TT[test]; ylabel = j == 1 ? "relative rms depth error at T" : "", anchor = 0.5y0, slopes = (1, 2))
        for (st, mk, lab, col, marker, ls) in CONF
            scatterlines!(ax, hs, [RL[(m, st, test, mk)].rh for m in MS]; color = col, marker = marker, linestyle = ls, markersize = 10,
                          linewidth = 2.2, label = lab)
        end
    end
    Legend(f[2, 1:3], ax; orientation = :horizontal, framevisible = false)
    CairoMakie.save(thesis_out("fig_th_swlin_conv.png"), f; px_per_unit = 2)
end

let f = Figure(size = (1500, 450), fontsize = 15)
    ax1 = Axis(f[1, 1]; xlabel = "day", ylabel = "|E(t)/E(0) − 1|", yscale = log10, title = "(a) implicit midpoint, M0 = A, plane wave", axstyle_th...)
    for (mn, ls) in (("mesh4", :solid), ("mesh5", :dash)), st in ("H_DEC", "H_P2")
        haskey(RL, (mn, st, :midpoint)) || continue
        Es = RL[(mn, st, :midpoint)]; t = (1:length(Es)) .* (15 / length(Es))
        lines!(ax1, t, max.(abs.(Es), 1e-17); color = STARCOL[st], linestyle = ls, linewidth = 2.2, label = "$st, $(mn)")
    end
    axislegend(ax1; position = :rc, framevisible = false)
    ax2 = Axis(f[1, 2]; xscale = log10, yscale = log10, xreversed = true, xlabel = "h", ylabel = "max_t |E(t)/E(0) − 1|, RK4",
               title = "(b) energy error with RK4 (plane wave)", axstyle_th..., xticks = ([RL[(m, "H_DEC", :plane, :A)].h for m in MS],
               ["1/" * string(round(Int, 1 / RL[(m, "H_DEC", :plane, :A)].h)) for m in MS]))
    for (st, mk, lab, col, marker, ls) in CONF
        scatterlines!(ax2, [RL[(m, st, :plane, mk)].h for m in MS], [max(RL[(m, st, :plane, mk)].dEmax, 1e-17) for m in MS];
                      color = col, marker = marker, linestyle = ls, markersize = 10, linewidth = 2.2, label = lab)
    end
    axislegend(ax2; position = :lb, framevisible = false, labelsize = 13)
    CairoMakie.save(thesis_out("fig_th_swlin_energy.png"), f; px_per_unit = 2)
end

let mn = "mesh5", mesh = VoronoiMesh("meshes/mesh5.nc"), S = cached_stars("mesh5"; mesh = VoronoiMesh("meshes/mesh5.nc"))
    η, u, T = exact_lin(:plane); polys, own = periodic_copies(cellpolys(mesh))
    f = Figure(size = (1500, 640), fontsize = 15)
    for (i, mk) in enumerate((:A, :C))
        sol = Dict{String,Any}(); hT = nothing
        for st in ("H_DEC", "H_P2")
            O = sw_ops(mesh, S[st]); h0, v0, _, _ = cochains(O, η, u, 0.0); hT, _, _, _ = cochains(O, η, u, T)
            rhs, _, _ = lin_ops(O, mk); n = ceil(Int, T / (0.1 * minimum(O.le) / sqrt(GJ * H0J)))
            sol[st] = rk4_lin(rhs, vcat(h0, v0), T / n, n)[1:O.nc]
        end
        r = extrema(hT); lab = mk == :A ? "M0 = A" : "corrected M0"
        map_panel(f[i, 1], polys, own, hT, "exact η at T = 1.5 days"; range = r)
        for (j, st) in enumerate(("H_DEC", "H_P2"))
            map_panel(f[i, 1+j], polys, own, sol[st], "$st, $lab"; range = r)
            err_panel(f[i, 3+j], polys, own, sol[st] .- hT, "error $st, $lab")
        end
    end
    CairoMakie.save(thesis_out("fig_th_swlin_sol.png"), f; px_per_unit = 2)
end
println("linear rotating figures written")
