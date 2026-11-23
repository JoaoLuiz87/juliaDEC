# ============================================================================
#  gap_barrier_figure.jl -- fig_gap_barrier.png for Section 3.2 of paper/hodge_gap.tex (Theorem barrier, Corollary size).
#
#      julia scripts/barrier/gap_barrier_figure.jl      (writes output/ts_data/barrier_skew.{jls,txt})
#
#  (a) the barrier on meshes 2-7: every symmetric star has r_inf(u2) + r_inf(d) >= |s_h(u2,d)| / (2 |u2|inf |d|inf)
#      (Theorem barrier (b) with |Omega| = 1), against the measured errors of H_DEC, H_P1, the symmetric [P2]^2 star on
#      the 4-ring (all symmetric) and H_P2 (not symmetric).  Errors from ts_data/paper_elliptic.jls and p2_symmetric.jls.
#  (b) the logic of Corollary size with k = m = 2: the lower bound c_S h^2/(2|u||v|) and the upper bound of Theorem order
#      (1+Λ)(ρh)^3/3! (|u|_3/|u| + |v|_3/|v|) for Λ = 1.35, ρ = 3; they cross at h*, below which Λ must grow.
# ============================================================================
include(joinpath(@__DIR__, "..", "stars", "three_stars.jl"))
using CairoMakie
empty!(lines)
const MS = ("mesh2", "mesh3", "mesh4", "mesh5", "mesh6", "mesh7")
const NU, ND = 1.0, supn.d                       # sup norms of u2 and d

f_skew = joinpath(TSDIR, "barrier_skew.jls")
SK = isfile(f_skew) ? deserialize(f_skew) : Dict{String,Any}()
for mn in MS
    haskey(SK, mn) && continue
    mesh = VoronoiMesh("meshes/$mn.nc"); h = meshsize(mesh)
    α(v) = reduce_primal_edges(mesh, v).values; β(v) = reduce_dual_edges(mesh, star_field(v)).values
    s = dot(α(U2), β(fields.d)) - dot(α(fields.d), β(U2))
    SK[mn] = (h = h, s = s)
    serialize(f_skew, SK)
end
logp("skew of (u2, d) and the barrier for symmetric stars: r_inf(u2) + r_inf(d) >= |s_h| / (2 |u2| |d|),  |d|inf = $(ND)")
RE = deserialize(joinpath(TSDIR, "paper_elliptic.jls")); RP = deserialize(joinpath(TSDIR, "p2_symmetric.jls"))
err(mn, s) = RE[(mn, s)][:hu2].inf + RE[(mn, s)][:hd].inf
for mn in MS
    h, s = SK[mn].h, SK[mn].s; b = abs(s) / (2NU * ND)
    sym4 = haskey(RP, (mn, 4, :sym)) ? @sprintf("%.2e", RP[(mn, 4, :sym)].hinf) : "   --   "
    logp(@sprintf("%s  h = 1/%-4d  s_h = %+.3e = %+.4f h^2   barrier %.2e   H_DEC %.2e  H_P1 %.2e  sym P2 4-ring %s  H_P2 %.2e (ratio to barrier %.2f)",
                  mn, round(Int, 1 / h), s, s / h^2, b, err(mn, "H_DEC"), err(mn, "H_P1"), sym4, err(mn, "H_P2"), err(mn, "H_P2") / b))
end
# constants of Corollary size (Remark sizeconst): c_S = |s_h|/h^2 on mesh 6, |u2|_3 = 496, |d|_3 = 248
cS = abs(SK["mesh6"].s) / SK["mesh6"].h^2; U3, D3 = 496.0, 248.0; Λ0, ρ = 1.35, 3.0
lower(h) = cS * h^2 / (2NU * ND)
upper(h; Λ = Λ0) = (1 + Λ) * (ρ * h)^3 / 6 * (U3 / NU + D3 / ND)
hstar = cS / (2NU * ND) / ((1 + Λ0) * ρ^3 / 6 * (U3 / NU + D3 / ND))
c = cS * 6 / (2 * (ND * U3 + NU * D3))
logp(@sprintf("Corollary size: c_S = %.3f, c = %.2e; with Λ = %.2f, ρ = %.0f the two bounds cross at h* = %.2e (%.1e cells)", cS, c, Λ0, ρ, hstar, 1 / hstar^2))
write(joinpath(TSDIR, "barrier_skew.txt"), join(lines, "\n") * "\n")

# ---------------------------------------------------------------- figure
f = Figure(size = (1450, 600))
hs = [SK[m].h for m in MS]; xt = (hs, ["1/8", "1/16", "1/32", "1/64", "1/128", "1/256"])
ax = Axis(f[1, 1]; xscale = log10, yscale = log10, xreversed = true, xticks = xt, xlabel = "h", titlesize = 14,
          ylabel = "r∞(u₂) + r∞(d)", title = "(a) the barrier on SCVT meshes 2–7",
          yticks = (10.0 .^ (-7:0), ["10⁻⁷", "10⁻⁶", "10⁻⁵", "10⁻⁴", "10⁻³", "10⁻²", "10⁻¹", "1"]))
bar = [abs(SK[m].s) / (2NU * ND) for m in MS]
band!(ax, hs, fill(1e-8, length(hs)), bar; color = (:gray, 0.25))
scatterlines!(ax, hs, bar; color = :black, linewidth = 2.5, marker = :hline, markersize = 14)
text!(ax, hs[2], bar[2] / 6; text = "forbidden for every\nsymmetric star", fontsize = 13, align = (:left, :top))
styles = [("H_DEC", "H_DEC (symmetric)", colorant"#7a7a7a", :circle, :white),
          ("H_P1", "H_P1 (symmetric, exact on [P1]²)", colorant"#2b6cb0", :xcross, colorant"#2b6cb0"),
          ("H_P2", "H_P2 (not symmetric, exact on [P2]²)", colorant"#c23b3a", :rect, colorant"#c23b3a")]
for (s, lab, col, mk, mc) in styles
    scatterlines!(ax, hs, [err(m, s) for m in MS]; color = col, marker = mk, markersize = 11, markercolor = mc, strokecolor = col,
                  strokewidth = 1.5, linewidth = 2, label = lab)
end
m4 = [m for m in MS if haskey(RP, (m, 4, :sym))]
scatterlines!(ax, [SK[m].h for m in m4], [RP[(m, 4, :sym)].hinf for m in m4]; color = colorant"#2f8f4e", linestyle = :dash,
              marker = :utriangle, markersize = 11, markercolor = :white, strokecolor = colorant"#2f8f4e", strokewidth = 1.5, linewidth = 2,
              label = "symmetric, exact on [P2]², 4-ring")
lines!(ax, [NaN], [NaN]; color = :black, linewidth = 2.5, label = "barrier |s_h(u₂,d)| / (2‖u₂‖∞‖d‖∞)")
ylims!(ax, 1e-7, 1.0)
axislegend(ax; position = :rt, framevisible = false, labelsize = 12)

#   (b) The error of a symmetric star exact on [P2]^2 must lie ABOVE the black line (barrier) and BELOW the red line (worst-case
#   bound of exactness and order).  Where the red line is above the black one there is room (green band) and the corollary says
#   nothing; where it is below, no such star exists with that Λ.  h decreases to the right, as in (a).
pow10(k) = "10" * join(Dict('-' => '⁻', '0' => '⁰', '1' => '¹', '2' => '²', '3' => '³', '4' => '⁴', '5' => '⁵', '6' => '⁶', '7' => '⁷', '8' => '⁸', '9' => '⁹')[ch] for ch in string(k))
ax2 = Axis(f[1, 2]; xscale = log10, yscale = log10, xreversed = true, xlabel = "h   (finer meshes to the right →)", ylabel = "r∞(u₂) + r∞(d)",
           titlesize = 14, title = "(b) when the corollary excludes a symmetric star exact on [P2]² (k = m = 2)",
           xticks = (10.0 .^ (0:-1:-8), [k == 0 ? "1" : pow10(k) for k in 0:-1:-8]),
           yticks = (10.0 .^ (-20:4:4), [k == 0 ? "1" : pow10(k) for k in -20:4:4]))
hh = 10.0 .^ range(-0.5, -8; length = 300)
h100 = hstar * (1 + Λ0) / (1 + 100.0)                       # where the Λ = 100 bound crosses the barrier
vspan!(ax2, 1 / 4, 1 / 256; color = (:gray, 0.18))
vspan!(ax2, hstar, 1e-8; color = (colorant"#c23b3a", 0.07))
text!(ax2, sqrt(1 / 4 * 1 / 256), 3e3; text = "meshes 1–7", fontsize = 13, align = (:center, :center))
text!(ax2, 10^-4.6, 1e1; text = "room between the bounds:\nthe corollary says nothing here", fontsize = 13, align = (:center, :center))
text!(ax2, 10^-7.0, 1e1; text = @sprintf("excluded for Λ = %.2f:\nthe upper bound is below\nthe lower bound", Λ0), fontsize = 13,
      align = (:center, :center), color = colorant"#8a1f1f")
hb = filter(>(hstar), hh)
band!(ax2, hb, lower.(hb), upper.(hb); color = (colorant"#2f8f4e", 0.22),
      label = @sprintf("where the error of such a star with Λ = %.2f must lie", Λ0))
lines!(ax2, hh, lower.(hh); color = :black, linewidth = 2.5, label = "lower bound for every symmetric star (barrier): c_S h² / (2‖u₂‖∞‖d‖∞)")
lines!(ax2, hh, upper.(hh); color = colorant"#c23b3a", linewidth = 2.5,
       label = @sprintf("upper bound for a star exact on [P2]² (exactness and order), Λ = %.2f, ρ = 3", Λ0))
lines!(ax2, hh, upper.(hh; Λ = 100.0); color = colorant"#c23b3a", linewidth = 1.5, linestyle = :dash, label = "the same upper bound with Λ = 100")
scatter!(ax2, [hstar], [lower(hstar)]; color = :black, markersize = 12)
text!(ax2, hstar, lower(hstar); text = "h* = 4·10⁻⁶  ", fontsize = 13, align = (:right, :top))
scatter!(ax2, [h100], [lower(h100)]; color = :white, strokecolor = :black, strokewidth = 1.5, markersize = 11)
text!(ax2, h100, lower(h100); text = "Λ = 100: excluded\nbelow h ≈ 10⁻⁷  ", fontsize = 12, align = (:right, :top))
ylims!(ax2, 1e-20, 1e5)
axislegend(ax2; position = :lb, framevisible = false, labelsize = 12)
logp(@sprintf("panel (b): Λ = 100 bound crosses the barrier at h = %.2e", h100))
CairoMakie.save(joinpath(@__DIR__, "..", "..", "output", "figures", "fig_gap_barrier.png"), f; px_per_unit = 2)
