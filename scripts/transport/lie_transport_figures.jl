# ============================================================================
#  lie_transport_figures.jl -- figures of lie_transport_check.jl
#
#      julia scripts/transport/lie_transport_figures.jl       ->  output/figures/fig_lie_convergence.png, output/figures/fig_lie_spectra.png
# ============================================================================
using Serialization, CairoMakie, Printf
const TSD = joinpath(@__DIR__, "..", "..", "output", "ts_data")
R = deserialize(joinpath(TSD, "lie_transport.jls"))
M = [m for m in ("mesh2", "mesh3", "mesh4", "mesh5", "mesh6", "mesh7") if haskey(R, (m, :h))]
h = [R[(m, :h)] for m in M]
BLUE, RED, GREEN, GRAY, ORANGE, PURPLE = colorant"#2b6cb0", colorant"#c23b3a", colorant"#2f8f4e", colorant"#6b6b6b", colorant"#d08a1c", colorant"#7b4fa0"
sup(k) = Dict(0 => "⁰", 1 => "¹", 2 => "²", 3 => "³", 4 => "⁴")[k]
hticks = ([1/4, 1/8, 1/16, 1/32, 1/64, 1/128, 1/256], ["1/4", "1/8", "1/16", "1/32", "1/64", "1/128", "1/256"])
function slope!(ax, k, x0, y0; x1 = x0 / 8)
    xx = [x0, x1]; lines!(ax, xx, y0 .* (xx ./ x0) .^ k; color = :black, linestyle = :dot)
    text!(ax, x1, y0 * (x1 / x0)^k; text = " h" * sup(k), fontsize = 13, align = (:left, :center))
end
logax(pos, title, ylabel) = Axis(pos; xscale = log10, yscale = log10, xreversed = true, xticks = hticks, xlabel = "h", ylabel = ylabel,
                                 title = title, titlesize = 14)

f = Figure(size = (1600, 520), fontsize = 13)
# (a) free stream
ax = logax(f[1, 1], "(a) free stream: max |D̃ H v / A|  (ρ ≡ 1, div u = 0)", "max error")
for (s, c, mk) in (("H_DEC", RED, :circle), ("H_P1", ORANGE, :rect), ("H_P2", BLUE, :utriangle))
    scatterlines!(ax, h, [R[(m, s, :free)].inf for m in M]; color = c, marker = mk, label = s)
end
slope!(ax, 1, 0.1, 2e-2); slope!(ax, 2, 0.1, 2e-3)
axislegend(ax; position = :lb, framevisible = false)
# (b) truncation of the density Lie derivative and (c) transport error
sel = (("H_DEC", "TRiSK", "H_DEC  (= DEC = TRiSK)", RED, :circle, :solid),
       ("H_P2", "TRiSK", "H_P2  TRiSK  (ρ̄ₑ avg ⊙ Hv)", PURPLE, :diamond, :dash),
       ("H_P2", "DEC", "H_P2  DEC  (H(ρ∧v))", PURPLE, :diamond, :solid),
       ("H_P2", "Peixoto", "H_P2  Peixoto  (barycentric ρₑ)", GREEN, :star5, :solid),
       ("H_DEC", "CFV2", "H_DEC  CFV2", RED, :xcross, :dash),
       ("H_P1", "CFV2", "H_P1  CFV2", ORANGE, :rect, :solid),
       ("H_P2", "CFV2", "H_P2  CFV2", BLUE, :utriangle, :solid),
       ("exact", "CFV2", "exact flux  CFV2", GRAY, :hexagon, :dot))
axb = logax(f[1, 2], "(b) truncation of the Lie derivative of ρ̃ (max)", "max |τ| / max|div(ρu)|")
axc = logax(f[1, 3], "(c) transport by the deformational flow, error at T = 1 (max)", "max |ρ̄(T) − ρ̄(0)|")
for (s, v, lab, c, mk, ls) in sel
    scatterlines!(axb, h, [R[(m, s, v, :trunc)].inf for m in M]; color = c, marker = mk, linestyle = ls, label = lab)
    scatterlines!(axc, h, [R[(m, s, v, :transport)].inf for m in M]; color = c, marker = mk, linestyle = ls, label = lab)
end
slope!(axb, 1, 0.03, 1.5e-3); slope!(axb, 2, 0.03, 2e-4)
slope!(axc, 2, 0.03, 5e-2); slope!(axc, 3, 0.03, 4e-3)
Legend(f[1, 4], axb; framevisible = false, labelsize = 12)
CairoMakie.save(joinpath(@__DIR__, "..", "..", "output", "figures", "fig_lie_convergence.png"), f; px_per_unit = 2)

# spectra (if computed)
if haskey(R, ("mesh4", "H_DEC", "TRiSK", :spec))
    g = Figure(size = (1500, 480), fontsize = 13)
    for (j, (s, v, lab)) in enumerate((("H_DEC", "TRiSK", "H_DEC  (= DEC = TRiSK)"), ("H_P2", "DEC", "H_P2  DEC"),
                                       ("H_P2", "Ham", "H_P2  Hamiltonian"), ("H_P2", "CFV2", "H_P2  CFV2")))
        λ = R[("mesh4", s, v, :spec)]
        ax = Axis(g[1, j]; xlabel = "Re λ", ylabel = j == 1 ? "Im λ" : "", title = @sprintf("%s:  max Re λ = %+.2e", lab, maximum(real.(λ))),
                  titlesize = 13)
        scatter!(ax, real.(λ), imag.(λ); markersize = 3, color = BLUE)
        vlines!(ax, [0]; color = :black, linewidth = 0.5)
    end
    Label(g[0, 1:4], "spectrum of −L for the steady swirl on mesh4 (1024 cells); the continuous operator is skew (Re λ = 0)")
    CairoMakie.save(joinpath(@__DIR__, "..", "..", "output", "figures", "fig_lie_spectra.png"), g; px_per_unit = 2)
end
