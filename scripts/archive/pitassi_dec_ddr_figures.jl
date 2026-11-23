# ============================================================================
#  pitassi_dec_ddr_figures.jl -- new figures of pitassi_dec_ddr.tex
#
#      julia scripts/archive/pitassi_dec_ddr_figures.jl
#
#  fig_pdd_sigma.png       stabilisation coefficient σ_T of the DEC, Whitney and DDR stars
#                          against the shape of the triangle, and how DEC splits the
#                          energy of the rotation mode (Part I)
#  fig_pdd_convergence.png DDR mixed Poisson, k = 0 and k = 1 (Part III)
#  fig_pdd_dual.png        the dual structure on a computed k = 1 solution (Part III)
# ============================================================================
using LinearAlgebra, Printf, CairoMakie
include(joinpath(@__DIR__, "ddr_mixed_poisson.jl"))           # solver (its main block does not run when included)

const INK = colorant"#2b2b29"; const MUTED = colorant"#8a8a85"; const GRIDC = colorant"#e6e5e1"
const BLUE = colorant"#2a78d6"; const ORANGE = colorant"#eb6834"; const AQUA = colorant"#1baf7a"; const RED = colorant"#c23b3a"
set_theme!(Theme(fontsize = 13, textcolor = INK,
    Axis = (xgridcolor = GRIDC, ygridcolor = GRIDC, spinewidth = 0.8, leftspinecolor = MUTED,
            bottomspinecolor = MUTED, topspinevisible = false, rightspinevisible = false,
            xtickcolor = MUTED, ytickcolor = MUTED, titlefont = :regular, titlesize = 13),
    Legend = (framevisible = false, labelsize = 11)))

# ---------------------------------------------------------------- Part I: σ against the shape
circ(a, b, c) = (d = 2cross2(b - a, c - a); a + J * (dot(c - a, c - a) * (b - a) - dot(b - a, b - a) * (c - a)) / d)
function sigmas(V)
    T = cross2(V[2] - V[1], V[3] - V[1]) / 2
    E = [(V[1], V[2]), (V[2], V[3]), (V[3], V[1])]
    θ = [atan(cross2(V[mod1(i + 1, 3)] - V[i], V[mod1(i + 2, 3)] - V[i]), dot(V[mod1(i + 1, 3)] - V[i], V[mod1(i + 2, 3)] - V[i])) for i in 1:3]
    σdec = 1 / (2sum(tan.(θ)))
    σw = sum(norm(a - b)^2 for (a, b) in E) / (144T)
    h = maximum(norm(a - b) for (a, b) in E); σddr = h / 9 * sum(1 / norm(a - b) for (a, b) in E)
    x̄ = sum(V) / 3; c = circ(V...)
    Rc1 = sum(J * ((a + b) / 2 - c) for (a, b) in E) / T                    # R_c applied to the rotation mode 1
    return σdec, σw, σddr, T * dot(Rc1, Rc1), 9σdec, sum(cot.(θ)) / 2
end
apex = range(35, 160, length = 600)
vals = [sigmas([[0.0, 0.0], [1.0, 0.0], [0.5, 0.5 / tand(φ / 2)]]) for φ in apex]
fig = Figure(size = (1100, 420))
ax1 = Axis(fig[1, 1], xlabel = "apex angle of the isosceles triangle (degrees)", ylabel = "σ_T",
           title = "(a) the stabilisation coefficient σ_T of three stars")
hlines!(ax1, [0.0], color = MUTED, linewidth = 0.8); vlines!(ax1, [60, 90], color = GRIDC, linewidth = 1.2, linestyle = :dash)
lines!(ax1, apex, first.(vals), color = ORANGE, linewidth = 2.5, label = "DEC diagonal: 1/(2 Σ tan θ)")
lines!(ax1, apex, [v[2] for v in vals], color = BLUE, linewidth = 2.5, label = "Whitney (Galerkin): Σ|E|²/(144|T|)")
lines!(ax1, apex, [v[3] for v in vals], color = AQUA, linewidth = 2.5, label = "DDR k = 0: (h_T/9) Σ 1/|E|")
text!(ax1, 61, -0.17, text = "equilateral", color = MUTED, fontsize = 11); text!(ax1, 91, -0.17, text = "right", color = MUTED, fontsize = 11)
ylims!(ax1, -0.3, 0.75); axislegend(ax1, position = :rt)
ax2 = Axis(fig[1, 2], xlabel = "apex angle of the isosceles triangle (degrees)", ylabel = "energy of the rotation mode α = (1,1,1)",
           title = "(b) DEC splits the rotation energy: node part + stabilisation")
hlines!(ax2, [0.0], color = MUTED, linewidth = 0.8); vlines!(ax2, [60, 90], color = GRIDC, linewidth = 1.2, linestyle = :dash)
lines!(ax2, apex, [v[6] for v in vals], color = INK, linewidth = 2.5, label = "total αᵀHα = ½ Σ cot θ")
lines!(ax2, apex, [v[4] for v in vals], color = BLUE, linewidth = 2, linestyle = :dash, label = "node part |T| |R_c α|²")
lines!(ax2, apex, [v[5] for v in vals], color = ORANGE, linewidth = 2, linestyle = :dot, label = "stabilisation 9 σ_T")
ylims!(ax2, -1.5, 3.0); axislegend(ax2, position = :lt)
save(joinpath(@__DIR__, "..", "..", "output", "figures", "fig_pdd_sigma.png"), fig, px_per_unit = 2)
println("wrote fig_pdd_sigma.png")

# ---------------------------------------------------------------- Part III: convergence
data = [split(l) for l in eachline(joinpath(@__DIR__, "..", "..", "output", "figures", "ddr_mixed_data.txt")) if !startswith(l, "#")]
fig = Figure(size = (1100, 420))
for (col, (lbl, idx)) in enumerate((("(a) flux: discrete L²-product norm of σ_h − I σ", 5), ("(b) pressure: ‖u_h − π u‖ (L²)", 7)))
    ax = Axis(fig[1, col], xscale = log10, yscale = log10, xlabel = "h", ylabel = "relative error", title = lbl, xreversed = true)
    for (kind, mk) in (("tri", :circle), ("quad", :rect)), (k, colr) in (("0", BLUE), ("1", ORANGE))
        rows = [d for d in data if d[1] == kind && d[2] == k]
        h = parse.(Float64, getindex.(rows, 4)); e = parse.(Float64, getindex.(rows, idx))
        scatterlines!(ax, h, e, color = colr, marker = mk, markersize = 9, linewidth = 2,
                      label = "k = $k, $(kind == "tri" ? "triangles" : "quadrilaterals")")
    end
    hh = [1 / 4, 1 / 64]
    for (p, y0) in (idx == 5 ? ((1, 0.02), (2, 0.06)) : ((2, 0.4), (3, 0.06)))
        lines!(ax, hh, y0 .* (hh ./ hh[1]) .^ p, color = MUTED, linewidth = 1, linestyle = :dash)
        text!(ax, hh[2], y0 * (hh[2] / hh[1])^p, text = "h^$p ", color = MUTED, fontsize = 11, align = (:right, :bottom))
    end
    col == 1 && axislegend(ax, position = :lb)
end
save(joinpath(@__DIR__, "..", "..", "output", "figures", "fig_pdd_convergence.png"), fig, px_per_unit = 2)
println("wrote fig_pdd_convergence.png")

# ---------------------------------------------------------------- Part III: the dual structure on a computed solution
let M = gridmesh(4; kind = :tri, pert = 0.2, seed = 7), k = 1
    r = solve(M, k, uex, σex, fex)
    fig = Figure(size = (1100, 520))
    ax1 = Axis(fig[1, 1], aspect = DataAspect(), title = "(a) k = 1 solution, N = 4: dual pieces x̄ → q± carrying the half-edge fluxes")
    ax2 = Axis(fig[1, 2], aspect = DataAspect(), title = "(b) one cell: σ_G = R_flux − |T|⁻¹∫ f (x − x̄)")
    hidedecorations!(ax1); hidespines!(ax1); hidedecorations!(ax2); hidespines!(ax2)
    φmax = 0.0; pieces = []
    zoomc = 9
    for c in eachindex(M.cells)
        L = r.locs[c]; C = L.C; x = r.σh[r.gσ(c, L)]; n = length(L.E)
        Rf = zeros(2)
        for i in 1:n
            a, b = L.E[i]; ω = L.ω[i]; s = x[L.eslot(i)]; len = norm(b - a)
            Φ = ω * s[1] * len; μ = s[2] * len / 12; τl = ω * (b - a) / len; ml = (a + b) / 2
            φm = (Φ - 3μ) / 2; φp = (Φ + 3μ) / 2; qm = ml - len / 3 * τl; qp = ml + len / 3 * τl
            push!(pieces, (c, C.x̄, qm, φm)); push!(pieces, (c, C.x̄, qp, φp)); φmax = max(φmax, abs(φm), abs(φp))
            Rf += Φ * (ml - C.x̄) + len * μ * τl
        end
        Rf /= C.area
        for ax in (ax1, ax2)
            (ax === ax2 && c != zoomc) && continue
            poly!(ax, Point2f.(C.V), color = (ax === ax2 ? colorant"#f4f3ef" : :transparent), strokecolor = MUTED, strokewidth = 1)
        end
        if c == zoomc
            σG = x[1:2]; fmom = sum(w * fex(q) * (q - C.x̄) for (q, w) in zip(C.qx, C.qw)) / C.area
            sc = 0.045
            @printf("   zoomed cell: |T| = %.5f, σ_G = (%.4f, %.4f), R_flux = (%.4f, %.4f), |T|⁻¹∫f(x-x̄) = (%.4f, %.4f), -∇u_h = (%.4f, %.4f)\n",
                    C.area, σG..., Rf..., fmom..., (-r.uh[3(c-1)+2:3(c-1)+3] / C.h)...)
            arrows2d!(ax2, [Point2f(C.x̄)], [Vec2f(sc * Rf)], color = RED, shaftwidth = 2.5, tipwidth = 9)
            arrows2d!(ax2, [Point2f(C.x̄)], [Vec2f(sc * σG)], color = INK, shaftwidth = 2.5, tipwidth = 9)
            arrows2d!(ax2, [Point2f(C.x̄ + sc * Rf)], [Vec2f(-sc * fmom)], color = AQUA, shaftwidth = 1.5, tipwidth = 7)
        end
    end
    for (c, x̄, q, φ) in pieces
        col = φ > 0 ? ORANGE : BLUE; lw = 0.5 + 4.5abs(φ) / φmax
        lines!(ax1, [Point2f(x̄), Point2f(q)], color = col, linewidth = lw)
        scatter!(ax1, [Point2f(q)], color = col, markersize = 4)
        if c == zoomc
            lines!(ax2, [Point2f(x̄), Point2f(q)], color = col, linewidth = 1 + 6abs(φ) / φmax)
            scatter!(ax2, [Point2f(q)], color = col, markersize = 9)
            text!(ax2, Point2f(q), text = @sprintf(" %.3f", φ), color = col, fontsize = 11)
        end
    end
    scatter!(ax1, [Point2f(r.locs[c].C.x̄) for c in eachindex(M.cells)], color = INK, markersize = 5)
    scatter!(ax2, [Point2f(r.locs[zoomc].C.x̄)], color = INK, markersize = 10)
    Legend(fig[2, 1:2], [LineElement(color = ORANGE, linewidth = 3), LineElement(color = BLUE, linewidth = 3),
                         LineElement(color = INK, linewidth = 3), LineElement(color = RED, linewidth = 3), LineElement(color = AQUA, linewidth = 3)],
           ["outward half-edge flux > 0", "outward half-edge flux < 0", "σ_G (cell unknown)", "R_flux (dual pieces)", "−|T|⁻¹∫f(x−x̄)"],
           orientation = :horizontal, framevisible = false)
    save(joinpath(@__DIR__, "..", "..", "output", "figures", "fig_pdd_dual.png"), fig, px_per_unit = 2)
    println("wrote fig_pdd_dual.png (zoomed cell $zoomc)")
end
