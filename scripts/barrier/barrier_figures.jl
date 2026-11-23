# ============================================================================
#  barrier_figures.jl -- figure of barrier_paper.tex from ts_data/skew_split.jls (after skew_split_check.jl).
#
#      julia scripts/barrier/barrier_figures.jl
#
#  fig_barrier_skew.png   |s_h(u,v)| against h for gradient / rotated-gradient pairs on SCVT, lattice and perturbed meshes
# ============================================================================
include(joinpath(@__DIR__, "..", "stars", "three_stars.jl"))
using CairoMakie
const R = deserialize(joinpath(TSDIR, "skew_split.jls"))
const KINDS = ((:gg, "∇p, ∇q", colorant"#2b6cb0", :circle), (:cc, "J∇p, J∇q", colorant"#c23b3a", :rect),
               (:gc, "∇p, J∇q", colorant"#2f8f4e", :diamond), (:cg, "J∇p, ∇q", colorant"#d08a1c", :utriangle),
               (:ud, "u₂, d", colorant"#4d4d4d", :xcross))
let f = Figure(size = (1450, 500))
    for (i, (fam, title)) in enumerate((("SCVT", "(a) SCVT meshes 2–6"), ("lattice", "(b) lattices 2–6"), ("perturbed", "(c) perturbed lattices 1–6")))
        ks = sort([k for k in keys(R) if k[1] == fam], by = k -> -R[k].h)
        hs = [R[k].h for k in ks]
        ax = Axis(f[1, i]; xscale = log10, yscale = log10, xreversed = true, xlabel = "h", ylabel = "|s_h(u, v)|", title = title,
                  titlesize = 14, xticks = (hs, [@sprintf("%.3f", h) for h in hs]), xticklabelrotation = π / 5,
                  yticks = (10.0 .^ (-16:2:0), ["10⁻¹⁶", "10⁻¹⁴", "10⁻¹²", "10⁻¹⁰", "10⁻⁸", "10⁻⁶", "10⁻⁴", "10⁻²", "1"]))
        for (fld, lab, col, mk) in KINDS
            scatterlines!(ax, hs, [max(abs(getfield(R[k], fld)), 1e-16) for k in ks]; color = col, marker = mk, markersize = 10, label = lab)
        end
        y0 = fam == "SCVT" ? 2e-3 : fam == "lattice" ? 1e-12 : 3e-3
        lines!(ax, [hs[2], hs[end]], y0 .* ([hs[2], hs[end]] ./ hs[2]) .^ 2; color = :black, linestyle = :dot)
        text!(ax, hs[end], y0 * (hs[end] / hs[2])^2; text = "  h²", fontsize = 13, align = (:left, :center))
    end
    Legend(f[2, 1:3], [[LineElement(color = c), MarkerElement(color = c, marker = m, markersize = 10)] for (_, _, c, m) in KINDS],
           ["s_h(" * l * ")" for (_, l, _, _) in KINDS]; orientation = :horizontal, framevisible = false, labelsize = 13)
    CairoMakie.save(joinpath(@__DIR__, "..", "..", "output", "figures", "fig_barrier_skew.png"), f; px_per_unit = 2)
end

# ---------------------------------------------------------------- the offset of a Voronoi edge, and the per-triangle identity
"circumcentre of the triangle (a, b, c)"
function circumcentre(a, b, c)
    d = 2 * (a[1] * (b[2] - c[2]) + b[1] * (c[2] - a[2]) + c[1] * (a[2] - b[2]))
    ux = (sum(abs2, a) * (b[2] - c[2]) + sum(abs2, b) * (c[2] - a[2]) + sum(abs2, c) * (a[2] - b[2])) / d
    uy = (sum(abs2, a) * (c[1] - b[1]) + sum(abs2, b) * (a[1] - c[1]) + sum(abs2, c) * (b[1] - a[1])) / d
    return [ux, uy]
end
let f = Figure(size = (1050, 470))
    # (a) one edge e = ab, its two triangles, the Voronoi edge c1c2 and the offset of its midpoint
    a, b, p1, p2 = [0.0, 0.0], [1.0, 0.0], [0.30, -0.62], [0.72, 0.78]
    c1, c2 = circumcentre(a, b, p1), circumcentre(a, b, p2)
    me = (a + b) / 2; mt = (c1 + c2) / 2
    ax = Axis(f[1, 1]; aspect = DataAspect(), title = "(a) the Voronoi edge ẽ is not centred on e", titlesize = 19); hidedecorations!(ax); hidespines!(ax)
    for (p, q) in ((a, p1), (b, p1), (a, p2), (b, p2)); lines!(ax, [p[1], q[1]], [p[2], q[2]]; color = :gray, linewidth = 1.5); end
    lines!(ax, [a[1], b[1]], [a[2], b[2]]; color = :black, linewidth = 3)
    lines!(ax, [c1[1], c2[1]], [c1[2], c2[2]]; color = colorant"#c23b3a", linewidth = 3, linestyle = :dash)
    scatter!(ax, [me[1], mt[1]], [me[2], mt[2]]; color = [:black, colorant"#c23b3a"], markersize = 12)
    scatter!(ax, [c1[1], c2[1]], [c1[2], c2[2]]; color = colorant"#c23b3a", marker = :diamond, markersize = 12)
    text!(ax, me[1], me[2]; text = "  m_e", align = (:left, :top), fontsize = 19)
    text!(ax, mt[1], mt[2]; text = "  m̃_e = m_e + δ_e n_e", align = (:left, :center), fontsize = 19, color = colorant"#c23b3a")
    text!(ax, c1[1], c1[2]; text = "  c₁ (circumcentre of T₁)", align = (:left, :center), fontsize = 17, color = colorant"#c23b3a")
    text!(ax, c2[1], c2[2]; text = "  c₂ (circumcentre of T₂)", align = (:left, :center), fontsize = 17, color = colorant"#c23b3a")
    text!(ax, 0.22, -0.02; text = "e", fontsize = 21, align = (:center, :top)); text!(ax, 0.33, -0.36; text = "T₁", fontsize = 19); text!(ax, 0.6, 0.35; text = "T₂", fontsize = 19)
    arrows2d!(ax, [Point2f(0.15, 0.0)], [Vec2f(0.25, 0.0)]; color = :black, shaftwidth = 2, tipwidth = 10, tiplength = 10)
    arrows2d!(ax, [Point2f(0.15, 0.0)], [Vec2f(0.0, 0.25)]; color = :black, shaftwidth = 2, tipwidth = 10, tiplength = 10)
    text!(ax, 0.40, 0.0; text = "t_e", align = (:left, :top), fontsize = 18); text!(ax, 0.15, 0.25; text = "n_e", align = (:right, :bottom), fontsize = 18)
    xlims!(ax, -0.25, 1.85); ylims!(ax, -0.75, 0.95)
    # (b) the right triangle (0,0), (2,0), (0,1): 3|T|(xbar - c) = -1/4 Σ |E|^3 ν_E
    A, B, C = [0.0, 0.0], [2.0, 0.0], [0.0, 1.0]; cc = circumcentre(A, B, C); xb = (A + B + C) / 3
    ax = Axis(f[1, 2]; aspect = DataAspect(), title = "(b) 3|T| (x̄_T − c_T) = −¼ Σ_E |E|³ ν_E  on one triangle", titlesize = 19)
    hidedecorations!(ax); hidespines!(ax)
    poly!(ax, [Point2f(A...), Point2f(B...), Point2f(C...)]; color = (:gray, 0.12), strokecolor = :black, strokewidth = 2)
    scatter!(ax, [cc[1]], [cc[2]]; color = colorant"#c23b3a", marker = :diamond, markersize = 13)
    scatter!(ax, [xb[1]], [xb[2]]; color = colorant"#2b6cb0", markersize = 12)
    text!(ax, cc[1], cc[2]; text = "  c_T = (1, ½)", align = (:left, :bottom), fontsize = 18, color = colorant"#c23b3a")
    text!(ax, xb[1], xb[2]; text = "x̄_T = (⅔, ⅓)  ", align = (:right, :top), fontsize = 18, color = colorant"#2b6cb0")
    for (p, q, lab) in ((A, B, "|E|³ν = (0, −8)"), (B, C, "|E|³ν = (5, 10)"), (C, A, "|E|³ν = (−1, 0)"))
        τ = q - p; ν = [τ[2], -τ[1]] / norm(τ); m = (p + q) / 2
        arrows2d!(ax, [Point2f(m...)], [Vec2f((0.35 .* ν)...)]; color = colorant"#2f8f4e", shaftwidth = 2, tipwidth = 10, tiplength = 10)
        text!(ax, (m + 0.42 .* ν)...; text = lab, fontsize = 17, color = colorant"#2f8f4e",
              align = (ν[1] > 0.1 ? :left : ν[1] < -0.1 ? :right : :center, ν[2] > 0.1 ? :bottom : ν[2] < -0.1 ? :top : :center))
    end
    text!(ax, 0.65, -0.95; text = "Σ |E|³ν_E = (4, 2),  −¼ (4, 2) = (−1, −½) = 3·1·(−⅓, −⅙)", fontsize = 17, align = (:center, :center))
    xlims!(ax, -1.7, 3.0); ylims!(ax, -1.15, 1.75)
    CairoMakie.save(joinpath(@__DIR__, "..", "..", "output", "figures", "fig_barrier_offset.png"), f; px_per_unit = 2)
end

# ---------------------------------------------------------------- locality: the interior term, and how far one moved generator reaches
let L = deserialize(joinpath(TSDIR, "locality.jls")), LS = deserialize(joinpath(TSDIR, "locality_subspace.jls")),
    CS = deserialize(joinpath(TSDIR, "conformal_star.jls")), f = Figure(size = (1450, 560))
    # (a) dQ/dx against the prediction dD/dx, mesh 5, ρ = 0.3, both directions
    pairsA = [k for k in keys(L) if k[1] == "mesh5" && k[2] == :A && k[3] == 0.3]
    sort!(pairsA, by = k -> k[4] * k[5])
    labs = [replace(k[4], r" \(.*\)" => "") * " · " * replace(k[5], r" \(.*\)" => "") for k in pairsA]
    ax = Axis(f[1, 1]; title = "(a) moving one interior generator: dQ(P)/dx and the prediction dD(P)/dx", titlesize = 14,
              xticks = (1:length(pairsA), labs), xticklabelrotation = π / 4, ylabel = "|derivative| / h³  (mesh 5, ρ = 0.3)",
              yscale = log10, yticks = (10.0 .^ (-16:2:-2), ["10⁻¹⁶", "10⁻¹⁴", "10⁻¹²", "10⁻¹⁰", "10⁻⁸", "10⁻⁶", "10⁻⁴", "10⁻²"]))
    qv = [max(abs(L[k].qx), 1e-17) for k in pairsA]; dv = [max(abs(L[k].dx), 1e-17) for k in pairsA]
    barplot!(ax, (1:length(pairsA)) .- 0.18, qv; width = 0.34, color = colorant"#2b6cb0", label = "dQ/dx (computed)", fillto = 1e-17)
    barplot!(ax, (1:length(pairsA)) .+ 0.18, dv; width = 0.34, color = colorant"#d08a1c", label = "dD/dx (predicted)", fillto = 1e-17)
    ylims!(ax, 1e-17, 1e-1)
    axislegend(ax; position = :lt, framevisible = false, labelsize = 12)
    # (b) reach of one moved generator
    ax = Axis(f[1, 2]; xscale = log10, yscale = log10, xlabel = "distance r / h of the row edge from the moved generator",
              ylabel = "max |∂H_ji/∂x| · h", title = "(b) how far one moved generator changes the star (mesh 5)", titlesize = 14,
              xticks = ([1, 2, 5, 10, 20, 40], ["1", "2", "5", "10", "20", "40"]),
              yticks = (10.0 .^ (-8:2:0), ["10⁻⁸", "10⁻⁶", "10⁻⁴", "10⁻²", "1"]))
    r = collect(0.5:1:39.5)
    curves = (("H_P1 (symmetric, exact on [P₁]², global)", L[("mesh5", :B, "H_P1")].bins, colorant"#2b6cb0", :diamond),
              ("symmetric, exact on constants + conformal (smallest correction)", LS[("mesh5", "constants + conformal")].bins, colorant"#7a7a7a", :circle),
              ("symmetric, exact on constants + shears (smallest correction)", LS[("mesh5", "constants + shears")].bins, colorant"#9a6fb0", :utriangle),
              ("H_conf (explicit, local)", CS[:reach], colorant"#2f8f4e", :rect),
              ("H_P2 (row by row)", L[("mesh5", :B, "H_P2")].bins, colorant"#c23b3a", :xcross))
    for (lab, b, col, mk) in curves
        keep = b .> 0
        scatterlines!(ax, r[keep], b[keep]; color = col, marker = mk, markersize = 9, label = lab)
    end
    lines!(ax, [4.0, 40.0], 1e-3 .* ([4.0, 40.0] ./ 4) .^ -3; color = :black, linestyle = :dot)
    text!(ax, 40.0, 1e-3 * 10.0^-3; text = "  r⁻³", fontsize = 13, align = (:left, :center))
    xlims!(ax, 0.4, 60); ylims!(ax, 1e-9, 5)
    Legend(f[2, 1:2], ax; orientation = :horizontal, nbanks = 2, framevisible = false, labelsize = 12)
    CairoMakie.save(joinpath(@__DIR__, "..", "..", "output", "figures", "fig_barrier_locality.png"), f; px_per_unit = 2)
end
