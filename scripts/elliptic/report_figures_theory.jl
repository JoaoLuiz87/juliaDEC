# report_figures_theory.jl -- figures of hodge_report.tex, theory part:
#   fig_rep_gauss.png   the balance ("Gauss law") behind the non-locality theorem: hexagonal lattice with one displaced generator; SCVT mesh 4, change of the crossing sum
#   fig_rep_growth.png  smallest possible Λ of symmetric stars exact on [P1]² and [P2]² on SCVT meshes (certified), and H_P1, H_P2, least-change stars
#   fig_rep_budget.png  the budget: the error allowed by exactness against the skew
#   julia -t 2 scripts/elliptic/report_figures_theory.jl
include(joinpath(@__DIR__, "..", "stars", "stars_cache.jl")); include(joinpath(@__DIR__, "..", "shared", "thesis_plot_helpers.jl")); using Serialization
const AXS = axstyle_th
crg(a, b) = a[1] * b[2] - a[2] * b[1]
function clipv(poly, p, q)
    m = (p .+ q) ./ 2; n = q .- p; f(x) = dot(x .- m, n); out = Vector{Vector{Float64}}()
    for i in eachindex(poly)
        a, b = poly[i], poly[mod1(i + 1, length(poly))]; fa, fb = f(a), f(b)
        fa <= 0 && push!(out, a); (fa < 0) != (fb < 0) && push!(out, a .+ (fa / (fa - fb)) .* (b .- a))
    end
    return out
end
function vorc(X, i; R = 3.0)
    P = [X[i] .+ [R, R], X[i] .+ [-R, R], X[i] .+ [-R, -R], X[i] .+ [R, -R]]
    for j in eachindex(X); j == i || norm(X[j] .- X[i]) >= 2R || (P = clipv(P, X[i], X[j])); end
    return P
end
parea(P) = sum(crg(P[k], P[mod1(k + 1, length(P))]) for k in eachindex(P)) / 2
hexp(n) = [[i + j / 2, j * sqrt(3) / 2] for i in -2n:2n for j in -2n:2n if norm([i + j / 2, j * sqrt(3) / 2]) < n + 0.01]
γh(x) = (x[1]^2 - x[2]^2) / 2
let X0 = hexp(6), ε = 0.35
    i0 = findfirst(x -> norm(x) < 1e-9, X0); X1 = deepcopy(X0); X1[i0] = [ε, 0.0]; D, h, x0m = deserialize(joinpath(TSDIR, "green_charge_mesh.jls"))
    f = Figure(size = (1560, 1060), fontsize = 16)
    ax = Axis(f[1, 1]; aspect = DataAspect(), title = "(a) one generator of a lattice moves by ε", titlealign = :left); hidedecorations!(ax); hidespines!(ax)
    sh = findall(x -> norm(x) < 3.3, X0); ring = [i for i in sh if 0.5 < norm(X0[i]) < 1.5]
    dA = Dict(i => parea(vorc(X1, i)) - sqrt(3) / 2 for i in vcat(ring, i0)); m = maximum(abs, values(dA))
    for i in sh; poly!(ax, [Point2f(p...) for p in vorc(X1, i)]; color = fill(haskey(dA, i) ? dA[i] : 0.0, 1)[1], colormap = DIVM, colorrange = (-m, m), strokecolor = colorant"#b9b7b0", strokewidth = 1.2); end
    scatter!(ax, [Point2f(x...) for x in X1[sh]]; color = INK, markersize = 6); scatter!(ax, [Point2f(0, 0)]; color = :white, strokecolor = INK, strokewidth = 1.5, markersize = 10)
    arrows2d!(ax, [Point2f(0, 0)], [Vec2f(ε, 0)]; color = INK, shaftwidth = 2, tipwidth = 8, tiplength = 8)
    th = range(0, 2π; length = 200); lines!(ax, 2.5 .* cos.(th), 2.5 .* sin.(th); color = C_P1, linewidth = 2.5)
    text!(ax, 1.85, 1.9; text = "K", color = C_P1, fontsize = 22, font = :bold); text!(ax, 0.08, -0.48; text = "ε", fontsize = 20); xlims!(ax, -3.4, 3.4); ylims!(ax, -3.75, 3.2)
    text!(ax, -3.3, -3.65; text = "red: cell grew   blue: cell shrank", fontsize = 14, color = INK2)
    ax = Axis(f[1, 2]; title = "(b) the charge Q inside K changes", titlealign = :left, xlabel = "displacement ε  (edge length 1)", ylabel = "ΔQ = Δ Σ |cell| γ(x)", AXS...)
    ee = 0:0.01:0.9; lines!(ax, ee, sqrt(3) / 12 .* ee .^ 2 .* (1 .- ee .^ 2); color = INK, linewidth = 2)
    Kc = [i for i in eachindex(X0) if norm(X0[i]) < 3.2]
    Qof(e) = (X = deepcopy(X0); X[i0] = [e, 0.0]; sum(parea(vorc(X, i)) * γh(X[i]) for i in Kc)); q0 = Qof(0.0); es = 0.1:0.1:0.8
    scatter!(ax, collect(es), [Qof(e) - q0 for e in es]; color = C_P2, markersize = 11)
    text!(ax, 0.02, 0.0385; text = "curve: (√3/12) ε²(1 − ε²), by hand\ndots: computed from the Voronoi cells", fontsize = 14, color = INK2, align = (:left, :top)); ylims!(ax, -0.002, 0.041)
    ax = Axis(f[2, 1]; title = "(c) SCVT mesh 4, one generator moved by 0.15 h", titlealign = :left, xlabel = "radius of the circle / h", ylabel = "change of the crossing sum / h⁴", AXS...)
    rs = [3.0, 4.0, 6.0, 8.0, 10.0, 12.0]; q = D[("H_P1", 3.0)][6] - D[("H_P1", 3.0)][3]
    hlines!(ax, [q / h^4]; color = INK, linestyle = :dot, linewidth = 1.6); text!(ax, 11.9, q / h^4; text = "2ΔQ", align = (:right, :bottom), fontsize = 15)
    cP(nm) = [(D[(nm, r)][4] - D[(nm, r)][1]) / h^4 for r in rs]
    scatterlines!(ax, rs, cP("H_P1"); color = C_P1, marker = :diamond, markersize = 13, linewidth = 2, label = "H_P1 (symmetric)")
    scatterlines!(ax, rs, cP("H_P2"); color = C_P2, marker = :rect, markersize = 12, linewidth = 2, label = "H_P2 (not symmetric)")
    axislegend(ax; position = :rc, framevisible = false); ylims!(ax, 1.4 * min(0.0, minimum(cP("H_P2"))), 1.25q / h^4)
    ax = Axis(f[2, 2]; title = "(d) the same move: largest change in each row of H", titlealign = :left, yscale = log10, xlabel = "distance of the row from the generator / h", ylabel = "largest |ΔH_ej| in the row", AXS...)
    for (nm, col, mk) in (("H_P1", C_P1, :diamond), ("H_P2", C_P2, :rect))
        geo, rowch = D[(nm, :rows)]; r = [norm(g) / h for g in geo]; keep = rowch .> 1e-9
        scatter!(ax, r[keep], rowch[keep]; color = (col, 0.55), marker = mk, markersize = 6, label = nm)
    end
    rr = [2.0, 15.0]; lines!(ax, rr, 2.2e-3 .* (rr ./ 2) .^ (-3); color = INK, linestyle = :dash); text!(ax, 9.2, 3.2e-5; text = "r⁻³", fontsize = 16)
    axislegend(ax; position = :rt, framevisible = false); xlims!(ax, 0, 16.5)
    CairoMakie.save(thesis_out("fig_rep_gauss.png"), f; px_per_unit = 2)
end
let
    L2 = Dict("mesh2" => (1.161, 1.138), "mesh3" => (1.415, 1.336), "mesh4" => (2.399, 2.009))
    for l in eachline(joinpath(TSDIR, "min_lambda.txt"))
        mm = match(r"^(mesh\d)  k=(\d).*feasible ([\d.]+)   certified lower bound ([\d.]+)", l); mm !== nothing && mm[2] == "2" && (L2[mm[1]] = (parse(Float64, mm[3]), parse(Float64, mm[4])))
    end
    ms = sort(collect(keys(L2))); hh = Dict("mesh2" => 1 / 8, "mesh3" => 1 / 16, "mesh4" => 1 / 32, "mesh5" => 1 / 64, "mesh6" => 1 / 128, "mesh7" => 1 / 256)
    hs = [1 / 8, 1 / 16, 1 / 32, 1 / 64, 1 / 128, 1 / 256]; xt = (hs, ["1/8", "1/16", "1/32", "1/64", "1/128", "1/256"])
    f = Figure(size = (1560, 580), fontsize = 16)
    ax = Axis(f[1, 1]; xscale = log10, xreversed = true, yscale = log10, xticks = xt, yticks = ([1, 1.5, 2, 3, 5, 10, 15], string.([1, 1.5, 2, 3, 5, 10, 15])), xlabel = "h", ylabel = "Λ  (size of the coefficients)",
              title = "(a) exact on [P2]² and symmetric: the coefficients must grow", titlealign = :left, AXS...)
    band!(ax, [hh[m] for m in ms], [L2[m][2] for m in ms], [L2[m][1] for m in ms]; color = (VIOLET, 0.25))
    scatterlines!(ax, [hh[m] for m in ms], [L2[m][2] for m in ms]; color = VIOLET, marker = :utriangle, markersize = 13, linewidth = 2, label = "lower bound for every symmetric star (certified)")
    scatterlines!(ax, [hh[m] for m in ms], [L2[m][1] for m in ms]; color = VIOLET, marker = :dtriangle, markersize = 13, linewidth = 2, linestyle = :dash, label = "best symmetric star found")
    scatterlines!(ax, hs[1:4], [1.49, 2.45, 4.46, 15.0]; color = colorant"#1b1b1b", marker = :xcross, markersize = 12, linewidth = 1.5, linestyle = :dot, label = "least-change symmetric star")
    scatterlines!(ax, hs, [1.31, 1.35, 1.36, 1.34, 1.35, 1.36]; color = C_P2, marker = :rect, markersize = 11, linewidth = 2, label = "H_P2 (not symmetric)")
    hx = [1 / 16, 1 / 64]; lines!(ax, hx, 3.0 .* (hx[1] ./ hx) ./ 1.6; color = INK2, linestyle = :dash); text!(ax, 1 / 66, 7.4; text = "slope 1/h", fontsize = 15, color = INK2, align = (:left, :center))
    axislegend(ax; position = :lt, framevisible = false, labelsize = 13); ylims!(ax, 0.9, 28)
    ax = Axis(f[1, 2]; xscale = log10, xreversed = true, xticks = xt, xlabel = "h", ylabel = "Λ", title = "(b) exact on [P1]² and symmetric: no growth is needed", titlealign = :left, AXS...)
    scatterlines!(ax, hs[1:3], [1.030, 1.032, 1.035]; color = C_P1, marker = :utriangle, markersize = 13, linewidth = 2, linestyle = :dash, label = "smallest Λ of a symmetric star exact on [P1]² (3-ring)")
    scatterlines!(ax, hs, [1.127, 1.183, 1.204, 1.219, 1.252, 1.356]; color = C_P1, marker = :diamond, markersize = 13, linewidth = 2, label = "H_P1 (least change)")
    hlines!(ax, [1.0]; color = C_DEC, linestyle = :dot); text!(ax, 1 / 250, 1.005; text = "H_DEC", align = (:right, :bottom), fontsize = 14); ylims!(ax, 0.97, 1.62); axislegend(ax; position = :lt, framevisible = false, labelsize = 13)
    CairoMakie.save(thesis_out("fig_rep_growth.png"), f; px_per_unit = 2)
end
let
    f = Figure(size = (900, 560), fontsize = 17); hr = 10 .^ range(-3.2, -0.6; length = 60)
    ax = Axis(f[1, 1]; xscale = log10, yscale = log10, xreversed = true, xlabel = "h", ylabel = "size", title = "bill and budget (schematic, constants set to one)", titlealign = :left, AXS...)
    lines!(ax, hr, hr .^ 2; color = INK, linewidth = 3, label = "the bill: the skew c h², to be paid by the errors of the star")
    lines!(ax, hr, 4 .* hr .^ 2; color = C_P1, linewidth = 2.5, linestyle = :dash, label = "budget of exactness on [P1]²: (1+Λ) h², Λ bounded")
    lines!(ax, hr, 4 .* hr .^ 3; color = C_P2, linewidth = 2.5, linestyle = :dash, label = "budget of exactness on [P2]²: (1+Λ) h³, Λ bounded"); axislegend(ax; position = :lb, framevisible = false, labelsize = 14)
    CairoMakie.save(thesis_out("fig_rep_budget.png"), f; px_per_unit = 2)
end
println("ok")
