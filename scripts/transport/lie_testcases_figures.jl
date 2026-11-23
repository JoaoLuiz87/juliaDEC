# ============================================================================
#  lie_testcases_figures.jl -- figures of lie_testcases_check.jl
#
#      julia -t 4 scripts/transport/lie_testcases_figures.jl
#          ->  output/figures/fig_lie_tests_convergence.png   (ℓ2 and ℓ∞ errors against h, five tests)
#              output/figures/fig_lie_tests_gallery.png       (mesh5: initial, t = T/2, exact at T, errors of two schemes)
# ============================================================================
include(joinpath(@__DIR__, "lie_derivative.jl"))
using CairoMakie, Printf, Serialization
R = deserialize(joinpath(TSDIR, "lie_testcases.jls"))
M = [m for m in ("mesh2", "mesh3", "mesh4", "mesh5", "mesh6", "mesh7") if haskey(R, (m, :h))]
h = [R[(m, :h)] for m in M]
BLUE, RED, GREEN, GRAY, ORANGE, PURPLE = colorant"#2b6cb0", colorant"#c23b3a", colorant"#2f8f4e", colorant"#6b6b6b", colorant"#d08a1c", colorant"#7b4fa0"
sup(k) = Dict(1 => "¹", 2 => "²", 3 => "³", 4 => "⁴")[k]
hticks = ([1/4, 1/16, 1/64, 1/256], ["1/4", "1/16", "1/64", "1/256"])
names = collect(String.(keys(TESTS)))
sel = (("H_DEC", "TRiSK", "H_DEC (= DEC = TRiSK)", RED, :circle, :solid),
       ("H_P2", "TRiSK", "H_P2 TRiSK", PURPLE, :diamond, :dash),
       ("H_P2", "DEC", "H_P2 DEC", PURPLE, :diamond, :solid),
       ("H_P2", "CFV2", "H_P2 CFV2 (centred, unstable)", GREEN, :star5, :dash),
       ("H_DEC", "UPW2", "H_DEC UPW2", RED, :xcross, :dash),
       ("H_P1", "UPW2", "H_P1 UPW2", ORANGE, :rect, :solid),
       ("H_P2", "UPW2", "H_P2 UPW2", BLUE, :utriangle, :solid),
       ("exact", "UPW2", "exact flux UPW2", GRAY, :hexagon, :dot))

f = Figure(size = (1900, 760), fontsize = 13)
for (r, (norm_, lab)) in enumerate(((:l2, "ℓ₂"), (:linf, "ℓ∞")))
    for (j, name) in enumerate(names)
        ax = Axis(f[r, j]; xscale = log10, yscale = log10, xreversed = true, xticks = hticks, xlabel = r == 2 ? "h" : "",
                  ylabel = j == 1 ? "relative $lab error at T" : "", title = r == 1 ? replace(name, "_" => " ") : "", titlesize = 15)
        for (s, v, l, c, mk, ls) in sel
            scatterlines!(ax, h, [getfield(R[(m, name, s, v)], norm_) for m in M]; color = c, marker = mk, linestyle = ls, label = l, markersize = 8)
        end
        y0 = getfield(R[(M[end-2], name, "H_P2", "CFV2")], norm_)
        for k in (2, 3)
            xx = [h[end-2], h[end]]; lines!(ax, xx, 1.6 * y0 .* (xx ./ xx[1]) .^ k; color = :black, linestyle = :dot)
            text!(ax, xx[2], 1.6 * y0 * (xx[2] / xx[1])^k; text = " h" * sup(k), fontsize = 12, align = (:left, :center))
        end
        r == 1 && j == length(names) && Legend(f[1:2, length(names) + 1], ax; framevisible = false, labelsize = 12)
    end
end
CairoMakie.save(joinpath(@__DIR__, "..", "..", "output", "figures", "fig_lie_tests_convergence.png"), f; px_per_unit = 2)

# ---------------- gallery on mesh5
G = lie_setup("mesh5"; cfv = (), upw = (2,)); mesh = G["mesh"]; A = G["A"]; hh = G["h"]
polys = Vector{Vector{Point2f}}(); owner = Int[]
for c in 1:mesh.cells.n
    vs = [Point2f(_vec2(closest(mesh.cells.position[c], mesh.vertices.position[v], 1.0, 1.0))...) for v in mesh.cells.vertices[c]]
    for sx in (-1, 0, 1), sy in (-1, 0, 1)
        w = [v + Point2f(sx, sy) for v in vs]
        ((sx == 0 && sy == 0) || any(p -> 0 <= p[1] <= 1 && 0 <= p[2] <= 1, w)) && (push!(polys, w); push!(owner, c))
    end
end
g = Figure(size = (1700, 1530), fontsize = 13)
rowname = ("initial ρ", "t = T/2  (H_P2 UPW2)", "exact at T", "error at T:  H_DEC (= DEC = TRiSK)", "error at T:  H_P2 UPW2")
for (j, name) in enumerate(names)
    test = TESTS[Symbol(name)]
    ρ0 = cell_averages(mesh, test.ρ0) .* A
    n = round(Int, test.T / (hh / 8))
    _, fr = rk4(rhs_function(G, "UPW2", "H_P2", test), ρ0, test.T, hh / 8; save_every = n ÷ 2)
    ex = haskey(R, ("mesh5", name, :exact)) ? R[("mesh5", name, :exact)] : exact_at_T(mesh, test)
    e1 = R[("mesh5", name, "H_DEC", "TRiSK", :final)] .- ex; e2 = R[("mesh5", name, "H_P2", "UPW2", :final)] .- ex
    vals = (ρ0 ./ A, fr[2] ./ A, ex, e1, e2)
    vmax = maximum(maximum.((vals[1], vals[2], vals[3]))); emax = maximum(abs.(e1))
    for (r, v) in enumerate(vals)
        ax = Axis(g[r, j]; aspect = DataAspect(), title = r == 1 ? replace(name, "_" => " ") : "", titlesize = 15)
        hidedecorations!(ax); limits!(ax, 0, 1, 0, 1)
        cm, cr = r <= 3 ? (:turbo, (-0.03 * vmax, vmax)) : (:balance, (-emax, emax))
        poly!(ax, polys; color = v[owner], colormap = cm, colorrange = cr, strokewidth = 0)
        r >= 4 && text!(ax, 0.02, 0.02; text = @sprintf("max |e| = %.1e", maximum(abs.(v))), space = :relative, fontsize = 12)
        r == 3 && text!(ax, 0.02, 0.02; text = @sprintf("error scale ±%.1e", emax), space = :relative, fontsize = 12, color = :white)
    end
end
for (r, rn) in enumerate(rowname)
    Label(g[r, 0], rn; rotation = π / 2, tellheight = false, fontsize = 13)
end
# (title in the caption of the note: mesh5, rows 4-5 on one symmetric colour scale per test, that of H_DEC)


CairoMakie.save(joinpath(@__DIR__, "..", "..", "output", "figures", "fig_lie_tests_gallery.png"), g; px_per_unit = 1.2)
