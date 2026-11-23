# ============================================================================
#  three_stars_figures.jl -- the figures of three_stars_report.tex, from ts_data/results.jls.
#
#      julia -t 4 scripts/stars/three_stars_figures.jl      (after three_stars_run.jl)
#
#  fig_ts_example.png   the rows of H_P1 and H_P2 for edge 17 of mesh 3, and DEC's residual on the [P2]^2 basis
#  fig_ts_hodge.png     Hodge errors on meshes 1-6
#  fig_ts_poisson.png   Poisson errors on meshes 1-6          fig_ts_poisson_sol.png  solutions and errors on mesh 4
#  fig_ts_spectra.png   eigenvalues in the complex plane, mesh 4
#  fig_ts_darcy.png     Darcy errors on meshes 1-6            fig_ts_darcy_sol.png    pressure, velocity and errors on mesh 4
# ============================================================================
include(joinpath(@__DIR__, "three_stars.jl"))
using CairoMakie
const R = deserialize(joinpath(TSDIR, "results.jls"))
const COL = Dict("H_DEC" => colorant"#4d4d4d", "H_P1" => colorant"#2b6cb0", "H_P2" => colorant"#c23b3a")
const MK = Dict("H_DEC" => :circle, "H_P1" => :diamond, "H_P2" => :rect)
const LBL = Dict("H_DEC" => "H_DEC (diagonal)", "H_P1" => "H_P1 (symmetric, exact on [P₁]²)", "H_P2" => "H_P2 (exact on [P₂]², 2% non-symmetric)")
const DIV = cgrad([colorant"#1c5cab", colorant"#f0efec", colorant"#c23b3a"])
const HS = [R[(m, :h)] for m in MESHES]
const XT = (HS, ["1/4", "1/8", "1/16", "1/32", "1/64", "1/128"])
sup(k) = Dict(0 => "⁰", 1 => "", 2 => "²", 3 => "³", 4 => "⁴")[k]
outp(name) = joinpath(@__DIR__, name)

"log-log convergence axis; the reference slopes `slopes` fan out from one point in the empty lower-left corner."
function rate_axis(pos, title, ylabel, f; slopes = (1, 2, 3), span = 3)
    ax = Axis(pos; xscale = log10, yscale = log10, xreversed = true, xticks = XT, xlabel = "h", ylabel = ylabel, title = title,
              titlesize = 14)
    for name in STARS
        scatterlines!(ax, HS, [f(R[(m, name)]) for m in MESHES]; color = COL[name], marker = MK[name], markersize = 11)
    end
    lo = minimum(minimum(f(R[(m, n)]) for m in MESHES) for n in STARS)
    hh = [HS[1], HS[span]]                                              # each reference line spans HS[1] -> HS[span]
    y0 = 2 * lo * (hh[1] / hh[2])^maximum(slopes)                       # common start; the steepest ends at 2·lo
    for (i, k) in enumerate(slopes)
        lines!(ax, hh, y0 .* (hh ./ hh[1]) .^ k; color = :gray, linestyle = [:dot, :dash, :solid, :dashdot][i], linewidth = 1.5)
        text!(ax, hh[2], y0 * (hh[2] / hh[1])^k; text = "  h" * sup(k), color = :gray, fontsize = 12, align = (:left, :center))
    end
    return ax
end

"One legend for the three stars, placed in its own layout cell."
star_legend(pos) = Legend(pos, [[LineElement(color = COL[n]), MarkerElement(color = COL[n], marker = MK[n], markersize = 11)] for n in STARS],
                          [LBL[n] for n in STARS]; orientation = :horizontal, framevisible = false, labelsize = 13, tellheight = true)

# ---------------------------------------------------------------- 1. the worked example
let ex = R[:example], f = Figure(size = (1300, 470))
    rowlim = maximum(maximum(abs, values(ex[Symbol("row_" * n)])) for n in ("H_P1", "H_P2")) / ex[:hd]
    for (i, name) in enumerate(("H_P1", "H_P2"))
        row = ex[Symbol("row_" * name)]
        ax = Axis(f[1, i]; aspect = DataAspect(), title = (i == 1 ? "(a) " : "(b) ") * "row e = 17 of " * name *
                  ": correction C_ej / H_DEC,ee", titlesize = 14)
        hidedecorations!(ax)
        for (j, (a, b)) in ex[:primal]
            lines!(ax, [a[1], b[1]], [a[2], b[2]]; color = (:gray, 0.35), linewidth = 0.8)
        end
        for (j, c) in row
            a, b = ex[:primal][j]
            lines!(ax, [a[1], b[1]], [a[2], b[2]]; color = c / ex[:hd], colormap = DIV, colorrange = (-rowlim, rowlim), linewidth = 5)
        end
        a, b = ex[:primal][ex[:e]]
        lines!(ax, [a[1], b[1]], [a[2], b[2]]; color = :black, linewidth = 2)
        q1, q2 = ex[:dual]
        lines!(ax, [q1[1], q2[1]], [q1[2], q2[2]]; color = :black, linewidth = 2, linestyle = :dash)
        s = ex[:s]; x0 = ex[:x0]; xlims!(ax, x0[1] - 3.2s, x0[1] + 3.2s); ylims!(ax, x0[2] - 3.2s, x0[2] + 3.2s)
    end
    Colorbar(f[1, 3]; colormap = DIV, limits = (-rowlim, rowlim), width = 12)
    ax = Axis(f[1, 4]; title = "(c) DEC residual r_e on the [P₂]² basis", xticks = (1:12, ["(1,0)", "(0,1)", "(X,0)", "(Y,0)", "(0,X)",
              "(0,Y)", "(X²,0)", "(XY,0)", "(Y²,0)", "(0,X²)", "(0,XY)", "(0,Y²)"]), xticklabelrotation = π / 2.5, titlesize = 14,
              ylabel = "(R̃⋆f_m)_e − H_DEC,ee (R_h f_m)_e")
    barplot!(ax, 1:12, ex[:r2]; color = [m <= 2 ? :gray : m <= 6 ? COL["H_P1"] : COL["H_P2"] for m in 1:12])
    colsize!(f.layout, 4, Relative(0.32))
    CairoMakie.save(outp("fig_ts_example.png"), f; px_per_unit = 2)
end

# ---------------------------------------------------------------- 2. Hodge errors
let f = Figure(size = (1150, 860))
    rate_axis(f[1, 1], "(a) field d, max norm", "max_e |ε_e| / (|ẽ| ‖u‖∞)", r -> r[:hd].inf)
    rate_axis(f[1, 2], "(b) field d, mean-square norm", "rms of |ε_e|/|ẽ|, relative", r -> r[:hd].l2)
    rate_axis(f[2, 1], "(c) field u₂, max norm", "max_e |ε_e| / (|ẽ| ‖u‖∞)", r -> r[:hu2].inf)
    rate_axis(f[2, 2], "(d) field u₂, mean-square norm", "rms of |ε_e|/|ẽ|, relative", r -> r[:hu2].l2)
    star_legend(f[3, 1:2])
    CairoMakie.save(outp("fig_ts_hodge.png"), f; px_per_unit = 2)
end

# ---------------------------------------------------------------- 3. Poisson errors
let f = Figure(size = (1450, 470))
    rate_axis(f[1, 1], "(a) nodal error, max", "max |p_h − p| / max |p|", r -> r[:poisson].inf; slopes = (2, 3, 4))
    rate_axis(f[1, 2], "(b) nodal error, L²", "‖p_h − p‖_A / ‖p‖_A", r -> r[:poisson].l2; slopes = (2, 3, 4))
    rate_axis(f[1, 3], "(c) gradient error", "‖d₀(p_h − p)‖_H_DEC, relative", r -> r[:poisson].grad; slopes = (1, 2, 3))
    rate_axis(f[1, 4], "(d) Laplacian truncation error", "max |τ_v| / max |Δp|", r -> r[:tau]; slopes = (0, 1, 2))
    star_legend(f[2, 1:4])
    CairoMakie.save(outp("fig_ts_poisson.png"), f; px_per_unit = 2)
end

# ---------------------------------------------------------------- polygons of mesh 4
mesh4 = VoronoiMesh("meshes/mesh4.nc")
cellpolys = [Point2f[(p[1], p[2]) for p in get_cells_points(mesh4, c)] for c in 1:mesh4.cells.n]
tris4, _, _ = triangles_ccw(mesh4)
tripolys = [Point2f[(p[1], p[2]) for p in T.pts] for T in tris4]

"Polygons cut by the periodic boundary are unwrapped about their own centre; add their shifted copies so that the
clipped unit square is fully covered.  Returns (polys, index of the owner of each polygon)."
function periodic_copies(polys)
    out = Vector{Vector{Point2f}}(); own = Int[]
    for (i, P) in enumerate(polys)
        push!(out, P); push!(own, i)
        any(p -> p[1] < 0 || p[1] > 1 || p[2] < 0 || p[2] > 1, P) || continue
        for sx in -1:1, sy in -1:1
            (sx, sy) == (0, 0) && continue
            push!(out, [p .+ Point2f(sx, sy) for p in P]); push!(own, i)
        end
    end
    return out, own
end

"Row 1: exact and the three solutions (shared colour range); row 2: the three errors (each its own range)."
function solution_figure(file, polys0, exact, sols, errs, ttl; arrows = nothing)
    polys, own = periodic_copies(polys0)
    f = Figure(size = (1500, 1000))
    lim = maximum(abs, exact)
    for (i, (name, v)) in enumerate(vcat([("exact", exact)], sols))
        ax = Axis(f[1, i]; aspect = DataAspect(), title = name == "exact" ? "exact " * ttl : name, titlesize = 14)
        hidedecorations!(ax)
        poly!(ax, polys; color = v[own], colormap = :viridis, colorrange = (-lim, lim), strokewidth = 0)
        if arrows !== nothing
            pts, dirs = arrows[i]
            arrows2d!(ax, pts, dirs; lengthscale = 0.012, color = :white, shaftwidth = 1.2, tipwidth = 5, tiplength = 5)
        end
        xlims!(ax, 0, 1); ylims!(ax, 0, 1)
    end
    Colorbar(f[1, 5]; colormap = :viridis, limits = (-lim, lim), width = 12)
    for (i, (name, v)) in enumerate(errs)
        m = maximum(abs, v)
        ax = Axis(f[2, i + 1]; aspect = DataAspect(), title = @sprintf("error of %s, max %.1e", name, m), titlesize = 14)
        hidedecorations!(ax)
        poly!(ax, polys; color = v[own], colormap = DIV, colorrange = (-m, m), strokewidth = 0)
        xlims!(ax, 0, 1); ylims!(ax, 0, 1)
        Colorbar(f[3, i + 1]; colormap = DIV, limits = (-m, m), vertical = false, height = 10, ticklabelsize = 10)
    end
    Label(f[2, 1], "row 2: computed − exact\n(each panel has its own\ncolour range)"; tellwidth = false, fontsize = 14)
    rowsize!(f.layout, 1, Relative(0.45)); rowsize!(f.layout, 2, Relative(0.45))
    CairoMakie.save(outp(file), f; px_per_unit = 2)
end

let r = Dict(n => R[("mesh4", n)][:sol] for n in STARS)
    ex = r["H_DEC"].poisson_exact
    solution_figure("fig_ts_poisson_sol.png", cellpolys, ex, [(n, r[n].poisson) for n in STARS],
                    [(n, r[n].poisson .- ex) for n in STARS], "p (mesh 4)")
end

# ---------------------------------------------------------------- 4. spectra
let f = Figure(size = (1450, 760))
    for (i, name) in enumerate(STARS)
        sp = R[("mesh4", name)][:spec]
        ax = Axis(f[1, i]; xlabel = "Re λ", ylabel = "Im λ", title = "eigenvalues of H_DEC⁻¹ H, " * name * " (mesh 4)", titlesize = 14)
        scatter!(ax, real.(sp.H), imag.(sp.H); color = COL[name], markersize = 4)
        xlims!(ax, 0.8, 1.25); ylims!(ax, -0.006, 0.006)
        λ = sp.L; λm = maximum(abs.(λ))
        ax2 = Axis(f[2, i]; xlabel = "Re λ / λ_max", ylabel = "Im λ / λ_max", title = "eigenvalues of A⁻¹d₀ᵀHd₀, " * name, titlesize = 14)
        scatter!(ax2, real.(λ) ./ λm, imag.(λ) ./ λm; color = COL[name], markersize = 4)
        xlims!(ax2, -0.02, 1.02); ylims!(ax2, -0.005, 0.005)
    end
    CairoMakie.save(outp("fig_ts_spectra.png"), f; px_per_unit = 2)
end

# ---------------------------------------------------------------- 5. Darcy errors and solutions
let f = Figure(size = (1450, 470))
    rate_axis(f[1, 1], "(a) flux", "‖f_h − f‖_H_DEC / ‖f‖_H_DEC", r -> r[:darcy].flux)
    rate_axis(f[1, 2], "(b) flux, max", "max_e |f_h − f|/(|e| ‖v‖∞)", r -> r[:darcy].flux_inf)
    rate_axis(f[1, 3], "(c) pressure at circumcentres, L²", "‖p_h − p(c_T)‖ / ‖p‖", r -> r[:darcy].p_l2; slopes = (2, 3, 4))
    rate_axis(f[1, 4], "(d) pressure, Hirani's measure", "‖p_h − p‖_L²(Ω), p_h constant on T", r -> r[:darcy].p_hirani;
              slopes = (1, 2), span = 2)
    star_legend(f[2, 1:4])
    CairoMakie.save(outp("fig_ts_darcy.png"), f; px_per_unit = 2)
end

let r = Dict(n => R[("mesh4", n)][:sol] for n in STARS), E = R[("mesh4", :exact)]
    pick = 1:9:length(tris4)
    cent = [Point2f(Tuple(sum(T.pts) ./ 3)) for T in tris4[pick]]
    arr(V) = (cent, [Vec2f(V[t, 1], V[t, 2]) for t in pick])
    solution_figure("fig_ts_darcy_sol.png", tripolys, E.darcy_p, [(n, r[n].darcy_p) for n in STARS],
                    [(n, r[n].darcy_p .- E.darcy_p) for n in STARS], "p and v (mesh 4)";
                    arrows = [arr(E.darcy_v), (arr(r[n].darcy_v) for n in STARS)...])
end
println("figures written")
