# report_figures_elliptic.jl -- figures of hodge_report.tex, numerical part:
#   fig_rep_poisson.png  Poisson P1 on mesh 4: exact and computed solutions, errors, and the error of the velocity (flux) on every edge
#   fig_rep_eig.png      eigenfunctions and errors of the eigenvalues
#   fig_rep_darcy.png    Darcy D1 on mesh 4: pressure and velocity of the three stars, pressure errors, velocity errors
#   fig_rep_conv.png     convergence: Poisson p and velocity, Darcy p and velocity, three stars, meshes 1-7
#   julia -t 4 scripts/elliptic/report_figures_elliptic.jl
include(joinpath(@__DIR__, "..", "stars", "stars_cache.jl")); include(joinpath(@__DIR__, "..", "shared", "thesis_plot_helpers.jl")); using Serialization
ENV["SW_MAIN"] = "0"; include(joinpath(@__DIR__, "..", "shallow_water", "sw_linear_check.jl"))
const S3 = ("H_DEC", "H_P1", "H_P2"); const VMAG = cgrad([colorant"#f0efec", colorant"#e9b97a", colorant"#c46a1f", colorant"#5a2a0a"])
RR = deserialize(joinpath(TSDIR, "report_runs.jls")); RE = deserialize(joinpath(TSDIR, "paper_elliptic.jls"))
mesh4 = VoronoiMesh("meshes/mesh4.nc"); St4 = cached_stars("mesh4"; mesh = mesh4)
# ---------------------------------------------------------------------------- Poisson
let mesh = mesh4, (polys, own) = periodic_copies(cellpolys(mesh4))
    f = Figure(size = (1560, 1230), fontsize = 15); r = Dict(s => RR[("mesh4", :P, 1, s)] for s in S3); pe = r["H_DEC"].pe; rg = extrema(pe)
    map_panel(f[1, 1], polys, own, pe, "exact solution p"; range = rg)
    for (j, s) in enumerate(S3); map_panel(f[1, 1+j], polys, own, r[s].ph, "computed p, $s"; range = rg); end
    Colorbar(f[1, 5]; colormap = :viridis, limits = rg, width = 12, label = "p")
    m = maximum(maximum(abs, r[s].ph .- pe) for s in S3)
    for (j, s) in enumerate(S3)
        e = r[s].ph .- pe; map_panel(f[2, 1+j], polys, own, e, @sprintf("error of p, %s, max %.1e", s, maximum(abs, e)); cmap = DIVM, range = (-m, m))
    end
    Colorbar(f[2, 5]; colormap = DIVM, limits = (-m, m), width = 12, tickformat = v -> [@sprintf("%.0e", x) for x in v], label = "error of p")
# row 3: velocity (flux) error on every edge of mesh 6, where the line defects are visible; left: the exact velocity
    mesh6 = VoronoiMesh("meshes/mesh6.nc"); S6 = cached_stars("mesh6"; mesh = mesh6); D6 = sparse(Float64.(d0_matrix_primal(mesh6))); ld6 = edge_lengths(mesh6, false)
    gr(x) = [TP * cos(TP * x[1]) * sin(TP * x[2]) - TP * sin(2TP * x[1]), TP * sin(TP * x[1]) * cos(TP * x[2])]; lapP1(x) = -2TP^2 * sin(TP * x[1]) * sin(TP * x[2]) - 2TP^2 * cos(2TP * x[1])
    Fex = reduce_dual_edges(mesh6, star_field(gr)).values; vn = maximum(norm(gr([a, b])) for a in 0:0.01:1, b in 0:0.01:1); fe6 = Dict{String,Vector{Float64}}()
    for s in S3; ph = solve_poisson_primal(mesh6, S6[s], x -> -lapP1(x)); fe6[s] = abs.(S6[s] * (D6 * ph) .- Fex) ./ ld6 ./ vn; end
    segs = [begin a, b = edge_endpoints(mesh6, e, true); (Point2f(a[1], a[2]), Point2f(b[1], b[2])) end for e in 1:mesh6.edges.n]; lo, hi = -6.0, -2.8
    for (j, s) in enumerate(S3)
        ax = Axis(f[3, 1+j]; aspect = DataAspect(), title = @sprintf("velocity error (mesh 6), %s, max %.1e", s, maximum(fe6[s])), titlealign = :left, titlesize = 14); hidedecorations!(ax)
        pts = Point2f[]; cols = Float64[]
        for (k, (a, b)) in enumerate(segs); push!(pts, a, b); append!(cols, fill(log10(max(fe6[s][k], 1e-12)), 2)); end
        linesegments!(ax, pts; color = cols, colormap = VMAG, colorrange = (lo, hi), linewidth = 1.3); xlims!(ax, 0, 1); ylims!(ax, 0, 1)
    end
    Colorbar(f[3, 5]; colormap = VMAG, limits = (lo, hi), width = 12, label = "log10 of the relative velocity error\n(mean normal component on a Voronoi edge)")
    ax = Axis(f[3, 1]; aspect = DataAspect(), title = "exact velocity −∇p", titlealign = :left, titlesize = 15); hidedecorations!(ax); gx = 0.04:0.08:1; P = vec([Point2f(x, y) for x in gx, y in gx]); Vv = [Vec2f(-gr([p[1], p[2]])...) for p in P]
    arrows2d!(ax, P, Vv; lengthscale = 0.005, color = INK, shaftwidth = 1.4, tipwidth = 6, tiplength = 6); xlims!(ax, 0, 1); ylims!(ax, 0, 1)
    ax = Axis(f[2, 1]); hidedecorations!(ax); hidespines!(ax); text!(ax, 0.5, 0.5; text = "errors of p (mesh 4)\non one colour scale", align = (:center, :center), fontsize = 15)
    CairoMakie.save(thesis_out("fig_rep_poisson.png"), f; px_per_unit = 2)
end
# ---------------------------------------------------------------------------- eigenfunctions and eigenvalues
let mesh = mesh4, (polys, own) = periodic_copies(cellpolys(mesh4))
    D0 = sparse(Float64.(d0_matrix_primal(mesh))); A = mesh.cells.area; M0 = Matrix(mass0_corrected(mesh)); L = Matrix(D0' * St4["H_P2"] * D0)
    F = eigen(M0 \ L); idx = sortperm(real.(F.values)); λ = real.(F.values[idx]); V = real.(F.vectors[:, idx]); f = Figure(size = (1560, 820), fontsize = 15)
    λex = sort([4π^2 * (i^2 + k^2) for i in -12:12 for k in -12:12 if (i, k) != (0, 0)])
    for (j, k) in enumerate((2, 6, 10, 14))
        v = V[:, k]; v ./= maximum(abs, v); e = abs(λ[k] - λex[k-1]) / λex[k-1]
        map_panel(f[1, j], polys, own, v, @sprintf("eigenfunction %d, H_P2: λ/4π² = %.5f  (exact %d)", k - 1, λ[k] / (4π^2), round(Int, λex[k-1] / (4π^2))); cmap = DIVM, range = (-1, 1), titlesize = 13)
    end
    E = deserialize(joinpath(TSDIR, "story_eigen.jls")); ax = Axis(f[2, 1:2]; yscale = log10, xlabel = "index of the eigenvalue", ylabel = "relative error of λ", title = "mesh 5, corrected mass matrix: relative error of the eigenvalues", titlealign = :left, axstyle_th...)
    for s in S3; scatter!(ax, 1:60, E[("mesh5", s, :C)].err[1:60]; color = STARCOL[s], marker = STARMK[s], markersize = 8, label = s); end
    axislegend(ax; position = :lt, framevisible = false)
    ax = Axis(f[2, 3:4]; xscale = log10, yscale = log10, xreversed = true, xlabel = "h", ylabel = "largest rel. error of λ₁…λ₂₀", title = "convergence: corrected mass matrix (solid), M0 = A (dashed)", titlealign = :left, xticks = ([1 / 8, 1 / 16, 1 / 32, 1 / 64], ["1/8", "1/16", "1/32", "1/64"]), axstyle_th...)
    hs = [1 / 8, 1 / 16, 1 / 32, 1 / 64]
    for s in S3, (mk, ls) in ((:C, :solid), (:A, :dash))
        y = [maximum(E[("mesh$k", s, mk)].err[1:20]) for k in 2:5]; scatterlines!(ax, hs, y; color = STARCOL[s], marker = STARMK[s], linestyle = ls, markersize = 11, linewidth = 2, label = mk == :C ? s : nothing)
    end
    lines!(ax, hs[2:4], 2.0e-2 .* (hs[2:4] ./ hs[2]) .^ 2; color = INK2, linestyle = :dot); text!(ax, 1 / 64, 2.0e-2 / 16; text = " h²", align = (:left, :center)); lines!(ax, hs[2:4], 6.0e-3 .* (hs[2:4] ./ hs[2]) .^ 4; color = INK2, linestyle = :dot); text!(ax, 1 / 64, 6.0e-3 / 256; text = " h⁴", align = (:left, :center)); axislegend(ax; position = :lb, framevisible = false)
    CairoMakie.save(thesis_out("fig_rep_eig.png"), f; px_per_unit = 2)
end
# ---------------------------------------------------------------------------- Darcy
let mesh = mesh4
    tris, D1, _ = triangles_ccw(mesh); At = [T.area for T in tris]; polys, own = periodic_copies(tripolys(tris)); cc = ccpoints(tris); ex = darcy_data(mesh, tris; v = VD, p = PD, g = x -> [0.0, 0.0], φ = PHID); shift(q) = q .- sum(At .* q) / sum(At); pe = shift(ex.p)
    Vref = triangle_velocity(mesh, tris, ex.f); vm = maximum(norm.(eachrow(Vref))); sel = 1:7:length(tris)
    f = Figure(size = (1560, 1230), fontsize = 15); rg = extrema(pe); sol = Dict(s => solve_darcy(St4[s], D1, ex.G, ex.Φ, At) for s in S3); E = Dict{String,Any}()
    for s in S3; ph = shift(sol[s][2]); E[s] = (dp = ph .- pe, dv = norm.(eachrow(triangle_velocity(mesh, tris, sol[s][1]) .- Vref)), ph = ph, V = triangle_velocity(mesh, tris, sol[s][1])); end
    map_panel(f[1, 1], polys, own, pe, "exact: pressure, arrows: velocity"; range = rg, pts = cc[sel], dirs = vecs(Vref)[sel], ls = 0.06 / vm, acol = :white, titlesize = 14)
    for (j, s) in enumerate(S3); map_panel(f[1, 1+j], polys, own, E[s].ph, "computed, $s"; range = rg, pts = cc[sel], dirs = vecs(E[s].V)[sel], ls = 0.06 / vm, acol = :white, titlesize = 14); end
    Colorbar(f[1, 5]; colormap = :viridis, limits = rg, width = 12, label = "pressure"); mp = maximum(maximum(abs, E[s].dp) for s in S3); mv = maximum(maximum(E[s].dv) for s in S3)
    for (j, s) in enumerate(S3)
        map_panel(f[2, 1+j], polys, own, E[s].dp, @sprintf("pressure error, %s, max %.1e", s, maximum(abs, E[s].dp) / maximum(abs, pe)); cmap = DIVM, range = (-mp, mp), titlesize = 14)
        map_panel(f[3, 1+j], polys, own, E[s].dv, @sprintf("velocity error, %s, max %.1e", s, maximum(E[s].dv) / vm); cmap = VMAG, range = (0, mv), titlesize = 14)
    end
    Colorbar(f[2, 5]; colormap = DIVM, limits = (-mp, mp), width = 12, tickformat = v -> [@sprintf("%.0e", x) for x in v], label = "pressure error"); Colorbar(f[3, 5]; colormap = VMAG, limits = (0, mv), width = 12, tickformat = v -> [@sprintf("%.0e", x) for x in v], label = "velocity error")
    for i in 2:3; ax = Axis(f[i, 1]); hidedecorations!(ax); hidespines!(ax); text!(ax, 0.5, 0.5; text = i == 2 ? "pressure errors\n(one colour scale)" : "velocity errors per triangle\n(one colour scale)", align = (:center, :center), fontsize = 15); end
    CairoMakie.save(thesis_out("fig_rep_darcy.png"), f; px_per_unit = 2)
end
# ---------------------------------------------------------------------------- convergence
let
    ms = ["mesh$i" for i in 1:7]; hs = [RR[(m, :h)] for m in ms]; f = Figure(size = (1560, 900), fontsize = 15); xt = (hs, ["1/$(round(Int, 1 / h))" for h in hs])
    function panel(pos, get, title, yl, slopes; ref = nothing)
        ax = Axis(pos; xscale = log10, yscale = log10, xreversed = true, xticks = xt, xlabel = "h", ylabel = yl, title = title, titlealign = :left, axstyle_th...)
        for s in S3; scatterlines!(ax, hs, [get(m, s) for m in ms]; color = STARCOL[s], marker = STARMK[s], markersize = 11, linewidth = 2, label = s); end
        for (k, a) in slopes; xx = [hs[3], hs[7]]; lines!(ax, xx, a .* (xx ./ xx[1]) .^ k; color = INK2, linestyle = :dot); text!(ax, hs[7], a * (hs[7] / hs[3])^k; text = " h" * Dict(1 => "", 2 => "²", 3 => "³", 4 => "⁴")[k], align = (:left, :center), color = INK2); end
        return ax
    end
    ax = panel(f[1, 1], (m, s) -> RR[(m, :P, 1, s)].inf, "Poisson: solution p (max)", "relative max error", ((2, 3e-2), (4, 1.2e-3))); axislegend(ax; position = :lb, framevisible = false)
    panel(f[1, 2], (m, s) -> RR[(m, :P, 1, s)].finf, "Poisson: velocity −∇p (max)", "relative max error of the flux", ((1, 6e-3), (2, 3e-2), (3, 4e-3)))
    panel(f[1, 3], (m, s) -> RR[(m, :P, 1, s)].f2, "Poisson: velocity (mean square)", "relative error of the flux", ((2, 1e-2), (3, 2e-3)))
    panel(f[2, 1], (m, s) -> RE[(m, :D, 1, s)].p_inf, "Darcy: pressure (max)", "relative max error", ((2, 2e-3), (4, 2.5e-4)))
    panel(f[2, 2], (m, s) -> RE[(m, :D, 1, s)].flux_inf, "Darcy: velocity (max)", "relative max error of the flux", ((1, 1.2e-2), (2, 2.4e-3), (3, 6e-4)))
    panel(f[2, 3], (m, s) -> RE[(m, :D, 1, s)].flux, "Darcy: velocity (mean square)", "relative error of the flux", ((2, 4e-3), (3, 1.5e-4)))
    CairoMakie.save(thesis_out("fig_rep_conv.png"), f; px_per_unit = 2)
end
println("elliptic figures ok")
