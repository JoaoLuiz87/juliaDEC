# paper_figures.jl -- new figures of paper/hodge_gap.tex
#      FIG=conv,stencil,spec julia scripts/elliptic/paper_figures.jl
#   conv    : fig_pp_elliptic_conv.png  Poisson P1-P3 (max nodal error) and Darcy D1-D3 (flux, H_DEC norm), meshes 1-7  (ts_data/paper_elliptic.jls)
#   stencil : fig_pp_stencil.png        one row of the Laplacian D0' H D0 for the three stars (mesh 3)
#   spec    : fig_pp_spectrum.png       complex eigenvalues of A^-1 D0' H_P2 D0 (ts_data/paper_spectra.jls)
include(joinpath(@__DIR__, "..", "stars", "stars_cache.jl"))
include(joinpath(@__DIR__, "..", "shared", "thesis_plot_helpers.jl"))
using Serialization
const FIGS = split(get(ENV, "FIG", "conv,stencil"), ",")
const S3 = ("H_DEC", "H_P1", "H_P2")
const M7 = ("mesh1", "mesh2", "mesh3", "mesh4", "mesh5", "mesh6", "mesh7")
LBL = Dict("H_DEC" => "H_DEC", "H_P1" => "H_P1", "H_P2" => "H_P2")

if "conv" in FIGS
    RE = deserialize(joinpath(TSDIR, "paper_elliptic.jls"))
    hs = [RE[(m, :h)] for m in M7]
    f = Figure(size = (1180, 760), fontsize = 16)
    pn = ("P1 smooth", "P2 localised", "P3 several scales"); dn = ("D1 potential flow", "D2 with body force", "D3 localised")
    for i in 1:3
        v1 = [RE[(m, :P, i, "H_DEC")].inf for m in M7]
        ax = conv_axis(f[1, i], hs, "Poisson " * pn[i]; ylabel = i == 1 ? "max nodal error (rel.)" : "", anchor = v1[2] * 0.5)
        for s in S3
            scatterlines!(ax, hs, [RE[(m, :P, i, s)].inf for m in M7]; color = STARCOL[s], marker = STARMK[s], markersize = 9, linewidth = 2, label = LBL[s])
        end
        i == 1 && axislegend(ax; position = :lb, framevisible = false, labelsize = 14)
        w1 = [RE[(m, :D, i, "H_DEC")].flux for m in M7]
        xlims!(ax, 0.32, 0.0026)
        ax = conv_axis(f[2, i], hs, "Darcy " * dn[i]; ylabel = i == 1 ? "flux error, H_DEC norm (rel.)" : "", anchor = w1[2] * 0.5)
        for s in S3
            scatterlines!(ax, hs, [RE[(m, :D, i, s)].flux for m in M7]; color = STARCOL[s], marker = STARMK[s], markersize = 9, linewidth = 2)
        end
        xlims!(ax, 0.32, 0.0026)
        begin
        end
    end
    CairoMakie.save(thesis_out("fig_pp_elliptic_conv.png"), f; px_per_unit = 2)
    println("fig_pp_elliptic_conv.png")
end

if "hodge" in FIGS
    RE = deserialize(joinpath(TSDIR, "paper_elliptic.jls"))
    hs = [RE[(m, :h)] for m in M7]
    f = Figure(size = (1180, 840), fontsize = 16)
    for (i, (fk, fn)) in enumerate(((:hd, "d"), (:hu2, "u₂"))), (j, (nk, nn)) in enumerate(((:inf, "r∞"), (:l2, "r₂")))
        a1 = RE[("mesh2", "H_DEC")][fk][nk]
        ax = conv_axis(f[i, j], hs, "$nn($fn)"; ylabel = j == 1 ? "relative Hodge error" : "", anchor = a1 * 0.4, slopes = (1, 2, 3))
        for s in S3
            scatterlines!(ax, hs, [RE[(m, s)][fk][nk] for m in M7]; color = STARCOL[s], marker = STARMK[s], markersize = 9, linewidth = 2, label = LBL[s])
        end
        xlims!(ax, 0.32, 0.0026)
        (i, j) == (1, 1) && axislegend(ax; position = :lb, framevisible = false, labelsize = 14)
    end
    CairoMakie.save(thesis_out("fig_pp_hodge.png"), f; px_per_unit = 2)
    println("fig_pp_hodge.png")
end

if "stencil" in FIGS
    mesh = VoronoiMesh("meshes/mesh3.nc"); S = cached_stars("mesh3"; mesh)
    D0 = sparse(Float64.(d0_matrix_primal(mesh)))
    P = [_pt(x) for x in mesh.cells.position]
    v = argmin([norm(p .- [0.5, 0.5]) for p in P])
    polys, own = periodic_copies(cellpolys(mesh))
    c0 = Point2f(P[v][1], P[v][2]); R = 0.21
    keep = [k for k in eachindex(polys) if norm(sum(polys[k]) / length(polys[k]) - c0) < R]
    f = Figure(size = (1180, 430), fontsize = 16)
    cm = cgrad([colorant"#fde8c8", colorant"#eb6834", colorant"#5a2a0a"])
    for (j, s) in enumerate(S3)
        L = D0' * S[s] * D0; row = L[v, :]
        nzs = findall(!iszero, row)
        ax = Axis(f[1, j]; aspect = DataAspect(), title = @sprintf("%s: %d entries in row %d of D₀ᵀHD₀", s, length(nzs), v), titlealign = :left, titlesize = 15)
        hidedecorations!(ax)
        vals = [abs(row[own[k]]) / abs(row[v]) for k in keep]
        cols = [x == 0 ? RGBf(0.95, 0.95, 0.94) : get(cm, (clamp(log10(x), -4, 0) + 4) / 4) for x in vals]
        poly!(ax, polys[keep]; color = cols, strokecolor = :white, strokewidth = 1)
        scatter!(ax, [c0]; color = :black, marker = :star5, markersize = 14)
        xlims!(ax, c0[1] - R * 0.75, c0[1] + R * 0.75); ylims!(ax, c0[2] - R * 0.75, c0[2] + R * 0.75)
    end
    Colorbar(f[1, 4]; colormap = cm, limits = (-4, 0), label = "log₁₀ |L_vw| / L_vv", width = 14)
    CairoMakie.save(thesis_out("fig_pp_stencil.png"), f; px_per_unit = 2)
    println("fig_pp_stencil.png")
end

if "spec" in FIGS
    RS = deserialize(joinpath(TSDIR, "paper_spectra.jls"))
    f = Figure(size = (1180, 470), fontsize = 16)
    ax = Axis(f[1, 1]; xlabel = "Re λ / λmax", ylabel = "Im λ / λmax", title = "(a) eigenvalues of A⁻¹D₀ᵀH_P2D₀", axstyle_th...)
    r5 = RS[("mesh5", "H_P2")]; λ5 = r5[:λ] ./ r5[:λmax]
    scatter!(ax, real.(λ5), imag.(λ5); color = (C_P2, 0.6), markersize = 4, label = "mesh 5, all (dense)")
    if haskey(RS, ("mesh7", "H_P2")) && haskey(RS[("mesh7", "H_P2")], :found)
        r7 = RS[("mesh7", "H_P2")]; λ7 = r7[:found] ./ r7[:λmax]
        scatter!(ax, real.(λ7), imag.(λ7); color = (C_P1, 0.7), markersize = 4, marker = :diamond, label = "mesh 7, found by Arnoldi")
    end
    axislegend(ax; position = :lb, framevisible = false, labelsize = 13)
    ms = [m for m in ("mesh2", "mesh3", "mesh4", "mesh5", "mesh6", "mesh7") if haskey(RS, (m, "H_P2"))]
    hs = [RS[(m, :h)] for m in ms]
    imx = [haskey(RS[(m, "H_P2")], :dense) ? RS[(m, "H_P2")][:dense].imax : RS[(m, "H_P2")][:arnoldi].imax for m in ms]
    βs = [RS[(m, "H_P2")][:β] / RS[(m, "H_P2")][:λmax] for m in ms]
    ax2 = Axis(f[1, 2]; xscale = log10, yscale = log10, xreversed = true, xticks = (hs, ["1/" * string(round(Int, 1 / h)) for h in hs]),
               xlabel = "h", ylabel = "relative to λmax", title = "(b) imaginary parts under refinement", axstyle_th...)
    nd = count(m -> haskey(RS[(m, "H_P2")], :dense), ms)
    scatterlines!(ax2, hs[1:nd], imx[1:nd]; color = C_P2, marker = :rect, markersize = 10, linewidth = 2, label = "max |Im λ| / λmax (all eigenvalues)")
    if nd < length(ms)
        lines!(ax2, hs[nd:end], imx[nd:end]; color = C_P2, linewidth = 1.5, linestyle = :dot)
        scatter!(ax2, hs[nd+1:end], imx[nd+1:end]; color = :white, strokecolor = C_P2, strokewidth = 2, marker = :rect, markersize = 11,
                 label = "lower bound (Arnoldi, part of the spectrum)")
    end
    scatterlines!(ax2, hs, βs; color = INK2, marker = :circle, markersize = 8, linewidth = 1.5, linestyle = :dash, label = "Bendixson bound β / λmax")
    if haskey(RS, (:rep, 3))
        hr = [0.0625 / n for n in 1:3]; ir = [RS[(:rep, n)].st.imax for n in 1:3]
        scatter!(ax2, hr, ir; color = AQUA, marker = :utriangle, markersize = 12, label = "replications of mesh 3 (h = 1/16, 1/32, 1/48)")
    end
    ylims!(ax2, 3e-4, 1e-1)
    axislegend(ax2; position = :lt, framevisible = false, labelsize = 13)
    CairoMakie.save(thesis_out("fig_pp_spectrum.png"), f; px_per_unit = 2)
    println("fig_pp_spectrum.png")
end
