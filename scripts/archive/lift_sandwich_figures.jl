# ============================================================================
#  lift_sandwich_figures.jl -- figures for the subsections
#     "Why the P1 lift has negative eigenvalues and the P2 lift does not"
#     "Why the sandwich gets the fluxes wrong"
#  of hodge_symmetry_consistency.tex
#
#      julia scripts/archive/lift_sandwich_figures.jl
# ============================================================================
using VoronoiMeshes, TensorsLite, TensorsLiteGeometry, SmallCollections, LinearAlgebra, NCDatasets
using StaticArrays, SparseArrays, QuadGK, Printf, CairoMakie
Base.similar(bc::Base.Broadcast.Broadcasted, ::Type{Union{}}) = Array{Union{}}(undef, length.(axes(bc)))
const ROOT = joinpath(@__DIR__, "..", "..")
cd(ROOT)
include(joinpath(ROOT, "src", "juliaDEC.jl")); include(joinpath(ROOT, "src", "hodge_stars.jl"))
include(get(ENV, "SYMSTAR", joinpath(ROOT, "symmetric_star.jl")))
const OUT = joinpath(ROOT, "report")

const INK = colorant"#2b2b29"; const MUTED = colorant"#8a8a85"; const GRID = colorant"#e6e5e1"
const BLUE = colorant"#2a78d6"; const ORANGE = colorant"#eb6834"; const AQUA = colorant"#1baf7a"
const DIV = cgrad([colorant"#1c5cab", colorant"#f0efec", colorant"#c23b3a"])
set_theme!(Theme(fontsize = 13, textcolor = INK,
    Axis = (xgridcolor = GRID, ygridcolor = GRID, spinewidth = 0.8, leftspinecolor = MUTED,
            bottomspinecolor = MUTED, topspinevisible = false, rightspinevisible = false,
            xtickcolor = MUTED, ytickcolor = MUTED, titlefont = :regular, titlesize = 13),
    Legend = (framevisible = false, labelsize = 12)))
savefig(name, f) = (CairoMakie.save(joinpath(OUT, name), f, px_per_unit = 2); println("  wrote ", name))
meanh(m) = sum(edge_lengths(m, true)) / m.edges.n
diag_star(m) = spdiagm(0 => edge_lengths(m, false) ./ edge_lengths(m, true))

# ============================================================================
# Fig A: exact plane-wave eigenvalues of L = d0' H d0 on the regular lattice
# ============================================================================
# On a Bravais lattice p_xi(x_c) = exp(i xi . x_c) is an eigenvector, with eigenvalue
# lambda(xi) = (d0 p)^* H (d0 p) / (p^* S0 p).  r(xi) = Re lambda(xi) / |xi|^2 (exact value 1).
function symbol(m, H, d0, s0, j, k)
    p = ComplexF64[exp(2pi*im*(j*m.cells.position[c][1] + k*m.cells.position[c][2])) for c in 1:m.cells.n]
    z = d0 * p
    return real(dot(z, H * z)) / real(dot(p, s0 .* p)) / (4pi^2 * (j^2 + k^2))
end
let m = VoronoiMesh("meshes/mesh4_regular.nc")
    d0 = sparse(Float64.(d0_matrix_primal(m))); s0 = Float64.(m.cells.area); h = meanh(m)
    stars = [("diagonal", diag_star(m), MUTED, :circle),
             ("lift [P₁]² (13 edges)", hodge_star_lift_matrix(m; only_linear = true), ORANGE, :utriangle),
             ("lift [P₂]² (19 edges)", hodge_star_lift_matrix(m; only_linear = false), BLUE, :rect)]
    n = 14
    f = Figure(size = (1200, 430))
    ax = Axis(f[1, 1], xlabel = "ν = |ξ| h / π   (1 ≈ grid scale)", ylabel = "r(ξ) = λ(ξ) / |ξ|²",
              title = "(a) exact eigenvalues along ξ = 2π(j, 0)")
    hlines!(ax, [0.0], color = INK, linewidth = 1); hlines!(ax, [1.0], color = MUTED, linestyle = :dot)
    for (lab, H, c, mk) in stars
        js = 1:n; nu = [2j*h for j in js]; r = [symbol(m, H, d0, s0, j, 0) for j in js]
        lines!(ax, nu, r, color = c, linewidth = 2.5, label = lab)
        scatter!(ax, nu, r, color = c, marker = mk, markersize = 9, strokecolor = :white, strokewidth = 1)
    end
    axislegend(ax, position = :rt)
    pts = [(j, k) for j in -n:n, k in -n:n if (j, k) != (0, 0) && j^2 + k^2 <= n^2]
    for (i, (lab, H)) in enumerate([("(b) lift [P₁]²", stars[2][2]), ("(c) lift [P₂]²", stars[3][2])])
        R = fill(NaN, 2n + 1, 2n + 1)
        for (j, k) in pts; R[j + n + 1, k + n + 1] = symbol(m, H, d0, s0, j, k); end
        rmin = minimum(filter(!isnan, R)); nneg = count(x -> !isnan(x) && x < 0, R)
        @printf("  %s: min r = %+.4f, negative at %d of %d wavevectors\n", lab, rmin, nneg, length(pts))
        ax2 = Axis(f[1, i + 1], aspect = DataAspect(), xlabel = "j", ylabel = i == 1 ? "k" : "",
                   title = @sprintf("%s:  min r = %+.3f,  r < 0 on %d%% of ξ shown", lab, rmin, round(Int, 100nneg/length(pts))))
        heatmap!(ax2, -n:n, -n:n, R, colormap = DIV, colorrange = (-1, 1), nan_color = :transparent)
        contour!(ax2, -n:n, -n:n, replace(R, NaN => 1.0), levels = [0.0], color = INK, linewidth = 2)
    end
    Colorbar(f[1, 4], colormap = DIV, limits = (-1, 1), label = "r(ξ)")
    Label(f[2, :], "Regular lattice mesh4_regular. For each ξ = 2π(j,k), r(ξ) is the exact eigenvalue of Lp = λS₀p belonging to the plane wave, " *
          "divided by the exact value |ξ|².\nBlack contour: r = 0. Inside it the [P₁]² lift has negative eigenvalues; the [P₂]² lift has none.",
          fontsize = 11, color = MUTED, halign = :left, justification = :left, tellwidth = false)
    colsize!(f.layout, 1, Relative(0.36))
    savefig("fig_lift_symbol.png", f)
end

# ============================================================================
# Fig B: the sandwich's flux defect and the mixed error bound
# ============================================================================
pex(x)  = sin(2pi*x[1])*sin(2pi*x[2]) + 0.3cos(2pi*(x[1] - 2x[2]))
gpex(x) = [2pi*cos(2pi*x[1])*sin(2pi*x[2]) - 0.6pi*sin(2pi*(x[1]-2x[2])), 2pi*sin(2pi*x[1])*cos(2pi*x[2]) + 1.2pi*sin(2pi*(x[1]-2x[2]))]
lapp(x) = -8pi^2*sin(2pi*x[1])*sin(2pi*x[2]) - 6pi^2*cos(2pi*(x[1]-2x[2]))
uex(x)  = (g = gpex(x); [g[2], -g[1]])
function mixed_data(m, H)
    D1 = sparse(Float64.(d1_matrix_primal(m))); nE, nT = m.edges.n, m.vertices.n
    w = edge_lengths(m, false) ./ edge_lengths(m, true)
    fex = reduce_primal_edges(m, uex).values; Su = reduce_dual_edges(m, star_field(uex)).values
    phi = discretize_scalar_field_triangles_ho(m, x -> -lapp(x)).values; phi .-= sum(phi)/nT
    F = svd(Matrix(D1); full = true); r = count(>(1e-10 * F.S[1]), F.S); Z = F.V[:, r+1:end]
    ZWZ = Symmetric(Z' * (w .* Z))
    eps = H * fex - Su; ref = sqrt(sum(Su.^2 ./ w))
    g = Z' * eps
    etot = sqrt(sum(eps.^2 ./ w)) / ref; edf = sqrt(dot(g, ZWZ \ g)) / ref
    c = minimum(eigvals(Symmetric(Z' * Matrix(0.5 .* (H + H')) * Z), ZWZ))
    K = [sparse(H) -D1'; D1 spzeros(nT, nT)]; keep = setdiff(1:nE+nT, [nE+1])
    sol = zeros(nE+nT); sol[keep] = K[keep, keep] \ [zeros(nE); phi][keep]
    nf = sqrt(sum(w .* fex.^2))
    ef = sqrt(sum(w .* (sol[1:nE] .- fex).^2)) / nf
    bound = edf * ref / (c * nf)
    return (etot = etot, edf = edf, ef = ef, bound = bound, c = c)
end
let f = Figure(size = (1150, 440))
    fams = [("regular", ["mesh2_regular", "mesh3_regular", "mesh4_regular"], BLUE),
            ("SCVT", ["mesh2", "mesh3", "mesh4"], ORANGE)]
    ax1 = Axis(f[1, 1], xscale = log10, yscale = log10, xlabel = "h", xreversed = true,
               ylabel = "relative size", title = "(a) sandwich [P₂]²: flux defect ε and its divergence-free part")
    ax2 = Axis(f[1, 2], xscale = log10, yscale = log10, xlabel = "bound  c⁻¹ ‖ε‖_* / ‖f‖", ylabel = "Darcy flux error",
               title = "(b) the bound of the mixed error identity is sharp")
    allv = Float64[]
    for (fam, mns, col) in fams
        hs = Float64[]; tot = Float64[]; df = Float64[]
        for mn in mns
            m = VoronoiMesh("meshes/$mn.nc"); push!(hs, meanh(m))
            Hl = hodge_star_lift_matrix(m; only_linear = false)
            stars = [("sandwich", hodge_star_sandwich_matrix(m; only_linear = false), :circle),
                     ("symmetric", hodge_star_symmetric_matrix(m; only_linear = false, level = 4, consistency = 2,
                                   hard_degree = (fam == "regular" ? 2 : 1), mu_soft = 1e-4, Href = Hl).H +
                                   polynomial_blind_stabilisation(m; only_linear = false, level = 3), :utriangle),
                     ("lift", Hl, :rect)]
            for (nm, H, mk) in stars
                d = mixed_data(m, H)
                @printf("  %-14s %-10s eps %.2e  eps_df %.2e  flux %.2e  bound %.2e  c %.2f\n", mn, nm, d.etot, d.edf, d.ef, d.bound, d.c)
                if nm == "sandwich"; push!(tot, d.etot); push!(df, d.edf); end
                scatter!(ax2, [d.bound], [d.ef], color = col, marker = mk, markersize = 12, strokecolor = :white, strokewidth = 1)
                push!(allv, d.bound, d.ef)
            end
        end
        lines!(ax1, hs, tot, color = col, linewidth = 2.5, label = "$fam: ‖ε‖ (all of it)")
        scatter!(ax1, hs, tot, color = col, markersize = 9, strokecolor = :white, strokewidth = 1)
        lines!(ax1, hs, df, color = col, linewidth = 2.5, linestyle = :dash, label = "$fam: ‖ε‖_* (divergence-free part)")
        scatter!(ax1, hs, df, color = col, marker = :rect, markersize = 9, strokecolor = :white, strokewidth = 1)
    end
    lo, hi = minimum(allv) / 2, maximum(allv) * 2
    lines!(ax2, [lo, hi], [lo, hi], color = MUTED, linestyle = :dash)
    text!(ax2, hi / 3, hi / 3, text = "error = bound", color = MUTED, fontsize = 11, align = (:right, :bottom), offset = (-4, 4))
    axislegend(ax1, position = :lb, labelsize = 10)
    # legend for (b): colours = families, markers = stars
    elems = [MarkerElement(marker = :circle, color = INK, markersize = 11), MarkerElement(marker = :utriangle, color = INK, markersize = 11),
             MarkerElement(marker = :rect, color = INK, markersize = 11), PolyElement(color = BLUE), PolyElement(color = ORANGE)]
    Legend(f[1, 3], elems, ["sandwich [P₂]²", "symmetric star", "lift [P₂]²", "regular meshes", "SCVT meshes"])
    Label(f[2, :], "Mixed Darcy problem, meshes 2–4. (a) ‖ε‖: weighted norm of the flux error Hℛu − ℛ̃⋆u; ‖ε‖_*: the part of it seen by discretely " *
          "divergence-free circulations. Both relative to ‖ℛ̃⋆u‖.\n(b) Each point is one star on one mesh: measured flux error against the bound of " *
          "Proposition 8.6. Points on or just below the dashed line mean the bound is sharp.",
          fontsize = 11, color = MUTED, halign = :left, justification = :left, tellwidth = false)
    savefig("fig_sandwich_flux.png", f)
end
println("done")
