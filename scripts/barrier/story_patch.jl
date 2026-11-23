# story_patch.jl -- the patch of the balance law on mesh 5 (hodge_story.tex, "Every symmetric star exact on linear fields is global")
#      julia -t 4 scripts/barrier/story_patch.jl     ->  fig_st_patch.png
#  Edges of a patch P coloured by their interior term ½|e|³|ẽ| nᵀZn (f = dilation, g = shear), and the coefficient pairs of H_P1
#  that cross ∂P.  (Same panel as (b) of story_global.jl, alone.)
ENV["LOC_MAIN"] = "0"; include(joinpath(@__DIR__, "locality_check.jl"))
include(joinpath(@__DIR__, "..", "stars", "stars_cache.jl"))
include(joinpath(@__DIR__, "..", "shared", "thesis_plot_helpers.jl"))
let mesh = VoronoiMesh("meshes/mesh5.nc"), S = cached_stars("mesh5"; mesh = VoronoiMesh("meshes/mesh5.nc"))
    v0 = argmin([norm(_pt(p) .- [0.5, 0.5]) for p in mesh.cells.position]); x0v = mesh.cells.position[v0]; x0 = _pt(x0v)
    geo = edge_geometry(mesh, x0v); ρ = 0.15
    Z = Zof(GBASIS["I"], GBASIS["D"])
    src = [0.5 * E.le^3 * E.ld * dot(E.n, Z * E.n) for E in geo]
    inP = [E.r < ρ for E in geo]
    Ht = sparse(S["H_P1"]'); cross = Tuple{Int,Int}[]
    for j in findall(inP), p in nzrange(Ht, j)
        i = rowvals(Ht)[p]; inP[i] || abs(nonzeros(Ht)[p]) < 1e-14 || push!(cross, (j, i))
    end
    f = Figure(size = (900, 820), fontsize = 17)
    ax = Axis(f[1, 1]; aspect = DataAspect()); hidedecorations!(ax)
    m = maximum(abs, src[inP])
    for (e, E) in enumerate(geo)
        E.r < 0.24 || continue
        a = E.m .- (E.le / 2) .* E.t; b = E.m .+ (E.le / 2) .* E.t
        if inP[e]
            lines!(ax, [a[1], b[1]], [a[2], b[2]]; color = src[e], colormap = DIVM, colorrange = (-m, m), linewidth = 2.4)
        else
            lines!(ax, [a[1], b[1]], [a[2], b[2]]; color = (:gray, 0.35), linewidth = 0.8)
        end
    end
    for (j, i) in cross
        lines!(ax, [geo[j].m[1], geo[i].m[1]], [geo[j].m[2], geo[i].m[2]]; color = (INK, 0.25), linewidth = 0.6)
    end
    θ = range(0, 2π; length = 200); lines!(ax, x0[1] .+ ρ .* cos.(θ), x0[2] .+ ρ .* sin.(θ); color = INK, linestyle = :dash, linewidth = 1.6)
    xlims!(ax, x0[1] - 0.24, x0[1] + 0.24); ylims!(ax, x0[2] - 0.24, x0[2] + 0.24)
    Colorbar(f[1, 2]; colormap = DIVM, limits = (-m, m), label = "interior term ½|e|³|ẽ| nᵀZn", width = 14,
             tickformat = v -> [@sprintf("%.0e", x) for x in v])
    CairoMakie.save(thesis_out("fig_st_patch.png"), f; px_per_unit = 2)
    println("patch figure: $(count(inP)) edges in P, $(length(cross)) crossing pairs")
end
