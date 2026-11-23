# ============================================================================
#  commuting_note_figures.jl -- fig_commuting.png for commuting_symmetric_note.tex
#
#      julia -t 4 scripts/stars/commuting_note_figures.jl
#
#  Reads gs_data/commuting_check.txt (written by commuting_check.jl) and recomputes two cheap things:
#    * the skew s_h(u2,d) on the SCVT and lattice meshes 2-5,
#    * λmin(Hd^-1/2 H Hd^-1/2) of the symmetric [P1]^2 star on mesh5, by bisection with sparse Cholesky.
#  Writes fig_commuting.png and gs_data/commuting_note_extra.txt.
# ============================================================================
ENV["PARTS"] = ""                                   # load the functions of commuting_check.jl only
include(joinpath(@__DIR__, "commuting_check.jl"))
using CairoMakie

txt = readlines(joinpath(OUT, "commuting_check.txt"))
hof = Dict("mesh2" => 1 / 8, "mesh3" => 1 / 16, "mesh4" => 1 / 32, "mesh5" => 1 / 64)
num(l, key) = parse(Float64, match(Regex(key * raw"\s+([-+0-9.e]+)"), l).captures[1])

# Part 1: the two stars of hodge_stars.jl
p1 = Dict{Tuple{String,String},Tuple{Float64,Float64}}()
let cur = ""
    for l in txt
        startswith(l, "Part 3") && break
        m = match(r"^(mesh\d)\s+\(h", l); m === nothing || (cur = m.captures[1])
        for name in ("DEC diagonal", "lift  [P1]^2", "sandw [P1]^2")
            occursin(name * "  asym", l) && (p1[(cur, name)] = (num(l, raw"r\(d\)"), num(l, "tau")))
        end
    end
end
# Parts 3 and 4: the symmetric corrections
p3 = Dict{Tuple{Int,Int,String},NamedTuple}()
for l in txt
    m = match(r"k=(\d) ring (\d)\s+(mesh\d)\s", l)
    m === nothing && continue
    p3[(parse(Int, m[1]), parse(Int, m[2]), m[3])] = (C = num(l, raw"\|C\|/\|Hd\|"), rd = num(l, raw"r\(d\)"), τ = num(l, "tau"))
end

# skew on SCVT and lattice meshes
u2(x) = [sin(TP * x[1]) * cos(TP * x[2]), -cos(TP * x[1]) * sin(TP * x[2])]
extra = String[]
skew = Dict{String,Vector{Tuple{Float64,Float64}}}()
for fam in ("", "_regular")
    skew[fam] = Tuple{Float64,Float64}[]
    for k in 2:5
        mesh = VoronoiMesh("meshes/mesh$(k)$(fam).nc")
        h = sqrt(mesh.x_period * mesh.y_period / mesh.cells.n)
        α(v) = reduce_primal_edges(mesh, v).values
        β(v) = reduce_dual_edges(mesh, star_field(v)).values
        s = dot(α(u2), β(fields.d)) - dot(α(fields.d), β(u2))
        push!(skew[fam], (h, abs(s)))
        push!(extra, @sprintf("skew  mesh%d%-8s h = %.5f  s_h(u2,d) = %+.4e  s_h/h^2 = %+.4f", k, fam, h, s, s / h^2))
    end
end

# λmin of the symmetric [P1]^2 star (3-ring) on meshes 2-5, by bisection: λ > t  <=>  Dm H Dm - tI is SPD
function lambda_min_bisect(H, Hd; lo = -1.0, hi = 1.0, tol = 1e-4)
    Dm = Diagonal(1 ./ sqrt.(diag(Hd))); S = sparse(Symmetric(Dm * H * Dm))
    isposdef(t) = issuccess(cholesky(S - t * I; check = false))
    while hi - lo > tol
        t = (lo + hi) / 2
        isposdef(t) ? (lo = t) : (hi = t)
    end
    return (lo + hi) / 2
end
for mn in ("mesh2", "mesh3", "mesh4", "mesh5")
    mesh = VoronoiMesh("meshes/$mn.nc")
    H, info = symmetric_correction(mesh, 1, 3)
    μ = lambda_min_bisect(H, hdiag(mesh))
    push!(extra, @sprintf("lambda_min  [P1]^2 3-ring  %s  = %.4f   (|C|/|Hd| = %.3e, res = %.1e)", mn, μ, info.cmax, info.res))
end
foreach(println, extra)
write(joinpath(OUT, "commuting_note_extra.txt"), join(extra, "\n") * "\n")

# ---------------------------------------------------------------- figure
ms = ("mesh2", "mesh3", "mesh4", "mesh5")
fig = Figure(size = (1300, 1000), fontsize = 15)
xt = ([1 / 8, 1 / 16, 1 / 32, 1 / 64], ["1/8", "1/16", "1/32", "1/64"])
style = [(:p1, "DEC diagonal", :black, :circle, "DEC (diagonal)"),
         (:p1, "lift  [P1]^2", :darkorange, :utriangle, "lift [P1]² (not symmetric)"),
         (:p1, "sandw [P1]^2", :purple, :xcross, "sandwich [P1]² (symmetric, smeared dual)"),
         (:p3, (1, 3), :steelblue, :diamond, "symmetric, exact on [P1]² (3-ring)"),
         (:p3, (2, 4), :firebrick, :rect, "symmetric, exact on [P2]² (4-ring)")]
function series(src, key, field)
    hs, vs = Float64[], Float64[]
    for m in ms
        if src == :p1
            haskey(p1, (m, key)) || continue
            push!(hs, hof[m]); push!(vs, field == :rd ? p1[(m, key)][1] : p1[(m, key)][2])
        else
            haskey(p3, (key..., m)) || continue
            push!(hs, hof[m]); push!(vs, getfield(p3[(key..., m)], field))
        end
    end
    return hs, vs
end
axA = Axis(fig[1, 1]; xscale = log10, yscale = log10, xreversed = true, xticks = xt, xlabel = "h", ylabel = "max_e |ε_e| / (|ẽ| ‖u‖)",
           title = "(a) Hodge error on the smooth field d")
axB = Axis(fig[1, 2]; xscale = log10, yscale = log10, xreversed = true, xticks = xt, xlabel = "h", ylabel = "max_v |τ_v| / max|Δp|",
           title = "(b) Laplacian truncation error τ")
for (src, key, c, mk, _) in style
    scatterlines!(axA, series(src, key, :rd)...; color = c, marker = mk, markersize = 12)
    scatterlines!(axB, series(src, key, :τ)...; color = c, marker = mk, markersize = 12)
end
lines!(axA, [1 / 16, 1 / 64], [1.5e-3, 1.5e-3 / 16]; color = :gray, linewidth = 2); text!(axA, 1 / 45, 1.0e-4; text = "h²", color = :gray)
lines!(axA, [1 / 16, 1 / 64], [3e-2, 3e-2 / 4]; color = :gray, linewidth = 2, linestyle = :dot); text!(axA, 1 / 30, 2.2e-2; text = "h", color = :gray)
lines!(axB, [1 / 16, 1 / 64], [2e-2, 2e-2 / 16]; color = :gray, linewidth = 2); text!(axB, 1 / 30, 2.5e-3; text = "h²", color = :gray)
lines!(axB, [1 / 16, 1 / 64], [1e-1, 1e-1 / 4]; color = :gray, linewidth = 2, linestyle = :dot); text!(axB, 1 / 30, 7e-2; text = "h", color = :gray)

axC = Axis(fig[2, 1]; xscale = log10, yscale = log10, xreversed = true, xticks = xt, xlabel = "h", ylabel = "|s_h(u₂, d)|", yticks = ([1e-12, 1e-9, 1e-6, 1e-3], ["10⁻¹²", "10⁻⁹", "10⁻⁶", "10⁻³"]),
           title = "(c) skew of the primal–dual pairing")
scatterlines!(axC, first.(skew[""]), last.(skew[""]); color = :black, marker = :circle, markersize = 12, label = "SCVT meshes 2–5")
scatterlines!(axC, first.(skew["_regular"]), last.(skew["_regular"]); color = :gray, marker = :rect, markersize = 12,
              label = "lattice meshes 2–5")
hh = [1 / 7, 1 / 70]; lines!(axC, hh, 0.147 .* hh .^ 2; color = :firebrick, linestyle = :dash, label = "0.147 h²")
axislegend(axC; position = :rc, framevisible = false)

axD = Axis(fig[2, 2]; xscale = log10, yscale = log10, xreversed = true, xticks = xt, xlabel = "h", ylabel = "max|C_ij| / max(H_DEC)",
           title = "(d) size of the symmetric correction C")
for (key, c, mk, lbl) in (((1, 3), :steelblue, :diamond, "[P1]², 3-ring"), ((1, 2), :steelblue, :circle, "[P1]², 2-ring"),
                          ((2, 4), :firebrick, :rect, "[P2]², 4-ring"), ((2, 3), :firebrick, :utriangle, "[P2]², 3-ring"))
    hs, vs = series(:p3, key, :C)
    scatterlines!(axD, hs, vs; color = c, marker = mk, markersize = 12, label = lbl,
                  linestyle = key[2] == (key[1] == 1 ? 3 : 4) ? :solid : :dash)
end
axislegend(axD; position = :lt, framevisible = false)

Legend(fig[3, 1:2], [MarkerElement(color = c, marker = mk, markersize = 12) for (_, _, c, mk, _) in style],
       [n for (_, _, _, _, n) in style]; orientation = :horizontal, nbanks = 2, framevisible = false, labelsize = 14)
CairoMakie.save(joinpath(@__DIR__, "..", "..", "output", "figures", "fig_commuting.png"), fig; px_per_unit = 2)
println("wrote fig_commuting.png")
