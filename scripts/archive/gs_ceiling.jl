# ============================================================================
#  gs_ceiling.jl -- how accurate can ANY symmetric star be on the juliaDEC SCVT meshes?
#
#      julia scripts/archive/gs_ceiling.jl          (writes output/gs_data/ceiling.txt)
#
#  For a symmetric H and smooth periodic u, v:   s_h(u,v) := α(u)ᵀβ(v) - α(v)ᵀβ(u) = α(v)ᵀε(u) - α(u)ᵀε(v),
#  with ε = Hα - β.  Since |α_e(u)| ≤ |e| ‖u‖∞ and Σ_e |e||ẽ| = 2|Ω|,
#        max( r(u), r(v) ) ≥ |s_h(u,v)| / (2|Ω| (‖u‖∞ + ‖v‖∞)),     r(u) = max_e |ε_e(u)| / |ẽ_e|.
#  So if s_h ~ h², no symmetric matrix H (local or not, bounded or not) has o(h²) relative error per edge.
#  Part 1: s_h for pairs of smooth fields on meshes 2-5, the lower bound, and the errors of DEC and of the Galerkin star.
#  Part 2: the stabilisation weight τ of the Galerkin star against its error constant (meshes 3, 4).
# ============================================================================
using VoronoiMeshes, TensorsLite, TensorsLiteGeometry, SmallCollections, LinearAlgebra, NCDatasets
using StaticArrays, SparseArrays, Printf
Base.similar(bc::Base.Broadcast.Broadcasted, ::Type{Union{}}) = Array{Union{}}(undef, length.(axes(bc)))
const ROOT = joinpath(@__DIR__, "..", ".."); cd(ROOT)
include(joinpath(ROOT, "src", "juliaDEC.jl")); include(joinpath(ROOT, "src", "hodge_stars.jl")); include(joinpath(ROOT, "src", "galerkin_star.jl"))
const OUT = joinpath(@__DIR__, "..", "..", "output", "gs_data"); mkpath(OUT)
lines = String[]
logp(s) = (println(s); flush(stdout); push!(lines, s))

fields = (u1 = x -> [sin(2pi * x[1]) * cos(2pi * x[2]), cos(2pi * x[1]) * sin(2pi * x[2])],
          u2 = x -> [sin(2pi * x[1]) * cos(2pi * x[2]), -cos(2pi * x[1]) * sin(2pi * x[2])],
          a  = x -> [sin(2pi * x[2]), cos(2pi * x[1])],
          b  = x -> [cos(2pi * (x[1] + x[2])), sin(2pi * (x[1] - 2x[2]))],
          c  = x -> [cos(2pi * x[1]) + sin(2pi * x[2]), cos(2pi * x[2])],
          d  = x -> [sin(2pi * x[1]) + cos(2pi * x[2]), sin(2pi * x[2]) + 0.5cos(2pi * x[1])])
names = collect(keys(fields))
grid = [[i / 400, j / 400] for i in 0:399 for j in 0:399]
sup = Dict(k => maximum(norm(fields[k](x)) for x in grid) for k in names)
pairs = [(names[i], names[j]) for i in 1:length(names) for j in i+1:length(names)]

logp("Part 1: skew of the primal-dual pairing and the lower bound for every symmetric star")
res = Dict{String,Any}()
for mn in ("mesh2", "mesh3", "mesh4", "mesh5")
    mesh = VoronoiMesh("meshes/$mn.nc"); area = mesh.x_period * mesh.y_period
    h = sqrt(area / mesh.cells.n)
    α = Dict(k => reduce_primal_edges(mesh, fields[k]).values for k in names)
    β = Dict(k => reduce_dual_edges(mesh, star_field(fields[k])).values for k in names)
    sh = Dict(p => dot(α[p[1]], β[p[2]]) - dot(α[p[2]], β[p[1]]) for p in pairs)
    lb = Dict(p => abs(sh[p]) / (2area * (sup[p[1]] + sup[p[2]])) for p in pairs)
    Hd = spdiagm(edge_lengths(mesh, false) ./ edge_lengths(mesh, true))
    HG, _ = galerkin_star_matrix(mesh; verbose = false)
    eD = Dict(k => gs_hodge_error(mesh, Hd, fields[k])[1] for k in names)
    eG = Dict(k => gs_hodge_error(mesh, HG, fields[k])[1] for k in names)
    res[mn] = (h = h, sh = sh, lb = lb, eD = eD, eG = eG)
    logp(@sprintf("%s (h = %.4f)", mn, h))
    for p in pairs
        logp(@sprintf("   s_h(%-2s,%-2s) = %+.3e   bound %.2e   max r: DEC %.2e  Galerkin %.2e", p[1], p[2], sh[p], lb[p],
                      max(eD[p[1]], eD[p[2]]), max(eG[p[1]], eG[p[2]])))
    end
end
logp("\norders of |s_h| between consecutive meshes (h halves):")
ms = ("mesh2", "mesh3", "mesh4", "mesh5")
for p in pairs
    v = [abs(res[m].sh[p]) for m in ms]
    logp(@sprintf("   (%-2s,%-2s): %s   |s_h|/h² = %s", p[1], p[2], join([@sprintf("%5.2f", log2(v[i] / v[i+1])) for i in 1:3], " "),
                  join([@sprintf("%.3f", v[i] / res[ms[i]].h^2) for i in 1:4], " ")))
end
best = pairs[argmax([res["mesh5"].lb[p] for p in pairs])]
logp(@sprintf("\nlargest bound at mesh5: pair (%s,%s); bound / h² = %s", best[1], best[2],
              join([@sprintf("%.3f", res[m].lb[best] / res[m].h^2) for m in ms], " ")))
logp(@sprintf("   Galerkin max r / h² over the pair = %s", join([@sprintf("%.2f", max(res[m].eG[best[1]], res[m].eG[best[2]]) / res[m].h^2) for m in ms], " ")))

logp("\nPart 2: stabilisation weight τ of the Galerkin star (max relative error, μ = λmin(Hd^-1/2 H Hd^-1/2))")
for mn in ("mesh3", "mesh4")
    mesh = VoronoiMesh("meshes/$mn.nc"); Hd = spdiagm(edge_lengths(mesh, false) ./ edge_lengths(mesh, true))
    for τ in (0.1, 0.3, 1.0, 3.0, 10.0)
        H, _ = galerkin_star_matrix(mesh; tau = τ, verbose = false)
        e1 = gs_hodge_error(mesh, H, fields.u1); e2 = gs_hodge_error(mesh, H, fields.u2)
        μ = gs_smallest_eigenvalue(H; D = Hd)
        logp(@sprintf("   %s τ = %5.1f   u1 max %.3e rms %.3e   u2 max %.3e rms %.3e   μ %.4f", mn, τ, e1..., e2..., μ))
    end
end
write(joinpath(OUT, "ceiling.txt"), join(lines, "\n") * "\n")
