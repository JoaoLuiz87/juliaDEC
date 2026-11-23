# ss_wellcentred.jl -- are the juliaDEC SCVT meshes well-centred (Assumption 2.1 of the guide)?
include(joinpath(@__DIR__, "..", "..", "SymStar.jl")); using .SymStar
include(joinpath(@__DIR__, "ss_mesh.jl")); using LinearAlgebra, Printf
for mn in ("mesh3", "mesh4", "mesh5")
    X, T, period, h = scvt_mesh(mn); g = edge_geometry(X, T, period); E = length(g.le)
    n = hcat(-g.t[:, 2], g.t[:, 1])
    s1 = [dot(g.c1[e, :] - g.me[e, :], n[e, :]) for e in 1:E]; s2 = [dot(g.c2[e, :] - g.me[e, :], n[e, :]) for e in 1:E]
    bad = count(e -> !(s1[e] < 0 < s2[e]), 1:E)
    # obtuse triangles
    angs = Float64[]
    for tri in T
        P = [SymStar.unwrapv(X[p], X[tri[1]], period) for p in tri]
        for k in 1:3
            a = P[k]; b = P[mod1(k + 1, 3)]; c = P[mod1(k + 2, 3)]
            push!(angs, acosd(clamp(dot(b - a, c - a) / (norm(b - a) * norm(c - a)), -1, 1)))
        end
    end
    @printf("%s: %d edges, %d not well-centred (%.2f%%), max angle %.1f°, triangles with an angle > 90°: %d of %d, max|δ|/|e| %.3f, min |ẽ|/|e| %.3f\n",
            mn, E, bad, 100bad / E, maximum(angs), count(>(90), reshape(angs, 3, :)' * [0, 0, 0] .+ maximum(reshape(angs, 3, :), dims = 1)'), length(T),
            maximum(abs.(g.delta) ./ g.le), minimum(g.lde ./ g.le))
end
