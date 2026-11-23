# ============================================================================
#  gs_hypotheses.jl -- direct checks of the hypotheses and identities used in galerkin_star_paper.tex
#
#      julia scripts/archive/gs_hypotheses.jl          (writes output/gs_data/hypotheses.txt)
#
#  For SCVT meshes 2-5:
#    offset      max_e |δ_e| / |e| and its mean, δ_e = (midpoint of ẽ - m_e) · n_e   (Example 1)
#    (A)         ‖Σ_e |e||ẽ| t_e⊗t_e / |Ω| - I‖ and |Σ_e |e||ẽ| / (2|Ω|) - 1|          (Lemma "identities")
#    (B)         max |Σ_e |e||ẽ| δ_e t⊗t⊗n| / Σ_e |e||ẽ||δ_e|                        (Lemma "identities")
#    (A4)        min_T σ_6(P_T)/σ_1(P_T)                                             (full column rank)
#    (U)         min over pairs of triangles sharing an edge of σ_6/σ_1 of the rows of P_T on S_T ∩ S_T',
#                and the smallest |S_T ∩ S_T'|                                        (unisolvent overlaps)
# ============================================================================
using VoronoiMeshes, TensorsLite, TensorsLiteGeometry, SmallCollections, LinearAlgebra, NCDatasets
using StaticArrays, SparseArrays, Printf
Base.similar(bc::Base.Broadcast.Broadcasted, ::Type{Union{}}) = Array{Union{}}(undef, length.(axes(bc)))
const ROOT = joinpath(@__DIR__, "..", ".."); cd(ROOT)
include(joinpath(ROOT, "src", "juliaDEC.jl")); include(joinpath(ROOT, "src", "hodge_stars.jl")); include(joinpath(ROOT, "src", "galerkin_star.jl"))
const OUT = joinpath(@__DIR__, "..", "..", "output", "gs_data"); mkpath(OUT)
lines = String[]
logp(s) = (println(s); flush(stdout); push!(lines, s))

logp(@sprintf("%-6s %6s | %9s %9s | %9s %9s | %9s | %9s | %9s %5s", "mesh", "E", "max δ/|e|", "mean", "(A) dev", "trace", "(B) rel",
              "A4 cond", "U cond", "min#"))
for mn in ("mesh2", "mesh3", "mesh4", "mesh5")
    mesh = VoronoiMesh("meshes/$mn.nc")
    E = mesh.edges.n; nT = mesh.vertices.n; area = mesh.x_period * mesh.y_period
    s = sum(edge_length(mesh, e, true) for e in 1:E) / E
    ed = [gs_edge(mesh, e, s) for e in 1:E]
    # offsets and the two identities
    TA = zeros(2, 2); tr2 = 0.0; TB = zeros(2, 2, 2); wabs = 0.0; dmax = 0.0; dsum = 0.0
    for d in ed
        t = (d.p2 - d.p1) / d.le; n = [-t[2], t[1]]
        @assert norm((d.v2 - d.v1) - d.lde * n) < 1e-10 * d.lde            # dual orientation v2 - v1 = |ẽ| R t
        δ = dot((d.v1 + d.v2) / 2 - d.m, n)
        dmax = max(dmax, abs(δ) / d.le); dsum += abs(δ) / d.le
        TA .+= d.le * d.lde .* (t * t'); tr2 += d.le * d.lde
        w = d.le * d.lde * δ; wabs += abs(w)
        for a in 1:2, b in 1:2, c in 1:2
            TB[a, b, c] += w * t[a] * t[b] * n[c]
        end
    end
    devA = opnorm(TA / area - I); devT = abs(tr2 / (2area) - 1); relB = maximum(abs, TB) / wabs
    # (A4) and (U)
    cells = [gs_cell(mesh, t, s) for t in 1:nT]
    a4 = minimum(begin sv = svdvals(C.P); sv[6] / sv[1] end for C in cells)
    ucond = Inf; umin = typemax(Int)
    for e in 1:E
        t1, t2 = Int.(collect(mesh.edges.vertices[e]))                     # the two triangles of e
        C = cells[t1]; common = intersect(C.S, cells[t2].S)
        rows = [findfirst(==(f), C.S) for f in common]
        sv = svdvals(C.P[rows, :]); ucond = min(ucond, sv[6] / sv[1]); umin = min(umin, length(common))
    end
    logp(@sprintf("%-6s %6d | %9.4f %9.4f | %9.1e %9.1e | %9.1e | %9.3f | %9.3f %5d", mn, E, dmax, dsum / E, devA, devT, relB,
                  a4, ucond, umin))
end
write(joinpath(OUT, "hypotheses.txt"), join(lines, "\n") * "\n")
