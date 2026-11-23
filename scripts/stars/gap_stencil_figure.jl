# ============================================================================
#  gap_stencil_figure.jl -- fig_gap_stencils.png of paper/hodge_gap.tex (Section 2): the four stencils of one edge of SCVT mesh 3.
#
#      julia scripts/stars/gap_stencil_figure.jl
#
#  (a) 13-edge      e, its two triangles, and the triangles across their four outer edges          ring(mesh, e, :t2)
#  (b) 3-ring       every edge at the four vertices of the two triangles of e (rows of H_P1, H_P2)   ring(mesh, e, 3)
#                   light: the entries that the symmetrised pattern of H_P1 adds (j with e ∈ S_j)
#  (c) 3 layers     (a) plus the next layer of triangles                                              ring(mesh, e, :t3)
#  (d) 4-ring       the union of the 13-edge stencils of the 13 edges of (a)                         ring(mesh, e, 4)
# ============================================================================
include(joinpath(@__DIR__, "three_stars.jl"))
using CairoMakie
let mesh = VoronoiMesh("meshes/mesh3.nc"), e = 17
    xe = mesh.edges.position[e]; x0 = _pt(xe); s = edge_lengths(mesh, true)[e]
    seg(j) = edge_endpoints(mesh, j, true; about = xe)
    near = [j for j in 1:mesh.edges.n if norm(_pt(closest(xe, mesh.edges.position[j], 1.0, 1.0)) .- x0) < 7s]
    q1, q2 = edge_endpoints(mesh, e, false; about = xe)
    verts(S) = collect(values(Dict(round.(p; digits = 10) => p for j in S for p in seg(j))))
    L = 3.1s
    f = Figure(size = (1900, 560))
    function panel(pos, title, S, col; extra = Int[])
        ax = Axis(pos; aspect = DataAspect(), title = title, titlesize = 15); hidedecorations!(ax)
        for j in near
            a, b = seg(j); lines!(ax, [a[1], b[1]], [a[2], b[2]]; color = (:gray, 0.3), linewidth = 0.8)
        end
        for j in extra
            a, b = seg(j); lines!(ax, [a[1], b[1]], [a[2], b[2]]; color = (col, 0.35), linewidth = 3.0, linestyle = :dash)
        end
        for j in S
            a, b = seg(j); lines!(ax, [a[1], b[1]], [a[2], b[2]]; color = col, linewidth = 3.5)
        end
        V = verts(S); scatter!(ax, [p[1] for p in V], [p[2] for p in V]; color = :black, markersize = 8)
        a, b = seg(e); lines!(ax, [a[1], b[1]], [a[2], b[2]]; color = :black, linewidth = 4)
        lines!(ax, [q1[1], q2[1]], [q1[2], q2[2]]; color = :black, linewidth = 2, linestyle = :dash)
        xlims!(ax, x0[1] - L, x0[1] + L); ylims!(ax, x0[2] - L, x0[2] + L)
    end
    S2, S3, St3, S4 = ring(mesh, e, :t2), ring(mesh, e, 3), ring(mesh, e, :t3), ring(mesh, e, 4)
    # symmetrised 3-ring pattern of H_P1: row e also couples to j when e is in the stencil of row j
    cand = [j for j in near if j ∉ S3]
    sym_extra = [j for j in cand if e in ring(mesh, j, 3)]
    nV(S) = length(verts(S))
    panel(f[1, 1], "(a) 13-edge: e, its two triangles and the\nfour triangles across their outer edges\n$(length(S2)) edges, $(nV(S2)) vertices", S2, colorant"#7a7a7a")
    panel(f[1, 2], "(b) 3-ring: every edge at the four vertices\nof the two triangles of e (rows of H_P1, H_P2)\n$(length(S3)) edges, $(nV(S3)) vertices; dashed: added by symmetry in H_P1", S3,
          colorant"#c23b3a"; extra = sym_extra)
    panel(f[1, 3], "(c) 3 triangle layers: (a) plus the\nnext layer of triangles\n$(length(St3)) edges, $(nV(St3)) vertices", St3, colorant"#2b6cb0")
    panel(f[1, 4], "(d) 4-ring: the union of the 13-edge\nstencils of the 13 edges of (a)\n$(length(S4)) edges, $(nV(S4)) vertices", S4, colorant"#2f8f4e")
    println("edge $e of mesh 3: |S| = $(length(S2)), $(length(S3)) (+$(length(sym_extra)) by symmetry), $(length(St3)), $(length(S4))")
    CairoMakie.save(joinpath(@__DIR__, "..", "..", "output", "figures", "fig_gap_stencils.png"), f; px_per_unit = 2)
end
