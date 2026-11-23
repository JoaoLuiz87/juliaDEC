# ============================================================================
#  gap_stencil_radius.jl -- Assumption (A3) of paper/hodge_gap.tex: every stencil, with the dual edges it involves, lies in
#  an open disc of radius 1/2 about m_e (the injectivity radius of the unit torus).  Largest such radius per mesh.
#
#      julia scripts/stars/gap_stencil_radius.jl          (writes output/ts_data/stencil_radius.txt)
#
#  rho_S(e) = max over the edges j of the stencil S_e (and the Voronoi edge of e) of the distance from m_e to their endpoints,
#  each endpoint taken at its periodic image closest to m_e.
# ============================================================================
include(joinpath(@__DIR__, "three_stars.jl"))
using Printf

function stencil_radius(mesh, S)
    r = 0.0
    for e in 1:mesh.edges.n
        xe = mesh.edges.position[e]; m = _pt(xe)
        for j in S(e), primal in (true, false)
            (j != e && !primal) && continue                     # dual edge of e only
            a, b = edge_endpoints(mesh, j, primal; about = xe)
            r = max(r, norm(a .- m), norm(b .- m))
        end
    end
    return r
end

"rows of D0' H D0 with H on the 3-ring: largest distance from x_v to the endpoints of the stencils of the edges at v"
function laplacian_radius(mesh)
    d0 = sparse(d0_matrix_primal(mesh)); r = 0.0; S = [ring(mesh, e, 3) for e in 1:mesh.edges.n]   # columns of d0 = cells
    for v in 1:mesh.cells.n
        xv = mesh.cells.position[v]; x = _pt(xv)
        for p in nzrange(d0, v), j in S[rowvals(d0)[p]]
            a, b = edge_endpoints(mesh, j, true; about = xv)
            r = max(r, norm(a .- x), norm(b .- x))
        end
    end
    return r
end

out = String["largest stencil radius (absolute, and in units of h); A3 needs < 0.5"]
for mn in ("mesh1", "mesh2", "mesh3")
    mesh = VoronoiMesh("meshes/$mn.nc"); h = meshsize(mesh)
    for (lab, S) in (("13-edge", e -> ring(mesh, e, :t2)), ("3-ring", e -> ring(mesh, e, 3)),
                     ("3 triangle layers", e -> ring(mesh, e, :t3)), ("4-ring", e -> ring(mesh, e, 4)))
        r = stencil_radius(mesh, S)
        push!(out, @sprintf("   %-6s h = %.4f  %-18s radius %.3f = %.2f h   %s", mn, h, lab, r, r / h, r < 0.5 ? "ok" : "VIOLATES A3"))
        println(out[end])
    end
    if mn == "mesh3"
        r = laplacian_radius(mesh)
        push!(out, @sprintf("   %-6s rows of D0'HD0 (3-ring H): radius %.3f = %.2f h   %s", mn, r, r / h, r < 0.5 ? "ok" : "VIOLATES A3"))
        println(out[end])
    end
end
write(joinpath(TSDIR, "stencil_radius.txt"), join(out, "\n") * "\n")
