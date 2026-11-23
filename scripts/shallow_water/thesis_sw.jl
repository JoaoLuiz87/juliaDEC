# ============================================================================
#  thesis_sw.jl -- linear shallow-water solution plot for hodge_thesis.tex
#
#      julia -t 4 scripts/shallow_water/thesis_sw.jl      ->  fig_th_sw.png
#  Travelling waves of sw_linear_check.jl at t = 1 on mesh 4 (RK4, Δt = 1e-3):
#  h (colour, on the Voronoi cells) and u (one arrow per triangle at its circumcentre, recovered from the circulations v),
#  exact / standard TRiSK (H_DEC, M0 = A) / H_P2 after one sweep with the corrected 0-form star; errors of h.
# ============================================================================
ENV["LS_MAIN"] = "0"; include(joinpath(@__DIR__, "..", "stars", "local_symmetric_check.jl"))
ENV["SW_MAIN"] = "0"; include(joinpath(@__DIR__, "sw_linear_check.jl"))
include(joinpath(@__DIR__, "..", "shared", "thesis_plot_helpers.jl"))
let mn = "mesh4"
    mesh = VoronoiMesh("meshes/$mn.nc"); nc = mesh.cells.n; Hd = hdiag(mesh)
    D2 = p1_rows(mesh; k = 2); rows = project_rows(D2, [zeros(length(D2.nbv[e])) for e in 1:D2.n]); rows = project_rows(D2, symmetrize(D2, rows))
    Hs1 = Hd + tosparse(D2, rows)
    hf, uf = wave(mesh); pts = [_pt(p) for p in mesh.cells.position]
    x0 = Vector{Float64}(vcat(hf.(pts, 0.0), reduce_primal_edges(mesh, x -> uf(x, 0.0)).values))
    hT = hf.(pts, 1.0); vT = Vector{Float64}(reduce_primal_edges(mesh, x -> uf(x, 1.0)).values)
    d0 = 1.0 * d0_matrix_primal(mesh); F0c = lu(mass0_corrected(mesh))
    xa = rk4(x -> sw_block(mesh, Hd) * x, x0, 1e-3, 1000)
    fc(x) = vcat(DEPTH .* (F0c \ (d0' * (Hs1 * x[nc+1:end]))), -GRAV .* (d0 * x[1:nc]))
    xc = rk4(fc, x0, 1e-3, 1000)
    tris, _, _ = triangles_ccw(mesh); cc = ccpoints(tris); polys, own = periodic_copies(cellpolys(mesh))
    r = (minimum(hT), maximum(hT)); VT = triangle_vector(mesh, tris, vT); ls = 0.03 / maximum(norm.(eachrow(VT)))
    f = Figure(size = (1500, 380), fontsize = 15)
    map_panel(f[1, 1], polys, own, hT, "exact h and u at t = 1"; range = r, pts = cc, dirs = vecs(VT), ls = ls)
    map_panel(f[1, 2], polys, own, xa[1:nc], "TRiSK: H_DEC, M0 = A"; range = r, pts = cc, dirs = vecs(triangle_vector(mesh, tris, xa[nc+1:end])), ls = ls)
    map_panel(f[1, 3], polys, own, xc[1:nc], "H_P2 (1 sweep), corrected M0"; range = r, pts = cc, dirs = vecs(triangle_vector(mesh, tris, xc[nc+1:end])), ls = ls)
    err_panel(f[1, 4], polys, own, xa[1:nc] .- hT, "error of h, TRiSK")
    err_panel(f[1, 5], polys, own, xc[1:nc] .- hT, "error of h, H_P2")
    CairoMakie.save(thesis_out("fig_th_sw.png"), f; px_per_unit = 2)
    @printf("SW mesh 4: max error of h  TRiSK %.2e   H_P2 swept + C0 %.2e\n", maximum(abs.(xa[1:nc] .- hT)), maximum(abs.(xc[1:nc] .- hT)))
end
