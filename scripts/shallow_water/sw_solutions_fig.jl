# ============================================================================
#  sw_solutions_fig.jl -- fig_sw_solutions.png: the plane wave with rotation (Peixoto et al. 2026) at 1.5 days on SCVT mesh 5, depth
#  perturbation η of the exact solution and of three schemes (TRSK with H_DEC, TRSK with H_P2, consistent with H_P2), and their errors.
#  Second figure fig_sw_balance_err.png: error of the geostrophic balance after 1 day on mesh 6, the same three schemes.
#
#      julia -t 4 scripts/shallow_water/sw_solutions_fig.jl
# ============================================================================
include(joinpath(@__DIR__, "sw_linear_three.jl"))
include(joinpath(@__DIR__, "sw_ablation.jl"))
const LK = LDOM / 1e6                                                     # domain side in 1000 km

"Voronoi cell polygons (unit: 1000 km), unwrapped about the generator, plus periodic copies; returns (polys, owner)"
function cell_polys(mesh)
    xp, yp = mesh.x_period, mesh.y_period; polys = Vector{Vector{Point2f}}(); own = Int[]
    for i in 1:mesh.cells.n
        c = mesh.cells.position[i]; pc = _p2(c)
        ang(v) = (q = _p2(closest(c, mesh.vertices.position[v], xp, yp)) .- pc; atan(q[2], q[1]))
        vs = sort(collect(Int.(mesh.cells.vertices[i])); by = ang)
        Q = [(q = _p2(closest(c, mesh.vertices.position[v], xp, yp)); Point2f(q[1] * LK, q[2] * LK)) for v in vs]
        push!(polys, Q); push!(own, i)
        any(p -> p[1] < 0 || p[1] > LK || p[2] < 0 || p[2] > LK, Q) || continue
        for sx in -1:1, sy in -1:1
            (sx, sy) == (0, 0) || (push!(polys, [p .+ Point2f(sx * LK, sy * LK) for p in Q]); push!(own, i))
        end
    end
    return polys, own
end

"final depth perturbation of a scheme and star on a mesh, for a test (exact solution of exact_lin)"
function final_depth(mn, st, scheme, test)
    mesh = VoronoiMesh("meshes/$mn.nc"); S = cached_stars(mn; mesh); O = sw_ops(mesh, S[st])
    O0 = sw_ops(mesh, S["H_DEC"]); P = scheme == :consistent ? perp_p2(mesh, O0.W * O0.H) : nothing
    M0c = scheme == :consistent ? lu(mass0_corrected(mesh) .* O0.L^2) : nothing
    η, u, T = exact_lin(test); h0, v0, _, _ = cochains(O, η, u, 0.0); hT, _, _, _ = cochains(O, η, u, T)
    rhs, _ = scheme_ops(O, scheme, P, M0c)
    c = sqrt(GJ * H0J); dtmax = 0.1 * minimum(O.le) / c; n = ceil(Int, T / dtmax); dt = T / n
    xT = rk4_lin(rhs, vcat(h0, v0), dt, n)
    return mesh, xT[1:O.nc], hT, h0
end

const CMB = cgrad([colorant"#1c5cab", colorant"#7fa9d9", colorant"#f0efec", colorant"#e39a8f", colorant"#c23b3a"])
panel!(pos, polys, own, q, title, rng) = (ax = Axis(pos; aspect = DataAspect(), title = title, titlesize = 13, titlealign = :left); hidedecorations!(ax);
                                         poly!(ax, polys; color = q[own], colormap = CMB, colorrange = rng, strokewidth = 0); xlims!(ax, 0, LK); ylims!(ax, 0, LK); ax)

let mn = "mesh5"
    res = [(st, sc, final_depth(mn, st, sc, :plane)) for (st, sc) in (("H_DEC", :trsk), ("H_P2", :trsk), ("H_P2", :consistent))]
    mesh, _, hT, h0 = res[1][3]; polys, own = cell_polys(mesh); m = maximum(abs, hT)
    rel(h) = sqrt(sum(mesh.cells.area .* (h .- hT) .^ 2) / sum(mesh.cells.area .* hT .^ 2))
    f = Figure(size = (1500, 780), fontsize = 14)
    panel!(f[1, 1], polys, own, hT, "exact depth perturbation at 1.5 days (m)", (-m, m))
    for (j, (st, sc, (_, h, _, _))) in enumerate(res)
        panel!(f[1, j+1], polys, own, h, @sprintf("%s, %s  (rel. rms error %.1e)", st, sc == :trsk ? "TRSK" : "consistent", rel(h)), (-m, m))
        e = h .- hT; me = maximum(abs, e)
        panel!(f[2, j+1], polys, own, e, @sprintf("error, %s %s (max %.1e m)", st, sc == :trsk ? "TRSK" : "consistent", me), (-me, me))
    end
    panel!(f[2, 1], polys, own, h0, "initial depth perturbation", (-m, m))
    Colorbar(f[1, 5]; colormap = CMB, limits = (-m, m), label = "η (m)")
    CairoMakie.save(joinpath(@__DIR__, "..", "..", "output", "figures", "fig_sw_solutions.png"), f; px_per_unit = 2)
    println("plane wave mesh5: relative errors ", [(st, sc, rel(r[2])) for (st, sc, r) in res])
end
let mn = "mesh6"
    res = [(st, sc, final_depth(mn, st, sc, :balance)) for (st, sc) in (("H_DEC", :trsk), ("H_P2", :trsk), ("H_P2", :consistent))]
    mesh, _, hT, _ = res[1][3]; polys, own = cell_polys(mesh)
    f = Figure(size = (1400, 480), fontsize = 14)
    for (j, (st, sc, (_, h, _, _))) in enumerate(res)
        e = h .- hT; me = maximum(abs, e)
        panel!(f[1, j], polys, own, e, @sprintf("error %s %s (max %.1e m)", st, sc == :trsk ? "TRSK" : "consistent", me), (-me, me))
    end
    Label(f[2, 1:3], "geostrophic balance after 1 day, mesh 6 (16384 cells): error of the depth; own symmetric range per panel", fontsize = 14)
    CairoMakie.save(joinpath(@__DIR__, "..", "..", "output", "figures", "fig_sw_balance_err.png"), f; px_per_unit = 2)
end
