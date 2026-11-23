# ============================================================================
#  lie_transport_animation.jl -- a density transported over an SCVT mesh by the discrete Lie derivative
#
#      julia -t 4 scripts/transport/lie_transport_animation.jl [mesh5] [CFV2]
#          ->  output/figures/lie_transport.mp4, output/figures/fig_lie_snapshots.png
#
#  Two bumps carried by u(x,t) = (1/T,0) + cos(πt/T) swirl(x - t/T, y), T = 1 (lie_derivative.jl); the exact solution
#  returns to the initial state at T.  Panels: exact (backward characteristics, ρ(x,t) = ρ0(X(0; x,t)) since div u = 0),
#  H_DEC with the averaged edge density (= DEC = TRiSK), and H_P2 with the high-order edge density (CFVp).
# ============================================================================
include(joinpath(@__DIR__, "lie_derivative.jl"))
using CairoMakie, Printf

mn = length(ARGS) >= 1 ? ARGS[1] : "mesh5"
hv = length(ARGS) >= 2 ? ARGS[2] : "CFV2"
G = lie_setup(mn; cfv = (parse(Int, hv[end:end]),)); mesh = G["mesh"]; A = G["A"]; h = G["h"]
ρ0f(x) = bump(x; x0 = (0.35, 0.5), σ = 0.1) + bump(x; x0 = (0.65, 0.5), σ = 0.1)
ρ0 = discretize_scalar_field_cells(mesh, ρ0f; qdeg = 5).values
T = 1.0; dt = h / 8; nsteps = round(Int, T / dt); every = max(1, nsteps ÷ 64)
schemes = [("H_DEC", "TRiSK", "DEC / TRiSK:  H_DEC, averaged edge density"), ("H_P2", hv, "H_P2, high-order edge density ($hv)")]
runs = map(schemes) do (s, var, _)
    _, frames = rk4_lie(deform_operators(G, var, s), ρ0, T, dt; save_every = every)
    [f ./ A for f in frames]
end
times = [k * every * (T / nsteps) for k in 0:length(runs[1])-1]

"Exact solution at the generators: follow the characteristic back to t = 0 (RK4 in time, periodic)."
function exact_at(t; nsub = 200)
    map(mesh.cells.position) do p
        X = _vec2(p); s = t; ds = t / max(1, ceil(Int, nsub * t))
        while s > 1e-12
            k1 = deform_u(X, s, T); k2 = deform_u(X .- ds / 2 .* k1, s - ds / 2, T)
            k3 = deform_u(X .- ds / 2 .* k2, s - ds / 2, T); k4 = deform_u(X .- ds .* k3, s - ds, T)
            X = X .- ds / 6 .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4); s -= ds
        end
        ρ0f(mod.(X, 1.0))
    end
end
exact = [exact_at(t) for t in times]

# Voronoi cells as polygons, with periodic copies of the cells that cross the boundary of [0,1]^2
polys = Vector{Vector{Point2f}}(); owner = Int[]
for c in 1:mesh.cells.n
    vs = [Point2f(_vec2(closest(mesh.cells.position[c], mesh.vertices.position[v], 1.0, 1.0))...) for v in mesh.cells.vertices[c]]
    for sx in (-1, 0, 1), sy in (-1, 0, 1)
        w = [v + Point2f(sx, sy) for v in vs]
        if (sx == 0 && sy == 0) || any(p -> 0 <= p[1] <= 1 && 0 <= p[2] <= 1, w)
            push!(polys, w); push!(owner, c)
        end
    end
end

crange = (-0.15, 1.15)
errs = [[runs[r][i] .- exact[i] for i in eachindex(times)] for r in 1:2]
emax = maximum(maximum(abs.(e)) for e in errs[1])                   # one symmetric error scale for both schemes
erange = (-emax, emax)
zoom = (0.25, 0.75, 0.25, 0.75)

# ---------------- animation: top row solutions, bottom row the mesh (zoom) and the two errors on the same scale
fig = Figure(size = (1260, 840), fontsize = 14)
k = Observable(1)
function panel!(pos, vals, title; cmap = :turbo, cr = crange, stroke = 0.0, lim = (0, 1, 0, 1))
    ax = Axis(pos; aspect = DataAspect(), title = title, titlesize = 13)
    hidedecorations!(ax); limits!(ax, lim...)
    poly!(ax, polys; color = vals, colormap = cmap, colorrange = cr, lowclip = :black, highclip = :black, strokewidth = stroke,
          strokecolor = (:white, 0.6))
    return ax
end
panel!(fig[1, 1], @lift(exact[$k][owner]), "exact")
panel!(fig[1, 2], @lift(runs[1][$k][owner]), "DEC / TRiSK: H_DEC, averaged edge density")
panel!(fig[1, 3], @lift(runs[2][$k][owner]), "H_P2 + high-order edge density ($hv)")
Colorbar(fig[1, 4]; colormap = :turbo, colorrange = crange, lowclip = :black, label = "ρ")
panel!(fig[2, 1], @lift(runs[2][$k][owner]), "the Voronoi cells (zoom, H_P2)"; stroke = 0.6, lim = zoom)
panel!(fig[2, 2], @lift(errs[1][$k][owner]), "error, DEC / TRiSK"; cmap = :balance, cr = erange)
panel!(fig[2, 3], @lift(errs[2][$k][owner]), "error, H_P2 + $hv  (same colour scale)"; cmap = :balance, cr = erange)
Colorbar(fig[2, 4]; colormap = :balance, colorrange = erange, label = "error")
info = @lift(@sprintf("%s: %d SCVT cells, h = 1/%d     t = %.2f     max error  DEC/TRiSK %.4f   H_P2+%s %.4f", mn, mesh.cells.n,
                      round(Int, 1 / h), times[$k], maximum(abs.(errs[1][$k])), hv, maximum(abs.(errs[2][$k]))))
Label(fig[0, 1:4], info; fontsize = 15)
record(fig, joinpath(@__DIR__, "..", "..", "output", "figures", "lie_transport.mp4"), 1:length(times); framerate = 12) do i
    k[] = i
end

# ---------------- snapshots t = 0, T/4, T/2, 3T/4, T: exact solution and the two errors
snap = Figure(size = (1500, 900), fontsize = 14)
idx = [argmin(abs.(times .- t)) for t in (0, 0.25, 0.5, 0.75, 1.0)]
rows = (("exact solution", [exact[i] for i in idx], :turbo, crange),
        ("error, DEC / TRiSK (H_DEC)", [errs[1][i] for i in idx], :balance, erange),
        ("error, H_P2 + $hv", [errs[2][i] for i in idx], :balance, erange))
for (r, (name, F, cm, cr)) in enumerate(rows)
    for j in 1:5
        ax = Axis(snap[r, j]; aspect = DataAspect(), title = r == 1 ? @sprintf("t = %.2f", times[idx[j]]) : "", titlesize = 14)
        hidedecorations!(ax); limits!(ax, 0, 1, 0, 1)
        poly!(ax, polys; color = F[j][owner], colormap = cm, colorrange = cr, lowclip = :black, highclip = :black, strokewidth = 0)
    end
    Label(snap[r, 0], name; rotation = π / 2, tellheight = false, fontsize = 13)
end
Colorbar(snap[1, 6]; colormap = :turbo, colorrange = crange, lowclip = :black)
Colorbar(snap[2:3, 6]; colormap = :balance, colorrange = erange)
CairoMakie.save(joinpath(@__DIR__, "..", "..", "output", "figures", "fig_lie_snapshots.png"), snap; px_per_unit = 1.5)
println(@sprintf("final max error: H_DEC/TRiSK %.3e   H_P2/%s %.3e;  min ρ: %.3f  %.3f", maximum(abs.(errs[1][end])), hv,
                 maximum(abs.(errs[2][end])), minimum(runs[1][end]), minimum(runs[2][end])))
