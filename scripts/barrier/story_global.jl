# ============================================================================
#  story_global.jl -- why symmetry makes a consistent star global: figure and numbers for hodge_story.tex
#
#      julia -t 4 scripts/barrier/story_global.jl     ->  fig_st_global.png, ts_data/story_global.txt
#
#  (a) The chain: N rows on a cycle, row k: (left coefficient) + (right coefficient) = r_k.
#      Symmetric: the coefficient between rows k and k+1 is ONE unknown x_k, shared by rows k and k+1: x_{k-1} + x_k = r_k.
#      Non-symmetric: row k has its own two unknowns; the least-norm solution is (r_k/2, r_k/2).
#      Perturb r at one row and plot the change of the coefficients along the chain.
#  (b) The patch identity on SCVT mesh 5: edges of a patch P coloured by their interior source ½|e|³|ẽ| nᵀZn (f = dilation,
#      g = shear), and the coefficient pairs of H_P1 that cross ∂P (they carry the balance).
#  (c) How far the coefficients react when one generator moves (ts_data/locality.jls): H_DEC, H_P1, H_P2.
# ============================================================================
ENV["LOC_MAIN"] = "0"; include(joinpath(@__DIR__, "locality_check.jl"))
include(joinpath(@__DIR__, "..", "stars", "stars_cache.jl"))
include(joinpath(@__DIR__, "..", "shared", "thesis_plot_helpers.jl"))
using Serialization
OUTG = String[]; sg(s) = (println(s); flush(stdout); push!(OUTG, s))

# ---------------------------------------------------------------------------- (a) the chain
N = 41; k0 = 21
M = spzeros(N, N); for k in 1:N; M[k, mod1(k - 1, N)] = 1; M[k, k] = 1; end      # row k: x_{k-1} + x_k
δr = zeros(N); δr[k0] = 1.0
δx_sym = Matrix(M) \ δr                                                           # change of the shared unknowns
δx_non = zeros(N); δx_non[k0] = 0.5                                               # row k0 only (its two own unknowns, 1/2 each)
sg(@sprintf("chain, N = %d: symmetric |Δx_k| min %.3f max %.3f (every coefficient changes by 1/2); non-symmetric: only row %d", N,
            minimum(abs, δx_sym), maximum(abs, δx_sym), k0))
# the patch identity in the chain: Σ_{k=p}^{q} (-1)^k (row k) = (-1)^p x_{p-1} + (-1)^q x_q
let r = randn(N); x = Matrix(M) \ r; p, q = 10, 30
    lhs = sum((-1)^k * r[k] for k in p:q); rhs = (-1)^p * x[mod1(p - 1, N)] + (-1)^q * x[q]
    sg(@sprintf("chain identity: Σ_{k=%d}^{%d} (-1)^k r_k = %+.12f ;  boundary unknowns (-1)^p x_{p-1} + (-1)^q x_q = %+.12f", p, q, lhs, rhs))
end

# ---------------------------------------------------------------------------- (b) the patch on mesh 5
mesh = VoronoiMesh("meshes/mesh5.nc"); S = cached_stars("mesh5"; mesh); h = meshsize(mesh)
v0 = argmin([norm(_pt(p) .- [0.5, 0.5]) for p in mesh.cells.position]); x0v = mesh.cells.position[v0]; x0 = _pt(x0v)
geo = edge_geometry(mesh, x0v); ρ = 0.15
Gf, Gg = GBASIS["I"], GBASIS["D"]; Z = Zof(Gf, Gg)
src = [0.5 * E.le^3 * E.ld * dot(E.n, Z * E.n) for E in geo]
inP = [E.r < ρ for E in geo]
H1 = S["H_P1"]; Ht = sparse(H1'); cross = Tuple{Int,Int}[]
for j in findall(inP), p in nzrange(Ht, j)
    i = rowvals(Ht)[p]; inP[i] || abs(nonzeros(Ht)[p]) < 1e-14 || push!(cross, (j, i))
end
for (name, H) in (("H_DEC", S["H_DEC"]), ("H_P1", S["H_P1"]), ("H_P2", S["H_P2"])), rr in (0.15, 0.30)
    q, Dsum = QD(geo, x0, rr, [0.0, 0], Gf, [0.0, 0], Gg)
    c = crossing(H, geo, x0, rr, [0.0, 0], Gf, [0.0, 0], Gg)
    f(x) = Gf * (x - x0); g(x) = Gg * (x - x0)
    Rf = [E.le * dot(E.t, f(E.m)) for E in geo]; Rg = [E.le * dot(E.t, g(E.m)) for E in geo]; HRf = H * Rf; HRg = H * Rg
    lhs = sum(Rf[j] * HRg[j] - Rg[j] * HRf[j] for j in eachindex(geo) if geo[j].r < rr)
    sg(@sprintf("patch ρ = %.2f, %-5s: Σ_P[(Rf)(HRg) − (Rg)(HRf)] = %+.6e   Q(P) = %+.6e   interior sum ½Σ|e|³|ẽ|nᵀZn = %+.6e   crossing pairs Σ H_ji K_ji = %+.6e",
                rr, name, lhs, q, Dsum, c))
end
sg(@sprintf("patch ρ = 0.15: %d edges inside, %d coefficient pairs of H_P1 cross the boundary", count(inP), length(cross)))

# ---------------------------------------------------------------------------- figure
L = deserialize(joinpath(TSDIR, "locality.jls"))
f = Figure(size = (1500, 520), fontsize = 15)
ax1 = Axis(f[1, 1]; xlabel = "row k along the chain", ylabel = "|change of the coefficients of row k|", title = "(a) a chain of 41 rows: perturb row 21",
           axstyle_th...)
scatterlines!(ax1, 1:N, abs.(δx_sym); color = C_P1, marker = :diamond, markersize = 8, label = "symmetric (shared unknowns)")
scatterlines!(ax1, 1:N, abs.(δx_non); color = C_P2, marker = :rect, markersize = 7, label = "non-symmetric (own unknowns)"); ylims!(ax1, -0.03, 0.75)
axislegend(ax1; position = :ct, framevisible = false)
ax2 = Axis(f[1, 2]; aspect = DataAspect(), title = "(b) mesh 5: a patch P and the pairs of H_P1 crossing ∂P", titlealign = :left)
hidedecorations!(ax2)
m = maximum(abs, src[inP])
for (e, E) in enumerate(geo)
    E.r < 0.24 || continue
    a = E.m .- (E.le / 2) .* E.t; b = E.m .+ (E.le / 2) .* E.t
    if inP[e]
        lines!(ax2, [a[1], b[1]], [a[2], b[2]]; color = src[e], colormap = DIVM, colorrange = (-m, m), linewidth = 2.2)
    else
        lines!(ax2, [a[1], b[1]], [a[2], b[2]]; color = (:gray, 0.35), linewidth = 0.8)
    end
end
for (j, i) in cross
    lines!(ax2, [geo[j].m[1], geo[i].m[1]], [geo[j].m[2], geo[i].m[2]]; color = (INK, 0.25), linewidth = 0.6)
end
θ = range(0, 2π; length = 200); lines!(ax2, x0[1] .+ ρ .* cos.(θ), x0[2] .+ ρ .* sin.(θ); color = INK, linestyle = :dash, linewidth = 1.5)
scatter!(ax2, [x0[1]], [x0[2]]; color = :black, markersize = 10, marker = :star5)
xlims!(ax2, x0[1] - 0.24, x0[1] + 0.24); ylims!(ax2, x0[2] - 0.24, x0[2] + 0.24)
ax3 = Axis(f[1, 3]; xscale = log10, yscale = log10, xlabel = "distance r/h of the row from the moved generator",
           ylabel = "max |∂H_ji/∂x_v| · h", title = "(c) mesh 5: who reacts when one generator moves", axstyle_th..., xticks = ([1, 2, 5, 10, 20, 40], ["1", "2", "5", "10", "20", "40"]))
for (name, col, mk) in (("H_DEC", C_DEC, :circle), ("H_P1", C_P1, :diamond), ("H_P2", C_P2, :rect))
    b = L[("mesh5", :B, name)].bins; rr = (0:length(b)-1) .+ 0.5; ok = b .> 1e-12
    scatterlines!(ax3, rr[ok], b[ok]; color = col, marker = mk, markersize = 8, linewidth = 2, label = name)
end
rr = [6.0, 38.0]; lines!(ax3, rr, 1.5e-4 .* (rr ./ 6) .^ -3; color = INK2, linestyle = :dot); text!(ax3, 20, 1.5e-4 * (20 / 6)^-3; text = "(h/r)³  ", align = (:right, :top), fontsize = 14, color = INK2)
axislegend(ax3; position = :rt, framevisible = false)
CairoMakie.save(thesis_out("fig_st_global.png"), f; px_per_unit = 2)
write(joinpath(TSDIR, "story_global.txt"), join(OUTG, "\n") * "\n")
println("global figure written")
