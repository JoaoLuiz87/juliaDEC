# ============================================================================
#  gs_tutorial2d_figures.jl -- figures and numbers of hodge_method_2d.tex (the 2D method, step by step)
#
#      julia scripts/archive/gs_tutorial2d_figures.jl     (needs output/gs_data/check_table.txt from gs_check.jl)
#
#  fig_t2_data.png       Step 1: the circulations of a field on the stencil, and the linear field reconstructed on T
#  fig_t2_invisible.png  Step 1: data of a linear basis field versus an invisible pattern (a column of W_T)
#  fig_t2_probe.png      Steps 3-4: what row e "sees": DEC (a point), least squares Φ_e, corrected Φ_e
#  fig_t2_diamonds.png   Step 4: the diamonds |e||ẽ|/2 tile the torus (identity A)
#  fig_t2_row.png        Step 5: the entries of one row of H on the mesh (DEC, least squares, corrected)
#  fig_t2_conv.png       Step 5: convergence of DEC, the least-squares energy star and the corrected star
#  gs_data/tutorial2d.txt  numbers quoted in the text
# ============================================================================
using VoronoiMeshes, TensorsLite, TensorsLiteGeometry, SmallCollections, LinearAlgebra, NCDatasets
using StaticArrays, SparseArrays, Printf
Base.similar(bc::Base.Broadcast.Broadcasted, ::Type{Union{}}) = Array{Union{}}(undef, length.(axes(bc)))
const ROOT = joinpath(@__DIR__, "..", ".."); cd(ROOT)
include(joinpath(ROOT, "src", "juliaDEC.jl")); include(joinpath(ROOT, "src", "hodge_stars.jl")); include(joinpath(ROOT, "src", "galerkin_star.jl"))
using CairoMakie
const OUT = joinpath(@__DIR__, "..", "..", "output", "gs_data"); mkpath(OUT)
L = String[]; logp(s) = (println(s); flush(stdout); push!(L, s))
unw(y, x, P) = [y[1] - P[1] * round((y[1] - x[1]) / P[1]), y[2] - P[2] * round((y[2] - x[2]) / P[2])]
u1(x) = [sin(2pi * x[1]) * cos(2pi * x[2]), cos(2pi * x[1]) * sin(2pi * x[2])]
u2(x) = [sin(2pi * x[1]) * cos(2pi * x[2]), -cos(2pi * x[1]) * sin(2pi * x[2])]

"the least-squares energy star (Z = 0), same penalty as galerkin_star_matrix"
function ls_star(mesh, cells)
    E = mesh.edges.n; Hi, Hj, Hv = Int[], Int[], Float64[]
    for C in cells
        R = C.R0; B = R' * C.M * R; Pr = I - C.P * R; Sm = Pr' * Diagonal(1 ./ C.lens .^ 2) * Pr
        B .+= (tr(B) / tr(Sm)) .* Sm; B = (B + B') / 2
        for (q1, e1) in enumerate(C.S), (q2, e2) in enumerate(C.S)
            push!(Hi, e1); push!(Hj, e2); push!(Hv, B[q1, q2])
        end
    end
    H = sparse(Hi, Hj, Hv, E, E); (H + H') / 2
end

mesh = VoronoiMesh("meshes/mesh3.nc"); P = (mesh.x_period, mesh.y_period)
H, info = galerkin_star_matrix(mesh; keep = true, verbose = false)
cells, edata, Rs = info.data.cells, info.data.edges, info.data.R; s = info.s
HLS = ls_star(mesh, cells); Hd = spdiagm(edge_lengths(mesh, false) ./ edge_lengths(mesh, true))
t0 = argmin([norm(C.xc - [0.5, 0.5]) for C in cells]); C = cells[t0]; c0 = C.xc
e0 = C.S[1]; ed = edata[e0]; m0 = ed.m
segP(e, ref) = (p = edge_endpoints(mesh, e, true; about = _gs_vec(ref)); (p[1], p[2]))
logp(@sprintf("mesh3: cell %d, centroid (%.4f, %.4f), stencil %d edges, s = %.5f; edge e = %d, |e| = %.5f, |ẽ| = %.5f", t0, c0..., length(C.S), s, e0, ed.le, ed.lde))
tvec = (ed.p2 - ed.p1) / ed.le; nvec = [-tvec[2], tvec[1]]; δ0 = dot((ed.v1 + ed.v2) / 2 - ed.m, nvec)
logp(@sprintf("   t_e = (%.4f, %.4f), n_e = (%.4f, %.4f), δ_e = %.3e (δ_e/|e| = %.4f)", tvec..., nvec..., δ0, δ0 / ed.le))
logp(@sprintf("   P_T has %d rows; rank %d; dim of the invisible patterns n_T - 6 = %d", size(C.P, 1), rank(C.P), size(C.W, 2)))

# ---------------------------------------------------------------- the three stars on mesh3: symmetry, exactness, errors
for (lbl, Hm) in (("DEC", Hd), ("least-squares energy star", HLS), ("corrected energy star", H))
    a1 = gs_hodge_error(mesh, Hm, u1)
    logp(@sprintf("   %-27s asym %.0e, linear exactness %.1e, u1 max error %.3e, nnz/row %.1f, λmin(H/Hd) %.3f", lbl,
                  norm(Hm - Hm', 1) / norm(Hm, 1), gs_row_exactness(mesh, Hm), a1[1], nnz(Hm) / mesh.edges.n,
                  gs_smallest_eigenvalue(sparse(Hm); D = Hd)))
end

# ---------------------------------------------------------------- figure: data and reconstruction (Step 1)
uf(x) = [sin(4pi * x[1]) * cos(4pi * x[2]), cos(4pi * x[1]) * sin(4pi * x[2])]      # a curved field, for illustration
data = [_gs_seg(x -> reshape(uf(x), 2, 1), segP(e, c0)...)[1] for e in C.S]          # circulations on the stencil
crec = C.R0 * data                                                                   # least-squares coefficients
tri = [unw(p, c0, P) for p in C.tri]
fig = Figure(size = (1400, 560))
ax = Axis(fig[1, 1]; aspect = DataAspect(), title = "(a) the data: circulation / |e| on the 24 stencil edges"); hidedecorations!(ax); hidespines!(ax)
for e in 1:mesh.edges.n
    p1, p2 = segP(e, c0); norm((p1 + p2) / 2 - c0) < 0.2 && lines!(ax, [p1[1], p2[1]], [p1[2], p2[2]]; color = (:gray, 0.25), linewidth = 0.7)
end
poly!(ax, Point2f[Tuple(p) for p in tri]; color = (:steelblue, 0.25))
vals = [data[q] / C.lens[q] for q in eachindex(C.S)]; lim = maximum(abs, vals)
for (q, e) in enumerate(C.S)
    p1, p2 = segP(e, c0)
    lines!(ax, [p1[1], p2[1]], [p1[2], p2[2]]; color = vals[q], colormap = :balance, colorrange = (-lim, lim), linewidth = 5)
    arrows2d!(ax, [Point2f(Tuple(p1 + 0.45(p2 - p1)))], [Vec2f(Tuple(0.12(p2 - p1)))]; color = :black, shaftwidth = 0.8, tipwidth = 6, tiplength = 6)
end
gp = [c0 + [i, j] for i in -0.15:0.05:0.15 for j in -0.15:0.05:0.15]
arrows2d!(ax, Point2f.(Tuple.(gp)), Vec2f.(Tuple.(0.03 .* uf.(gp))); color = (:gray30, 0.6), shaftwidth = 0.8, tipwidth = 5, tiplength = 5)
Colorbar(fig[1, 2]; colormap = :balance, colorrange = (-lim, lim), height = Relative(0.7))
xlims!(ax, c0[1] - 0.17, c0[1] + 0.17); ylims!(ax, c0[2] - 0.17, c0[2] + 0.17)
ax = Axis(fig[1, 3]; aspect = DataAspect(), title = "(b) on T: true field (black), reconstructed linear field (red)"); hidedecorations!(ax); hidespines!(ax)
poly!(ax, Point2f[Tuple(p) for p in tri]; color = (:steelblue, 0.15), strokecolor = :black, strokewidth = 1)
qp = [l[1] * C.tri[1] + l[2] * C.tri[2] + l[3] * C.tri[3] for l in ((1/3, 1/3, 1/3), (0.6, 0.2, 0.2), (0.2, 0.6, 0.2), (0.2, 0.2, 0.6),
      (0.45, 0.45, 0.1), (0.45, 0.1, 0.45), (0.1, 0.45, 0.45))]
sc = 0.45 * norm(tri[1] - tri[2]) / maximum(norm(uf(x)) for x in qp)
arrows2d!(ax, Point2f.(Tuple.(qp)), Vec2f.(Tuple.(sc .* uf.(qp))); color = :black, shaftwidth = 1.5, tipwidth = 9, tiplength = 9)
arrows2d!(ax, Point2f.(Tuple.(qp)), Vec2f.(Tuple.([sc .* (basis_at(x, C.xc, true, s) * crec) for x in qp])); color = :firebrick,
          shaftwidth = 1.5, tipwidth = 9, tiplength = 9)
pad = 0.35 * norm(tri[1] - tri[2])
xlims!(ax, minimum(p[1] for p in tri) - pad, maximum(p[1] for p in tri) + pad); ylims!(ax, minimum(p[2] for p in tri) - pad, maximum(p[2] for p in tri) + pad)
CairoMakie.save(joinpath(@__DIR__, "..", "..", "output", "figures", "fig_t2_data.png"), fig; px_per_unit = 2)
err_rec = maximum(norm(basis_at(x, C.xc, true, s) * crec - uf(x)) for x in qp) / maximum(norm(uf(x)) for x in qp)
logp(@sprintf("   reconstruction of uf on T: relative max error at the sample points %.3f (uf is not linear)", err_rec))
lin(x) = [0.3 + 2.0 * (x[1] - 0.5) - 1.0 * (x[2] - 0.5), -0.2 + 0.5 * (x[1] - 0.5) + 1.5 * (x[2] - 0.5)]
dl = [_gs_seg(x -> reshape(lin(x), 2, 1), segP(e, c0)...)[1] for e in C.S]
logp(@sprintf("   reconstruction of a linear field: max coefficient error %.1e (reproduction)", maximum(abs.(C.R0 * dl - C.R0 * (C.P * (C.R0 * dl))))))
logp(@sprintf("   invisible patterns: ‖P_Tᵀ W_T‖ = %.1e; least-squares reconstruction of a W_T column: %.1e", opnorm(C.P' * C.W), norm(C.R0 * C.W[:, 1])))

# ---------------------------------------------------------------- figure: a visible versus an invisible pattern (Step 1)
fig = Figure(size = (1300, 560))
for (col, v, ttl) in ((1, C.P[:, 3], "(a) data of the linear field φ₃ = (X, 0): a column of P_T"),
                      (2, C.W[:, 1], "(b) an invisible pattern: a column of W_T"))
    ax = Axis(fig[1, col]; aspect = DataAspect(), title = ttl); hidedecorations!(ax); hidespines!(ax)
    for e in 1:mesh.edges.n
        p1, p2 = segP(e, c0); norm((p1 + p2) / 2 - c0) < 0.2 && lines!(ax, [p1[1], p2[1]], [p1[2], p2[2]]; color = (:gray, 0.25), linewidth = 0.7)
    end
    poly!(ax, Point2f[Tuple(p) for p in tri]; color = (:steelblue, 0.25))
    lim = maximum(abs, v)
    for (q, e) in enumerate(C.S)
        p1, p2 = segP(e, c0)
        lines!(ax, [p1[1], p2[1]], [p1[2], p2[2]]; color = v[q], colormap = :balance, colorrange = (-lim, lim), linewidth = 6)
    end
    xlims!(ax, c0[1] - 0.17, c0[1] + 0.17); ylims!(ax, c0[2] - 0.17, c0[2] + 0.17)
end
CairoMakie.save(joinpath(@__DIR__, "..", "..", "output", "figures", "fig_t2_invisible.png"), fig; px_per_unit = 2)

# ---------------------------------------------------------------- figure: the probes (Steps 3-4)
function moments(e, which)
    acc = zeros(6)
    for (k, Ck) in enumerate(cells)
        q = findfirst(==(e), Ck.S); q === nothing && continue
        acc .+= Ck.Nm[q]' * (which == :lsq ? Ck.R0 : Rs[k])[:, q]
    end
    acc
end
mL = moments(e0, :lsq); mC = moments(e0, :con)
logp("   moments of ẽ           : " * join([@sprintf("%+.6f", x) for x in ed.β], " "))
logp("   moments of Φ_e, l.sq.   : " * join([@sprintf("%+.6f", x) for x in mL], " "))
logp("   moments of Φ_e, correct.: " * join([@sprintf("%+.6f", x) for x in mC], " "))
dec = [ed.lde / ed.le * x for x in [_gs_seg(x -> basis_at(x, m0, true, s), ed.p1, ed.p2)...]]
logp("   moments of the DEC probe: " * join([@sprintf("%+.6f", x) for x in dec], " "))
fig = Figure(size = (1500, 520))
for (col, which, ttl) in ((1, :dec, "(a) DEC: a point mass |ẽ| tₑ at mₑ"), (2, :lsq, "(b) least squares: Φₑ"), (3, :con, "(c) corrected: Φₑ"))
    ax = Axis(fig[1, col]; aspect = DataAspect(), title = ttl); hidedecorations!(ax); hidespines!(ax)
    pts = Point2f[]; vecs = Vec2f[]
    for (k, Ck) in enumerate(cells)
        q = findfirst(==(e0), Ck.S); q === nothing && continue
        sh = unw(Ck.xc, m0, P) - Ck.xc
        poly!(ax, Point2f[Tuple(p + sh) for p in Ck.tri]; color = (:steelblue, 0.10), strokecolor = (:gray, 0.6), strokewidth = 0.6)
        which == :dec && continue
        Rk = which == :lsq ? Ck.R0 : Rs[k]
        for l in ((1 / 3, 1 / 3, 1 / 3), (0.6, 0.2, 0.2), (0.2, 0.6, 0.2), (0.2, 0.2, 0.6))
            x = l[1] * Ck.tri[1] + l[2] * Ck.tri[2] + l[3] * Ck.tri[3]
            push!(pts, Point2f(Tuple(x + sh))); push!(vecs, Vec2f(Tuple(basis_at(x, Ck.xc, true, s) * Rk[:, q])))
        end
    end
    if which == :dec
        arrows2d!(ax, [Point2f(Tuple(m0))], [Vec2f(Tuple(0.6 * s * tvec))]; color = :black, shaftwidth = 2.5, tipwidth = 12, tiplength = 12)
    else
        sc = 0.3 * s / maximum(norm, vecs)
        arrows2d!(ax, pts, vecs .* sc; color = :black, shaftwidth = 1.1, tipwidth = 6, tiplength = 6)
    end
    lines!(ax, [ed.p1[1], ed.p2[1]], [ed.p1[2], ed.p2[2]]; color = :black, linewidth = 3)
    lines!(ax, [ed.v1[1], ed.v2[1]], [ed.v1[2], ed.v2[2]]; color = :firebrick, linewidth = 3)
    scatter!(ax, [Point2f(Tuple(m0))]; color = :black, markersize = 9)
    scatter!(ax, [Point2f(Tuple((ed.v1 + ed.v2) / 2))]; color = :firebrick, marker = :utriangle, markersize = 12)
    mm = which == :dec ? dec : which == :lsq ? mL : mC
    text!(ax, m0[1] - 0.14, m0[2] - 0.165; fontsize = 12, text = @sprintf("mass  (%.4f, %.4f)\ndual edge (%.4f, %.4f)", mm[1], mm[2], ed.β[1], ed.β[2]))
    xlims!(ax, m0[1] - 0.15, m0[1] + 0.15); ylims!(ax, m0[2] - 0.17, m0[2] + 0.15)
end
CairoMakie.save(joinpath(@__DIR__, "..", "..", "output", "figures", "fig_t2_probe.png"), fig; px_per_unit = 2)

# ---------------------------------------------------------------- figure: diamonds (identity A)
fig = Figure(size = (760, 720))
ax = Axis(fig[1, 1]; aspect = DataAspect(), title = "diamonds of the edges: area |e||ẽ|/2, they tile the torus"); hidedecorations!(ax); hidespines!(ax)
pal = [:lightskyblue, :navajowhite, :palegreen, :thistle, :lightpink, :khaki]
cnt = 0
for e in 1:mesh.edges.n
    dg = edata[e]; sh = unw(dg.m, c0, P) - dg.m
    norm(dg.m + sh - c0) > 0.16 && continue
    global cnt += 1
    poly!(ax, Point2f[Tuple(p + sh) for p in (dg.p1, dg.v1, dg.p2, dg.v2)]; color = (pal[mod1(e, 6)], 0.85), strokecolor = :white, strokewidth = 1.2)
end
dg = ed
poly!(ax, Point2f[Tuple(p) for p in (dg.p1, dg.v1, dg.p2, dg.v2)]; color = (:steelblue, 0.6), strokecolor = :black, strokewidth = 1.5)
lines!(ax, [dg.p1[1], dg.p2[1]], [dg.p1[2], dg.p2[2]]; color = :black, linewidth = 3)
lines!(ax, [dg.v1[1], dg.v2[1]], [dg.v1[2], dg.v2[2]]; color = :firebrick, linewidth = 3)
xlims!(ax, c0[1] - 0.12, c0[1] + 0.12); ylims!(ax, c0[2] - 0.12, c0[2] + 0.12)
CairoMakie.save(joinpath(@__DIR__, "..", "..", "output", "figures", "fig_t2_diamonds.png"), fig; px_per_unit = 2)
A = zeros(2, 2); B3 = zeros(2, 2, 2); wabs = 0.0
for dd in edata
    t = (dd.p2 - dd.p1) / dd.le; n = [-t[2], t[1]]; δ = dot((dd.v1 + dd.v2) / 2 - dd.m, n)
    A .+= dd.le * dd.lde .* (t * t'); global wabs += abs(dd.le * dd.lde * δ)
    for a in 1:2, b in 1:2, c in 1:2
        B3[a, b, c] += dd.le * dd.lde * δ * t[a] * t[b] * n[c]
    end
end
logp(@sprintf("   identity (A): Σ|e||ẽ| t⊗t = [%.12f %.12f; %.12f %.12f] (|Ω| = 1);  Σ|e||ẽ|/2 = %.12f", A..., sum(dd.le * dd.lde for dd in edata) / 2))
logp(@sprintf("   identity (B): max entry of Σ|e||ẽ|δ t⊗t⊗n = %.1e, relative to Σ|e||ẽ||δ| = %.1e", maximum(abs, B3), maximum(abs, B3) / wabs))

# ---------------------------------------------------------------- figure: one row of H on the mesh (Step 5)
fig = Figure(size = (1500, 540))
lim = 0.05
for (col, Hm, ttl) in ((1, Hd, "(a) DEC: 1 entry"), (2, HLS, "(b) least-squares energy star"), (3, H, "(c) corrected energy star"))
    Hm = sparse(Hm); row = Hm[e0, :]; nzf = findall(!iszero, row)
    ax = Axis(fig[1, col]; aspect = DataAspect(), title = ttl * (col > 1 ? @sprintf(": %d entries", length(nzf)) : "")); hidedecorations!(ax); hidespines!(ax)
    for e in 1:mesh.edges.n
        p1, p2 = segP(e, m0); norm((p1 + p2) / 2 - m0) < 0.26 && lines!(ax, [p1[1], p2[1]], [p1[2], p2[2]]; color = (:gray, 0.25), linewidth = 0.7)
    end
    for f in nzf
        p1, p2 = segP(f, m0)
        lines!(ax, [p1[1], p2[1]], [p1[2], p2[2]]; color = row[f], colormap = :balance, colorrange = (-lim, lim), linewidth = f == e0 ? 7 : 4.5)
    end
    lines!(ax, [ed.v1[1], ed.v2[1]], [ed.v1[2], ed.v2[2]]; color = :black, linewidth = 1.5, linestyle = :dash)
    xlims!(ax, m0[1] - 0.23, m0[1] + 0.23); ylims!(ax, m0[2] - 0.23, m0[2] + 0.23)
    logp(@sprintf("   row %d of %s: %d entries, diagonal %.4f, sum of |entries| %.4f", e0, ttl, length(nzf), row[e0], sum(abs, row)))
end
Colorbar(fig[1, 4]; colormap = :balance, colorrange = (-lim, lim), height = Relative(0.7), label = "H_ef (clipped at ±0.05)")
CairoMakie.save(joinpath(@__DIR__, "..", "..", "output", "figures", "fig_t2_row.png"), fig; px_per_unit = 2)

# ---------------------------------------------------------------- convergence: DEC, least-squares energy star, corrected star
T = [split(l) for l in readlines(joinpath(OUT, "check_table.txt"))[2:end] if !isempty(strip(l))]
lsq = Dict{String,NTuple{4,Float64}}()
for mn in ("mesh2", "mesh3", "mesh4", "mesh5")
    mm = VoronoiMesh("meshes/$mn.nc"); ss = sum(edge_length(mm, e, true) for e in 1:mm.edges.n) / mm.edges.n
    cc = [gs_cell(mm, t, ss) for t in 1:mm.vertices.n]; Hl = ls_star(mm, cc)
    lsq[mn] = (gs_hodge_error(mm, Hl, u1)..., gs_hodge_error(mm, Hl, u2)...)
    logp(@sprintf("   %s least-squares energy star: u1 max %.3e rms %.3e, u2 max %.3e rms %.3e; linear exactness %.2e", mn, lsq[mn]..., gs_row_exactness(mm, Hl)))
end
fig = Figure(size = (1200, 480))
for (i, (k, ttl)) in enumerate(((6, "u₁ (gradient field)"), (8, "u₂ (divergence-free field)")))
    ax = Axis(fig[1, i]; xscale = log10, yscale = log10, xreversed = true, xlabel = "h", ylabel = i == 1 ? "max_e |(Hα(u) - β(u))_e| / |ẽ_e|" : "", title = ttl)
    for (lbl, c, mk, name) in (("DEC", :black, :circle, "DEC"), ("Galerkin", :firebrick, :rect, "corrected energy star"))
        rs = [r for r in T if r[3] == lbl]
        scatterlines!(ax, parse.(Float64, getindex.(rs, 2)), parse.(Float64, getindex.(rs, k)); color = c, marker = mk, label = name)
    end
    hs = [parse(Float64, r[2]) for r in T if r[3] == "DEC"]
    scatterlines!(ax, hs, [lsq[m][i == 1 ? 1 : 3] for m in ("mesh2", "mesh3", "mesh4", "mesh5")]; color = :steelblue, marker = :diamond,
                  label = "least-squares energy star")
    lines!(ax, [0.03125, 0.015625], [4e-3, 1e-3]; color = :gray, linewidth = 2); text!(ax, 0.022, 2.2e-3; text = "h²", color = :gray, align = (:right, :top))
    lines!(ax, [0.03125, 0.015625], [3e-2, 1.5e-2]; color = :gray, linestyle = :dot, linewidth = 2); text!(ax, 0.022, 2.4e-2; text = "h", color = :gray, align = (:right, :bottom))
end
Legend(fig[2, 1:2], fig.content[1]; orientation = :horizontal, framevisible = false)
CairoMakie.save(joinpath(@__DIR__, "..", "..", "output", "figures", "fig_t2_conv.png"), fig; px_per_unit = 2)
write(joinpath(OUT, "tutorial2d.txt"), join(L, "\n") * "\n")
println("done")
