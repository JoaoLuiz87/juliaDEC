# ============================================================================
#  story_realrow.jl -- why symmetry makes H_P1 global, on the REAL stencil (hodge_story.tex, Section "in detail")
#
#      julia -t 4 scripts/barrier/story_realrow.jl
#          -> fig_st_realrow.png    mesh 4: (a) row e of H_P1 on its stencil, (b) the entry H_ej is shared by rows e and j
#             fig_st_realmove.png   mesh 4: one generator moved: which rows change, row by row (no symmetry) and H_P1; decay
#             fig_st_realmatrix.png mesh 3: the matrix coupling the rows' conditions, without and with symmetry, and its inverse
#             ts_data/story_realrow.txt (numbers quoted in the text), ts_data/story_realrow.jls
#
#  Case A (no symmetry):  H_P1^row = H_DEC + C, each row the least change exact on [P1]^2 on the 3-ring (rowwise_correction, k = 1)
#  Case B (symmetry):     H_P1     = H_DEC + C, C = C', the least change exact on [P1]^2 (symmetric_correction, k = 1)
# ============================================================================
ENV["LOC_MAIN"] = "0"; include(joinpath(@__DIR__, "locality_check.jl"))
include(joinpath(@__DIR__, "..", "stars", "stars_cache.jl"))
include(joinpath(@__DIR__, "..", "shared", "thesis_plot_helpers.jl"))
using Serialization
OUTR = String[]; say(s) = (println(s); flush(stdout); push!(OUTR, s))
const CACHE = joinpath(TSDIR, "story_realrow.jls")

rowP1(mesh) = rowwise_correction(mesh, hdiag(mesh), 1, 3)[1]
symP1(mesh) = symmetric_correction(mesh, 1, 3)[1]
rowP2(mesh) = rowwise_correction(mesh, hdiag(mesh), 2, 3)[1]
rowvec(H, e) = (v = H[e, :]; (v.nzind, v.nzval))

"The constraint matrix of the symmetric construction (k = 1, 3-ring): G (6 rows per edge, one column per pair), weights, pairs."
function constraint_matrix(mesh)
    n = mesh.edges.n; N = 6; lp = edge_lengths(mesh, true); ld = edge_lengths(mesh, false)
    S = [ring(mesh, e, 3) for e in 1:n]; nb = [Set(S[e]) for e in 1:n]
    for e in 1:n, j in S[e]; push!(nb[j], e); end
    nbv = [sort(collect(nb[e])) for e in 1:n]
    pid = Dict{Tuple{Int,Int},Int}()
    for e in 1:n, j in nbv[e]; key = minmax(e, j); haskey(pid, key) || (pid[key] = length(pid) + 1); end
    I_, J_, V_ = Int[], Int[], Float64[]
    for e in 1:n
        xv = mesh.edges.position[e]; x0 = _pt(xv); s = lp[e]
        for j in nbv[e]
            p1_, p2_ = edge_endpoints(mesh, j, true; about = xv); a = seg_moments(p1_, p2_, x0, s, 1)
            for m in 1:N; push!(I_, (e - 1) * N + m); push!(J_, pid[minmax(e, j)]); push!(V_, a[m] / ld[e]); end
        end
    end
    G = sparse(I_, J_, V_, n * N, length(pid))
    w = zeros(length(pid)); for (k_, p) in pid; w[p] = k_[1] == k_[2] ? 1.0 : 2.0; end
    return G, w, S, nbv
end

# ============================================================================ computations (cached)
D = isfile(CACHE) ? deserialize(CACHE) : Dict{Symbol,Any}()
if !haskey(D, :done)
    # ---------------------------------------------------------------- mesh 4: the row, its stencil, its six conditions
    mesh = VoronoiMesh("meshes/mesh4.nc"); h = meshsize(mesh); St = cached_stars("mesh4"; mesh)
    H1 = St["H_P1"]; Hd = St["H_DEC"]; n = mesh.edges.n
    vc = argmin([norm(_pt(p) .- [0.5, 0.5]) for p in mesh.cells.position]); xcv = mesh.cells.position[vc]
    geoc = edge_geometry(mesh, xcv)
    nearc = [k for k in 1:n if geoc[k].r < 3h]
    e = nearc[argmax([norm(geoc[k].mt .- geoc[k].m) for k in nearc])]        # near the centre, the edge with the largest offset
    pa = geoc[e].m .- (geoc[e].le / 2) .* geoc[e].t                           # one end of e: the generator that will move
    v0 = argmin([norm(_pt(closest(xcv, p, 1.0, 1.0)) .- pa) for p in mesh.cells.position]); x0v = mesh.cells.position[v0]; x0 = _pt(x0v)
    geo0 = edge_geometry(mesh, x0v)
    jcols, hv = rowvec(H1, e); hee = H1[e, e]
    off = [(j, v) for (j, v) in zip(jcols, hv) if j != e]
    jn = off[argmax([abs(v) for (_, v) in off])][1]            # the neighbour with the largest coupling
    jcols_n, _ = rowvec(H1, jn)
    r3 = ring(mesh, e, 3)
    # six conditions of row e: fields about the midpoint m_e, linear ones scaled by h
    E = geo0[e]; me = E.m
    F = [(x -> [1.0, 0.0], "(1,0)"), (x -> [0.0, 1.0], "(0,1)"), (x -> [(x[1] - me[1]) / h, 0.0], "(x_1,0)/h"),
         (x -> [(x[2] - me[2]) / h, 0.0], "(x_2,0)/h"), (x -> [0.0, (x[1] - me[1]) / h], "(0,x_1)/h"), (x -> [0.0, (x[2] - me[2]) / h], "(0,x_2)/h")]
    cond = map(F) do (f, name)
        Rf = [G.le * dot(G.t, f(G.m)) for G in geo0]
        exact = E.ld * dot(E.t, f(E.mt))
        (name = name, exact = exact / E.ld, dec = Hd[e, e] * Rf[e] / E.ld, p1 = sum(H1[e, j] * Rf[j] for j in jcols) / E.ld)
    end
    off_e = norm(E.mt .- E.m) / h                             # offset of the Voronoi midpoint from the Delaunay midpoint
    D[:row] = (h = h, e = e, jn = jn, v0 = v0, x0 = x0, xe = me, hee = hee, hd = Hd[e, e], cols = jcols, vals = hv, cols_n = jcols_n,
               r3 = r3, cond = cond, offset = off_e, nnz_mean = nnz(H1) / n, ring_mean = sum(length(ring(mesh, k, 3)) for k in 1:n) / n,
               geo = geo0)
    # ---------------------------------------------------------------- mesh 4: move generator v0, which rows change?
    ε = 1e-3 * h
    mp = moved_mesh(mesh, v0, [ε, 0.0]); mm = moved_mesh(mesh, v0, [-ε, 0.0])
    gp = edge_geometry(mp, x0v)
    changed = [k for k in 1:n if abs(gp[k].ld - geo0[k].ld) > 1e-15 || abs(gp[k].le - geo0[k].le) > 1e-15]
    rowch = Dict{String,Vector{Float64}}()
    for (name, build) in (("H_DEC", hdiag), ("H_P1row", rowP1), ("H_P1", symP1), ("H_P2", rowP2))
        dH = (build(mp) - build(mm)) / (2ε) * h
        dt = sparse(dH'); rc = zeros(n)
        for k in 1:n
            r_ = nzrange(dt, k); isempty(r_) || (rc[k] = maximum(abs, nonzeros(dt)[r_]))
        end
        rowch[name] = rc
        say(@sprintf("mesh4 move: %-8s rows changed %5d of %d, farthest changed row at %.1f h, max change %.2e", name, count(>(0), rc), n,
                     maximum((geo0[k].r for k in 1:n if rc[k] > 0); init = 0.0) / h, maximum(rc)))
    end
    D[:move] = (changed = changed, rowch = rowch, eps = ε)
    say(@sprintf("mesh4 move: %d edges change their geometry, all within %.2f h of the generator", length(changed),
                 maximum(geo0[k].r for k in changed) / h))
    # ---------------------------------------------------------------- mesh 3: the coupling matrices
    m3 = VoronoiMesh("meshes/mesh3.nc"); h3 = meshsize(m3); n3 = m3.edges.n
    G, w, S3, nbv3 = constraint_matrix(m3)
    K = G * Diagonal(1 ./ w) * G'
    KA = spzeros(6n3, 6n3)
    blocks = Vector{Matrix{Float64}}(undef, n3)
    for e3 in 1:n3
        rr = (e3 - 1) * 6 + 1:e3 * 6; Ge = G[rr, :]; blocks[e3] = Matrix(Ge * Ge')
    end
    KA = blockdiag((sparse(b) for b in blocks)...)
    # K is singular: for a symmetric star the conditions of the two constant fields are tied together over the whole torus
    # (sum over all edges of (Rf)(HRg) - (Rg)(HRf) = 0), so use the pseudo-inverse (the solver's δI only regularises this)
    ev = eigen(Symmetric(Matrix(K)))
    keep = ev.values .> 1e-10 * maximum(ev.values)
    Ki = ev.vectors[:, keep] * Diagonal(1 ./ ev.values[keep]) * ev.vectors[:, keep]'
    say(@sprintf("mesh3 K: %d conditions, %d zero eigenvalues (null space), smallest kept eigenvalue / largest %.1e", size(K, 1), count(!, keep),
                 minimum(ev.values[keep]) / maximum(ev.values)))
    D[:null3] = count(!, keep)
    mids3 = [_pt(p) for p in m3.edges.position]
    perm = sortperm(1:n3; by = k -> (floor(Int, mids3[k][2] / h3), mids3[k][1]))
    bmax(A) = begin
        B = zeros(n3, n3)
        if A isa SparseMatrixCSC
            for (i, j, v) in zip(findnz(A)...); bi, bj = (i - 1) ÷ 6 + 1, (j - 1) ÷ 6 + 1; B[bi, bj] = max(B[bi, bj], abs(v)); end
        else
            for bj in 1:n3, bi in 1:n3; B[bi, bj] = maximum(abs, view(A, (bi-1)*6+1:bi*6, (bj-1)*6+1:bj*6)); end
        end
        B[perm, perm]
    end
    BA, BK, BKi = bmax(KA), bmax(K), bmax(Ki)
    D[:mat] = (BA = BA, BK = BK, BKi = BKi, n = n3)
    say(@sprintf("mesh3 matrices: %d rows (%d conditions), %d shared pairs; nonzero blocks per row: A %.1f, K %.1f, K^-1 %.1f; smallest |K^-1| block / largest %.1e",
                 n3, 6n3, size(G, 2), count(>(0), BA) / n3, count(>(0), BK) / n3, count(>(0), BKi) / n3, minimum(BKi) / maximum(BKi)))
    D[:done] = true
    serialize(CACHE, D)
end

# ============================================================================ numbers for the text
R = D[:row]; h = R.h
say(@sprintf("mesh4 row e = %d at (%.3f, %.3f), |e| = %.3f h, |e~| = %.3f h, offset |m~ - m| = %.3f h; H_DEC = %.4f, H_P1 diagonal = %.4f",
             R.e, R.xe..., R.geo[R.e].le / h, R.geo[R.e].ld / h, R.offset, R.hd, R.hee))
say(@sprintf("   row e of H_P1: %d entries (1 diagonal + %d off-diagonal); 3-ring of e: %d edges; mean entries per row %.1f, mean 3-ring %.1f",
             length(R.cols), length(R.cols) - 1, length(R.r3), R.nnz_mean, R.ring_mean))
offs = sort([abs(v) for (j, v) in zip(R.cols, R.vals) if j != R.e]; rev = true)
say(@sprintf("   largest off-diagonal |H_ej|/H_ee: %s", join([@sprintf("%.3f", v / R.hee) for v in offs[1:5]], ", ")))
say(@sprintf("   neighbour j = %d, H_ej = %.4f; row j has %d entries; shared stencil edges %d", R.jn, R.vals[findfirst(==(R.jn), R.cols)],
             length(R.cols_n), length(intersect(R.cols, R.cols_n))))
for c in R.cond
    say(@sprintf("   field %-10s exact flux/|e~| %+.5f   H_DEC %+.5f (error %+.2e)   H_P1 %+.5f (error %+.1e)", c.name, c.exact, c.dec, c.dec - c.exact,
                 c.p1, c.p1 - c.exact))
end

# ============================================================================ figure 1: the row and the shared entry
geo = R.geo; xe = R.xe; W = 3.4h
seg(G) = (a = G.m .- (G.le / 2) .* G.t; b = G.m .+ (G.le / 2) .* G.t; ([a[1], b[1]], [a[2], b[2]]))
near(G, W) = maximum(abs.(G.m .- xe)) < W + G.le
f1 = Figure(size = (1500, 720), fontsize = 17)
ax1 = Axis(f1[1, 1]; aspect = DataAspect(), title = @sprintf("(a) row e of H_P1: %d entries, 6 conditions", length(R.cols)), titlealign = :left)
ax2 = Axis(f1[1, 3]; aspect = DataAspect(), title = "(b) one entry, two rows: H_ej = H_je", titlealign = :left)
for ax in (ax1, ax2)
    hidedecorations!(ax); hidespines!(ax)
    for G in geo
        near(G, W) || continue; xs, ys = seg(G); lines!(ax, xs, ys; color = (:gray, 0.45), linewidth = 0.9)
    end
end
vals = Dict(zip(R.cols, R.vals)); m = maximum(abs(v) for (j, v) in vals if j != R.e) / R.hee
for j in R.cols
    j == R.e && continue; xs, ys = seg(geo[j])
    lines!(ax1, xs, ys; color = vals[j] / R.hee, colormap = DIVM, colorrange = (-m, m), linewidth = 5)
end
xs, ys = seg(geo[R.e]); lines!(ax1, xs, ys; color = INK, linewidth = 7)
v1, v2 = R.geo[R.e].mt .- (R.geo[R.e].ld / 2) .* R.geo[R.e].n, R.geo[R.e].mt .+ (R.geo[R.e].ld / 2) .* R.geo[R.e].n
lines!(ax1, [v1[1], v2[1]], [v1[2], v2[2]]; color = VIOLET, linewidth = 4)
text!(ax1, v2[1], v2[2]; text = " ẽ", fontsize = 22, align = (:left, :bottom), color = VIOLET)
text!(ax1, xe[1], xe[2]; text = "e  ", fontsize = 22, align = (:right, :top), color = INK)
Colorbar(f1[1, 2], colormap = DIVM, limits = (-m, m), label = "H_ej / H_ee   (H_ee = $(round(R.hee, digits = 3)))", width = 14)
inE, inJ = Set(R.cols), Set(R.cols_n)
for G_ in union(inE, inJ)
    col = (G_ in inE && G_ in inJ) ? VIOLET : G_ in inE ? C_P2 : C_P1
    local xs, ys = seg(geo[G_]); lines!(ax2, xs, ys; color = col, linewidth = 4.5)
end
for (k, lab) in ((R.e, "e"), (R.jn, "j"))
    local xs, ys = seg(geo[k]); lines!(ax2, xs, ys; color = INK, linewidth = 7)
    text!(ax2, geo[k].m[1], geo[k].m[2]; text = "  " * lab, fontsize = 22, align = (:left, :bottom), color = INK)
end
a, b = geo[R.e].m, geo[R.jn].m; c = (a .+ b) ./ 2 .+ 0.6 .* norm(b .- a) .* [-(b[2] - a[2]), b[1] - a[1]] ./ norm(b .- a)
bz = [(1 - s)^2 .* a .+ 2s * (1 - s) .* c .+ s^2 .* b for s in range(0, 1; length = 40)]
lines!(ax2, first.(bz), last.(bz); color = INK, linewidth = 2.5)
lp_ = [xe[1] + 0.3W, xe[2] - 0.62W]; mid_ = bz[20]
lines!(ax2, [lp_[1], mid_[1]], [lp_[2], mid_[2]]; color = INK2, linewidth = 1.2, linestyle = :dot)
text!(ax2, lp_[1], lp_[2]; text = "H_ej = H_je:\none number, used by\nthe 6 conditions of row e\nand the 6 conditions of row j",
      fontsize = 16, align = (:left, :top), color = INK)
Legend(f1[1, 4], [LineElement(color = C_P2, linewidth = 5), LineElement(color = C_P1, linewidth = 5), LineElement(color = VIOLET, linewidth = 5)],
       ["entries of row e only", "entries of row j only", "in both rows"]; framevisible = false)
for ax in (ax1, ax2); xlims!(ax, xe[1] - W, xe[1] + W); ylims!(ax, xe[2] - W, xe[2] + W); end
CairoMakie.save(thesis_out("fig_st_realrow.png"), f1; px_per_unit = 2)

# ============================================================================ figure 2: one generator moved
M = D[:move]; x0 = R.x0
CHG = colorant"#c23b3a"
f2 = Figure(size = (1600, 900), fontsize = 16)
gm = f2[1, 1] = GridLayout()
for (idx, (name, lab)) in enumerate((("H_DEC", "H_DEC"), ("H_P1row", "no symmetry, exact on [P1]²"), ("H_P2", "H_P2 (no symmetry)"), ("H_P1", "H_P1 (symmetric)")))
    rc = M.rowch[name]; nch = count(>(0), rc)
    ax = Axis(gm[(idx - 1) ÷ 2 + 1, (idx - 1) % 2 + 1]; aspect = DataAspect(), titlealign = :left, titlesize = 15,
              title = "($('a' + idx - 1)) $lab: $nch of $(length(rc)) rows change")
    hidedecorations!(ax)
    for k in eachindex(rc)                                           # unchanged rows first, changed rows on top
        rc[k] > 0 && continue
        local xs, ys = seg(geo[k]); lines!(ax, xs, ys; color = (:gray, 0.3), linewidth = 0.7)
    end
    for k in eachindex(rc)
        rc[k] > 0 || continue
        local xs, ys = seg(geo[k]); lines!(ax, xs, ys; color = CHG, linewidth = 1.5)
    end
    scatter!(ax, [x0[1]], [x0[2]]; marker = :star5, markersize = 17, color = :black, strokecolor = :white, strokewidth = 1)
    xlims!(ax, x0[1] - 0.5, x0[1] + 0.5); ylims!(ax, x0[2] - 0.5, x0[2] + 0.5)
end
Legend(gm[3, 1:2], [LineElement(color = CHG, linewidth = 3), LineElement(color = (:gray, 0.5), linewidth = 2),
                    MarkerElement(marker = :star5, color = :black, markersize = 15)],
       ["row changes", "row does not change (exactly the same entries)", "moved generator"]; orientation = :horizontal, framevisible = false)
ax3 = Axis(f2[1, 2]; xscale = log10, yscale = log10, xlabel = "distance r/h of the row from the moved generator",
           ylabel = "largest change of an entry of the row (per unit displacement × h)",
           title = "(e) how much the rows change, against distance", axstyle_th...,
           xticks = ([1, 2, 5, 10, 20, 40], ["1", "2", "5", "10", "20", "40"]))
for (name, lab, colr, mk) in (("H_DEC", "H_DEC (mesh 4)", C_DEC, :circle), ("H_P1row", "no symmetry, exact on [P1]² (mesh 4)", AQUA, :utriangle),
                              ("H_P2", "H_P2 (mesh 4)", C_P2, :rect), ("H_P1", "H_P1 (mesh 4)", C_P1, :diamond))
    rc = M.rowch[name]; bins = zeros(23)
    for k in eachindex(rc); local bb = Int(floor(geo[k].r / h)) + 1; bb <= 23 && (bins[bb] = max(bins[bb], rc[k])); end
    rr = (0:22) .+ 0.5; ok = bins .> 0
    scatterlines!(ax3, rr[ok], bins[ok]; color = colr, marker = mk, markersize = 9, linewidth = 2, label = lab)
end
# mesh 5 (torus half-diagonal 45h instead of 22.6h): the same quantity, from locality_check.jl (ts_data/locality.jls)
L5 = deserialize(joinpath(TSDIR, "locality.jls"))[("mesh5", :B, "H_P1")].bins
r5 = collect(0.5:1:length(L5) - 0.5); ok5 = L5 .> 0
scatterlines!(ax3, r5[ok5], L5[ok5]; color = C_P1, marker = :diamond, markersize = 7, linewidth = 1.5, linestyle = :dash, label = "H_P1 (mesh 5)")
rr = [5.0, 40.0]; b0 = 0.25 * L5[6]                                  # guide below the mesh-5 curve, slope -3
lines!(ax3, rr, b0 .* (rr ./ 5) .^ -3; color = INK2, linestyle = :dot)
text!(ax3, 10, b0 * (10 / 5)^-3; text = "(h/r)³  ", align = (:right, :top), color = INK2)
axislegend(ax3; position = :lb, framevisible = false, labelsize = 13)
colsize!(f2.layout, 2, Relative(0.42))
CairoMakie.save(thesis_out("fig_st_realmove.png"), f2; px_per_unit = 2)

# ============================================================================ figure 3: the coupling matrices (mesh 3)
Mt = D[:mat]
f3 = Figure(size = (1560, 560), fontsize = 16)
gmax = maximum(Mt.BK); lo3 = -3.0
for (col, B, ttl) in ((1, Mt.BA, "(a) no symmetry: each row alone"), (2, Mt.BK, "(b) symmetry: rows coupled by shared entries"),
                      (3, Mt.BKi, "(c) inverse of (b): every row reaches every row"))
    ax = Axis(f3[1, col]; aspect = DataAspect(), yreversed = true, title = ttl, titlealign = :left, titlesize = 16,
              xlabel = "row j (edges ordered by position)", ylabel = col == 1 ? "row e" : "")
    hidedecorations!(ax; label = false)
    Bn = B ./ maximum(B); L = [v > 0 ? max(log10(v), lo3) : NaN for v in Bn]
    heatmap!(ax, 1:Mt.n, 1:Mt.n, L'; colormap = Reverse(:viridis), colorrange = (lo3, 0), nan_color = :white)
end
Colorbar(f3[1, 4]; colormap = Reverse(:viridis), limits = (lo3, 0), label = "log₁₀ |block| / largest (white: zero)", width = 12)
CairoMakie.save(thesis_out("fig_st_realmatrix.png"), f3; px_per_unit = 2)
write(joinpath(TSDIR, "story_realrow.txt"), join(OUTR, "\n") * "\n")
println("real-row figures written")
