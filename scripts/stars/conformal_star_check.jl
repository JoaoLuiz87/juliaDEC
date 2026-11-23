# ============================================================================
#  conformal_star_check.jl -- an explicit local symmetric star exact on constants, dilations and rotations
#  (barrier_paper.tex, Section "Locality")
#
#      julia -t 4 scripts/stars/conformal_star_check.jl      (writes output/ts_data/conformal_star.txt and .jls)
#
#      H_conf = H_DEC - D1' diag( (R_T^2 - Rref^2) / (4|T|) ) D1
#
#  Why it works.  D1 R_h u = ∫_T curl u (Stokes on the counter-clockwise triangles), so the correction vanishes on every
#  curl-free field: constants and the dilation x stay exact, as for DEC.  For the rotation u = J(x - x_c), curl u = 2 and
#  the residual of DEC is -|e~| δ_e = -(1/2)(R_2^2 - R_1^2) = -(D1' ρ)_e with ρ_T = R_T^2/2, which the correction
#  reproduces exactly; D1' kills the constant Rref^2, which only keeps the correction small.  Each entry involves the two
#  triangles of an edge, so the star is local, and it is symmetric by construction.
#  The patch identity of locality_check.jl says that no local symmetric rule can be exact on more linear fields than a
#  2-dimensional "isotropic" subspace (dilation + rotation is one); the shears are what require a global solve.
# ============================================================================
ENV["LOC_MAIN"] = "0"
include(joinpath(@__DIR__, "..", "barrier", "locality_check.jl"))
empty!(lines)

function conformal_star(mesh; tris_D1 = triangles_ccw(mesh))
    tris, D1, _ = tris_D1
    R2 = [sum(abs2, T.pts[1] .- T.cc) for T in tris]
    Rref2 = sum(R2) / length(R2)
    m = (R2 .- Rref2) ./ (4 .* [T.area for T in tris])
    return hdiag(mesh) - D1' * Diagonal(m) * D1
end

# rows exact on a field?  (independent check, as T3 of three_stars_tests.jl: own 5-point rule and own tangents)
function row_residual(mesh, H, F, rows)
    Ht = sparse(H'); worst = 0.0
    for e in rows
        xe = mesh.edges.position[e]
        a, b = edge_endpoints(mesh, e, true; about = xe); te = (b .- a) ./ norm(b .- a)
        v1, v2 = edge_endpoints(mesh, e, false; about = xe); le = norm(v2 .- v1)
        Fe = F(_pt(xe))                                                          # the field, centred at m_e
        lhs = 0.0
        for p in nzrange(Ht, e)
            j = rowvals(Ht)[p]; q1, q2 = edge_endpoints(mesh, j, true; about = xe)
            lhs += nonzeros(Ht)[p] * segint(Fe, q1, q2, (q2 .- q1) ./ norm(q2 .- q1))
        end
        worst = max(worst, abs(lhs - segint(Fe, v1, v2, te)) / le)
    end
    return worst
end
const GL5X = [-0.9061798459386640, -0.5384693101056831, 0.0, 0.5384693101056831, 0.9061798459386640]
const GL5W = [0.2369268850561891, 0.4786286704993665, 0.5688888888888889, 0.4786286704993665, 0.2369268850561891]
segint(F, a, b, τ) = norm(b .- a) * sum(w / 2 * dot(F(a .+ (ξ + 1) / 2 .* (b .- a)), τ) for (ξ, w) in zip(GL5X, GL5W))
const TESTF = (("constant (1,0)", c -> (x -> [1.0, 0.0])), ("constant (0,1)", c -> (x -> [0.0, 1.0])),
               ("dilation", c -> (x -> x .- c)), ("rotation", c -> (x -> [-(x[2] - c[2]), x[1] - c[1]])),
               ("shear (x,-y)", c -> (x -> [x[1] - c[1], -(x[2] - c[2])])), ("shear (y,x)", c -> (x -> [x[2] - c[2], x[1] - c[1]])))

R = Dict{Any,Any}()
logp("H_conf = H_DEC - D1' diag((R_T^2 - Rref^2)/(4|T|)) D1: exactness, symmetry, size, positivity, Hodge error")
for mn in MESHES
    mesh = VoronoiMesh("meshes/$mn.nc"); h = meshsize(mesh)
    H = conformal_star(mesh)
    rows = mesh.edges.n <= 3072 ? (1:mesh.edges.n) : (1:10:mesh.edges.n)
    ex = [row_residual(mesh, H, F, rows) for (_, F) in TESTF]
    exd = [row_residual(mesh, hdiag(mesh), F, rows) for (_, F) in TESTF]
    Hd = hdiag(mesh); C = H - Hd
    stars, _ = three_stars(mesh)
    hd = Dict(n => hodge_error(mesh, Hs, HFIELDS.d...) for (n, Hs) in merge(stars, Dict("H_conf" => H)))
    hu = Dict(n => hodge_error(mesh, Hs, HFIELDS.u2...) for (n, Hs) in merge(stars, Dict("H_conf" => H)))
    nnz_row = maximum(length(nzrange(sparse(C'), e)) for e in 1:mesh.edges.n)
    R[mn] = (h = h, ex = ex, exd = ex, hd = hd, hu = hu, g = gersh_sym(mesh, H), Λ = bigLambda(mesh, H), cmax = maximum(abs, C) / maximum(diag(Hd)))
    logp(@sprintf("%s (h = %.4f): symmetric %.0e, entries per row %d, max|C|/max H_DEC %.3f, Gershgorin g %.3f, Λ %.2f", mn, h,
                  norm(H - H') / norm(H), nnz_row, R[mn].cmax, R[mn].g, R[mn].Λ))
    logp("   row residual / |e~| (H_conf | H_DEC):  " * join([@sprintf("%s %.1e | %.1e", n, a, b) for ((n, _), a, b) in zip(TESTF, ex, exd)], ";  "))
    logp("   Hodge error, field d (max / rms):  " * join([@sprintf("%s %.2e / %.2e", n, hd[n].inf, hd[n].l2) for n in ("H_DEC", "H_conf", "H_P1", "H_P2")], ";  "))
    logp("   Hodge error, field u2 (max / rms): " * join([@sprintf("%s %.2e / %.2e", n, hu[n].inf, hu[n].l2) for n in ("H_DEC", "H_conf", "H_P1", "H_P2")], ";  "))
end
logp("\norders between consecutive meshes, field d (max / rms)")
for (a, b) in zip(MESHES[1:end-1], MESHES[2:end])
    o(x, y) = log(x / y) / log(R[a].h / R[b].h)
    logp(@sprintf("   %s->%s  ", a, b) * join([@sprintf("%s %.2f / %.2f", n, o(R[a].hd[n].inf, R[b].hd[n].inf), o(R[a].hd[n].l2, R[b].hd[n].l2)) for n in ("H_DEC", "H_conf", "H_P1", "H_P2")], ";  "))
end

# reach of one moved generator (as Part B of locality_check.jl)
let mesh = VoronoiMesh("meshes/mesh5.nc"), h = meshsize(mesh)
    v0 = argmin([norm(_pt(p) .- [0.5, 0.5]) for p in mesh.cells.position]); x0v = mesh.cells.position[v0]; ε = 1e-3 * h
    meshes = Dict{Any,Any}(s => moved_mesh(mesh, v0, s .* [ε, 0.0]) for s in (-1, 1))
    # the reference radius is a mesh average; freeze it so that the test sees only the local dependence
    tris0, _, _ = triangles_ccw(mesh); Rref2 = sum(sum(abs2, T.pts[1] .- T.cc) for T in tris0) / length(tris0)
    frozen(m) = (t = triangles_ccw(m); tr = t[1]; D1 = t[2];
                 hdiag(m) - D1' * Diagonal([(sum(abs2, T.pts[1] .- T.cc) - Rref2) / (4T.area) for T in tr]) * D1)
    bins = influence(frozen, meshes, edge_geometry(mesh, x0v), ε, h)
    R[:reach] = bins
    logp("\nmesh5, reach of one moved generator into H_conf, max |dH/dx|·h for r/h = 0, 1, 2, ...: " * join([@sprintf("%.1e", b) for b in bins[1:8]], " "))
end
serialize(joinpath(TSDIR, "conformal_star.jls"), R)
write(joinpath(TSDIR, "conformal_star.txt"), join(lines, "\n") * "\n")
