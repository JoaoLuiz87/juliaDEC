# ============================================================================
#  homogeneous_skew_check.jl -- is the h^2 skew of SCVT meshes a property of unstructured meshes, or of their irregularity?
#
#      julia -t 4 scripts/barrier/homogeneous_skew_check.jl      (writes output/ts_data/homogeneous_skew.txt and .jls)
#
#  A. Replicated families.  M_n = the n x n periodic replication of a fixed Delaunay mesh M_1, scaled by 1/n (h = h_1/n).
#     For u = a cos(k.x), v = b sin(k.x) with 2k not in 2πnZ^2, the sum over replicas is a character sum and gives exactly
#        s_h(u,v) = Σ_{e in M_1} |e||e~| (a.t)(b.t) sinc(|e| k.t / 2n) sinc(|e~| k.n / 2n) sin(δ_e k.n / n)        (sinc x = sin x / x)
#     whose 1/n term is the circumcentric identity T_1 = 0, so  s_h = -T_3[k,k,k,a,b] / (6 n^3) + O(n^-5):  O(h^3), not h^2.
#  B. A two-phase (bicrystal) family: a lattice of spacing 1/n below y = 1/2 and of spacing 1/(2n) above.  Each interface is a
#     fixed motif repeated along x, so interface effects give a skew of order h^2.
#  C. The defects of the SCVT meshes: generators with 5 or 7 neighbours.
# ============================================================================
include(joinpath(@__DIR__, "..", "stars", "three_stars.jl"))
empty!(lines)

mysinc(x) = abs(x) < 1e-12 ? 1.0 : sin(x) / x
function skew_of(mesh, u, v)
    α(w) = reduce_primal_edges(mesh, w).values; β(w) = reduce_dual_edges(mesh, star_field(w)).values
    return dot(α(u), β(v)) - dot(α(v), β(u))
end
"Base-mesh edge geometry: (|e|, |e~|, t, n, δ)."
function base_geometry(mesh)
    map(1:mesh.edges.n) do e
        xe = mesh.edges.position[e]
        a, b = edge_endpoints(mesh, e, true; about = xe); v1, v2 = edge_endpoints(mesh, e, false; about = xe)
        le = norm(b - a); t = (b - a) / le; n = [-t[2], t[1]]
        (le = le, ld = norm(v2 - v1), t = t, n = n, δ = dot((v1 + v2) / 2 - (a + b) / 2, n))
    end
end
exact_formula(G, k, a, b, n) = sum(g.le * g.ld * dot(a, g.t) * dot(b, g.t) * mysinc(g.le * dot(k, g.t) / 2n) * mysinc(g.ld * dot(k, g.n) / 2n) *
                                   sin(g.δ * dot(k, g.n) / n) for g in G)
T3pred(G, k, a, b, n) = -sum(dot(a, g.t) * dot(b, g.t) * g.le * g.ld * g.δ * dot(k, g.n) *
                             (g.δ^2 * dot(k, g.n)^2 + (g.le^2 * dot(k, g.t)^2 + g.ld^2 * dot(k, g.n)^2) / 4) for g in G) / (6n^3)
T1norm(G) = norm([sum(g.le * g.ld * g.δ * g.n[i] * g.t[j] * g.t[l] for g in G) for i in 1:2, j in 1:2, l in 1:2])

const WAVES = ((2π .* [1.0, 0.0], [1.0, 0.3], [-0.2, 1.0]), (2π .* [1.0, 1.0], [0.4, -1.0], [1.0, 0.5]), (2π .* [2.0, 1.0], [1.0, 1.0], [0.3, -0.7]))
R = Dict{Any,Any}()

logp("A. replicated families: s_h(a cos k.x, b sin k.x) on M_n, the exact formula, and the T_3 prediction -T_3/(6n^3)")
for base in ("mesh2", "mesh3")
    M1 = VoronoiMesh("meshes/$base.nc"); G = base_geometry(M1); h1 = meshsize(M1)
    xs = [p.x for p in M1.cells.position]; ys = [p.y for p in M1.cells.position]
    logp(@sprintf("base %s (h1 = %.4f, %d edges): |T_1| = %.1e (circumcentric identity)", base, h1, M1.edges.n, T1norm(G)))
    for n in (1, 2, 4, 8, 16)
        n^2 * M1.cells.n > 20000 && continue
        Mn = n == 1 ? M1 : delaunay_mesh(vcat([(xs .+ i) ./ n for i in 0:n-1, j in 0:n-1]...), vcat([(ys .+ j) ./ n for i in 0:n-1, j in 0:n-1]...))
        h = meshsize(Mn)
        row = String[]
        for (iw, (k, a, b)) in enumerate(WAVES)
            u(x) = a .* cos(dot(k, x)); v(x) = b .* sin(dot(k, x))
            s = skew_of(Mn, u, v); ex = exact_formula(G, k, a, b, n); pr = T3pred(G, k, a, b, n)
            R[(base, n, iw)] = (h = h, s = s, ex = ex, pr = pr)
            push!(row, @sprintf("k%d: s_h %+.4e  exact %+.4e  T3 %+.4e  (s_h/h^3 %+.3f)", iw, s, ex, pr, s / h^3))
        end
        su = skew_of(Mn, U2, fields.d)
        R[(base, n, :ud)] = (h = h, s = su)
        logp(@sprintf("   n = %2d (h = %.5f, %5d cells): ", n, h, Mn.cells.n) * join(row, " | ") * @sprintf(" | s_h(u2,d) %+.1e", su))
    end
end

logp("\nB. bicrystal family: lattice of spacing 1/n below y = 1/2, of spacing 1/(2n) above (n multiple of 4)")
function bicrystal(n)
    xs, ys = Float64[], Float64[]
    for j in 0:n÷2-1, i in 0:n-1                         # lower phase, rows y = (j + 1/2)/n
        push!(xs, (i + j / 2 + 0.25) / n); push!(ys, (j + 0.5) / n)
    end
    for j in 0:n-1, i in 0:2n-1                          # upper phase, rows y = 1/2 + (j + 1/2)/(2n)
        push!(xs, (i + j / 2 + 0.37) / (2n)); push!(ys, 0.5 + (j + 0.5) / (2n))
    end
    return delaunay_mesh(xs, ys)
end
function lattice(n)                                     # control: one phase only
    xs = [(i + j / 2 + 0.25) / n for j in 0:n-1 for i in 0:n-1]; ys = [(j + 0.5) / n for j in 0:n-1 for i in 0:n-1]
    return delaunay_mesh(xs, ys)
end
P(x) = sin(TP * x[1]) * sin(TP * x[2]) + 0.5cos(2TP * x[1]); GP(x) = [TP * cos(TP * x[1]) * sin(TP * x[2]) - TP * sin(2TP * x[1]), TP * sin(TP * x[1]) * cos(TP * x[2])]
GQ(x) = [-TP * sin(TP * x[1]) * cos(2TP * x[2]) + TP * cos(TP * (x[1] + x[2])), -2TP * cos(TP * x[1]) * sin(2TP * x[2]) + TP * cos(TP * (x[1] + x[2]))]
rot(f) = x -> (g = f(x); [-g[2], g[1]])
for (name, build) in (("bicrystal", bicrystal), ("lattice", lattice))
    for n in (4, 8, 16, 32, 64)
        M = build(n); h = meshsize(M); ld = edge_lengths(M, false)
        vals = (skew_of(M, U2, fields.d), skew_of(M, GP, GQ), skew_of(M, rot(GP), rot(GQ)), skew_of(M, GP, rot(GQ)))
        R[(name, n)] = (h = h, vals = vals)
        logp(@sprintf("   %-9s n = %2d (h = %.5f, %5d cells, min |e~|/h %.2f): s_h/h^2 = (u2,d) %+.4f  (∇p,∇q) %+.4f  (J∇p,J∇q) %+.4f  (∇p,J∇q) %+.4f",
                      name, n, h, M.cells.n, minimum(ld) / h, (vals ./ h^2)...))
    end
end

logp("\nC. defects of the SCVT meshes: generators with a number of neighbours other than 6")
for k in 2:6
    M = VoronoiMesh("meshes/mesh$k.nc")
    nb = [length(M.cells.edges[c]) for c in 1:M.cells.n]
    R[("defects", k)] = (pos = [_pt(p) for p in M.cells.position], nb = nb)
    logp(@sprintf("   mesh%d: %5d cells, valence 5: %4d, 6: %5d, 7: %4d, other: %d;  defects per sqrt(N) = %.2f", k, M.cells.n,
                  count(==(5), nb), count(==(6), nb), count(==(7), nb), count(x -> x < 5 || x > 7, nb), count(!=(6), nb) / sqrt(M.cells.n)))
end
serialize(joinpath(TSDIR, "homogeneous_skew.jls"), R)
write(joinpath(TSDIR, "homogeneous_skew.txt"), join(lines, "\n") * "\n")
