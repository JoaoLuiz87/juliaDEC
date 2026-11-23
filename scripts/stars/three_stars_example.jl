# ============================================================================
#  three_stars_example.jl -- the numbers of the worked example (mesh 3, edge 17) in Section 2 of three_stars_report.tex.
#
#      julia scripts/stars/three_stars_example.jl      (after three_stars_run.jl; writes output/ts_data/example.txt)
#
#  Geometry of the edge, the residual of DEC on the basis against the prediction of Lemma [the error of H_DEC on
#  linear fields], and the size of row 17 of H_P1 and H_P2.
# ============================================================================
include(joinpath(@__DIR__, "three_stars.jl"))
empty!(lines)
R = deserialize(joinpath(TSDIR, "results.jls")); ex = R[:example]
mesh = VoronoiMesh("meshes/mesh3.nc"); e = ex[:e]
xe = mesh.edges.position[e]
a, b = edge_endpoints(mesh, e, true; about = xe); v1, v2 = edge_endpoints(mesh, e, false; about = xe)
le, ld = norm(b - a), norm(v2 - v1); t = (b - a) / le; n = [-t[2], t[1]]
δ = dot((v1 + v2) / 2 - (a + b) / 2, n)                              # offset: m~_e = m_e + δ n_e
logp(@sprintf("edge %d of mesh 3: |e| = %.5f  |e~| = %.5f  H_DEC,ee = %.5f  delta = %.5f (%.4f |e|)  t = (%.4f, %.4f)  |S_e| = %d",
              e, le, ld, ld / le, δ, δ / le, t..., length(ex[:S])))
# Lemma: r_m = (R~*f_m)_e - H_DEC,ee (R_h f_m)_e = -eps_e = |e~| δ t'G_m n for the linear basis fields (G_m = gradient of f_m)
s = ex[:s]
Gs = [[1/s 0; 0 0], [0 1/s; 0 0], [0 0; 1/s 0], [0 0; 0 1/s]]       # f3 = (X,0), f4 = (Y,0), f5 = (0,X), f6 = (0,Y)
logp("residual r_e, constants      : " * join([@sprintf("%+.1e", x) for x in ex[:r2][1:2]], " "))
logp("residual r_e, linear fields  : " * join([@sprintf("%+.4e", x) for x in ex[:r2][3:6]], " "))
logp("Lemma |e~| δ t'G_m n         : " * join([@sprintf("%+.4e", ld * δ * dot(t, G * n)) for G in Gs], " "))
logp("residual r_e, quadratic fields: " * join([@sprintf("%+.4e", x) for x in ex[:r2][7:12]], " "))
sv = svdvals(ex[:A2])
logp(@sprintf("A_e for [P2]^2: %d x %d, singular values %.3e .. %.3e, cond %.2f", size(ex[:A2])..., sv[end], sv[1], sv[1] / sv[end]))
for name in ("H_P1", "H_P2")
    row = ex[Symbol("row_" * name)]
    v = sort(collect(row), by = x -> -abs(x[2]))
    logp(@sprintf("%s row %d: %d entries, sum|C|/H_DEC,ee = %.4f, max|C|/H_DEC,ee = %.4f, C_ee/H_DEC,ee = %+.4f, ||c||_2/H_DEC,ee = %.4f, max target error %.1e",
                  name, e, length(row), sum(abs, values(row)) / ex[:hd], abs(v[1][2]) / ex[:hd], row[e] / ex[:hd],
                  norm(collect(values(row))) / ex[:hd], ex[Symbol("check_" * name)]))
    logp("   largest five C_ej/H_DEC,ee: " * join([@sprintf("%d:%+.4f", j, c / ex[:hd]) for (j, c) in v[1:5]], "  "))
end
write(joinpath(TSDIR, "example.txt"), join(lines, "\n") * "\n")
