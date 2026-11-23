# ============================================================================
#  locality_subspace_check.jl -- which linear fields force a symmetric exact star to be global?  (barrier_paper.tex)
#
#      julia -t 4 scripts/barrier/locality_subspace_check.jl      (writes output/ts_data/locality_subspace.txt and .jls)
#
#  The interior term of the patch identity is 1/2 Σ |e|^3 |e~| n'Z(G_f, G_g)n, and Z vanishes on the pairs (dilation,
#  rotation) and (shear, shear), but not on (conformal, shear).  So the obstruction to locality couples the conformal
#  linear fields {x, Jx} with the shears {(x,-y), (y,x)}.  Here: the smallest symmetric correction of DEC exact on
#      constants + conformal  (dilation, rotation)       no obstruction predicted
#      constants + shears                                no obstruction predicted
#      all of [P1]^2  (= H_P1)                           obstruction
#  and the reach of one moved generator into each of them (as in Part B of locality_check.jl).
# ============================================================================
ENV["LOC_MAIN"] = "0"
include(joinpath(@__DIR__, "locality_check.jl"))
empty!(lines)

e(i) = (v = zeros(6); v[i] = 1.0; v)                     # basis f1..f6 = (1,0), (0,1), (X,0), (Y,0), (0,X), (0,Y)
const SUBSPACES = (("constants + conformal", hcat(e(1), e(2), e(3) + e(6), e(5) - e(4))),
                   ("constants + shears", hcat(e(1), e(2), e(3) - e(6), e(4) + e(5))),
                   ("all of [P1]^2 (H_P1)", Matrix{Float64}(I, 6, 6)))
R = Dict{Any,Any}()
for mn in ("mesh4", "mesh5")
    mesh = VoronoiMesh("meshes/$mn.nc"); h = meshsize(mesh)
    v0 = argmin([norm(_pt(p) .- [0.5, 0.5]) for p in mesh.cells.position]); x0v = mesh.cells.position[v0]
    ε = 1e-3 * h
    meshes = Dict{Any,Any}(s => moved_mesh(mesh, v0, s .* [ε, 0.0]) for s in (-1, 1))
    geo0 = edge_geometry(mesh, x0v)
    logp(@sprintf("%s  (h = %.5f)", mn, h))
    for (name, B) in SUBSPACES
        H, info = symmetric_correction(mesh, 1, 3; B = B)
        hd = hodge_error(mesh, H, HFIELDS.d...)
        bins = influence(m -> symmetric_correction(m, 1, 3; B = B)[1], meshes, geo0, ε, h)
        R[(mn, name)] = (h = h, bins = bins, res = info.res, cmax = info.cmax, hodge = hd.l2)
        logp(@sprintf("   %-24s residual %.1e  max|C|/max Hd %.3f  Hodge error r_2(d) %.2e", name, info.res, info.cmax, hd.l2))
        logp("      reach, max |dH/dx|·h for r/h = 0, 2, 4, ...: " * join([@sprintf("%.1e", b) for b in bins[1:2:end]], " "))
    end
end
serialize(joinpath(TSDIR, "locality_subspace.jls"), R)
write(joinpath(TSDIR, "locality_subspace.txt"), join(lines, "\n") * "\n")
