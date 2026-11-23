# ============================================================================
#  mesh_generator_scvt.jl -- next level of the SCVT family, refined the way mesh_generator.jl does it:
#      generators of mesh(k+1) = cell positions + edge positions of mesh(k)  (4N points), then Lloyd.
#
#      cd juliaDEC ; julia -t 4 meshes/mesh_generator_scvt.jl mesh7 mesh8 [tol] [chunk]      ->  meshes/mesh8.nc
#
#  Differences from mesh_generator.jl:
#   * Lloyd and the mesh construction use scvt_tools.jl: the same Lloyd step as VoronoiMeshes, but without its two
#     O(N^2) parts (DelaunayTriangulation's ghost-vertex test, and the vertex search that builds the mesh), so a step on
#     262k generators takes ~11 s and building the mesh < 1 s.
#   * Lloyd stops when every generator is within tol*h of the centroid of its cell (default tol = 1e-5), not after a
#     fixed max_iter, and there is no time limit (VoronoiMeshes' default max_time = 4 minutes stopped the fine levels).
#     Before that, Lloyd goes through a transient in which 5-7 pairs form and obtuse triangles come and go; the
#     residual there stays above ~1e-3.  Once it is below save_tol = 1e-4 the transient is over, and from then on the
#     current mesh is written to meshes/<out>.nc after every chunk (so a usable mesh exists before full convergence).
#   * Lloyd runs in chunks of `chunk` steps (default 100); after each chunk the generators are saved to
#     meshes/<out>.lloyd.jls, and a killed run continues from there when restarted with the same arguments.
#   * Progress (steps, residual, 5/7 defects, wall time) goes to meshes/<out>.lloyd.txt.
# ============================================================================
include(joinpath(@__DIR__, "scvt_tools.jl"))
using Serialization, Printf
cd(joinpath(@__DIR__, ".."))                           # relative paths: NCDatasets cannot open paths with non-ASCII characters

base, out = ARGS[1], ARGS[2]
tol = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 1e-5
chunk = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 100
save_tol = max(tol, 1e-4)
ckpt, logf = "meshes/$out.lloyd.jls", "meshes/$out.lloyd.txt"
say(s) = (println(s); flush(stdout); open(io -> println(io, s), logf, "a"))

if isfile(ckpt)
    C = deserialize(ckpt); P = TensorsLite.VecArray(x = C.x, y = C.y); steps = C.steps; tw = C.time
    say("continuing $out from $ckpt: $(length(P)) generators, $steps Lloyd steps done")
else
    M0 = VoronoiMesh("meshes/$base.nc"); P = _pts(vcat(M0.cells.position, M0.edges.position)); steps = 0; tw = 0.0
    say("$out = refinement of $base: $(M0.cells.n) -> $(length(P)) generators; Lloyd until res max < $tol (chunks of $chunk)")
end

while true
    global P, steps, tw
    t = @elapsed ((P, r, _, vor, inds) = lloyd_steps(P, chunk))
    steps += chunk; tw += t
    n5, n7 = defects(vor, length(P))
    say(@sprintf("steps %6d   res max %.2e   valence 5/7: %d/%d   %.2f s/step   total %.2f h", steps, r, n5, n7, t / chunk, tw / 3600))
    serialize(ckpt, (x = [p.x for p in P], y = [p.y for p in P], steps = steps, time = tw))
    if r < save_tol
        VoronoiMeshes.save("meshes/$out.nc", voronoi_mesh(vor, inds, P))
        say("   wrote meshes/$out.nc (res max $(round(r, sigdigits = 2)), $steps Lloyd steps)")
    end
    r < tol && break
end
say("done: meshes/$out.nc after $steps Lloyd steps ($(round(tw / 3600, digits = 2)) h)")
