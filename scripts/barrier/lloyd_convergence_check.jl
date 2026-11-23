# ============================================================================
#  lloyd_convergence_check.jl -- how many Lloyd steps does one refinement of the SCVT family need?
#
#      julia -t 4 scripts/barrier/lloyd_convergence_check.jl mesh5 4096           ->  output/ts_data/lloyd_mesh5.txt
#      julia -t 4 scripts/barrier/lloyd_convergence_check.jl mesh7 1024 continue  ->  output/ts_data/lloyd_mesh7_continue.txt
#
#  Refines meshes/<base>.nc as meshes/mesh_generator.jl does (generators = cell positions + edge positions, 4N points),
#  then applies Lloyd steps and reports mesh_quality_check.jl's quality() at 0, 1, 2, 3, 4, 6, 8, ..., K steps.
#  With "continue" the mesh is not refined: Lloyd continues from its own generators (is the stored mesh converged?).
#  Lloyd and the mesh construction use meshes/scvt_tools.jl (the same step as VoronoiMeshes, without its O(N^2) parts),
#  with no tolerance and no time limit.  The final generators are saved to ts_data/lloyd_<tag>_<K>.jls.
# ============================================================================
ENV["MQ_MAIN"] = "0"
include(joinpath(@__DIR__, "mesh_quality_check.jl"))
include(joinpath(@__DIR__, "..", "..", "meshes", "scvt_tools.jl"))
empty!(lines)

base = isempty(ARGS) ? "mesh5" : ARGS[1]
K = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 1024
cont = length(ARGS) >= 3 && ARGS[3] == "continue"
tag = cont ? "$(base)_continue" : base
cps = filter(<=(K), sort(unique([0; 2 .^ (0:20); 3 .* 2 .^ (0:20)])))

M0 = VoronoiMesh("meshes/$base.nc")
P = cont ? _pts(M0.cells.position) : _pts(vcat(M0.cells.position, M0.edges.position))
logp(cont ? "Lloyd steps continued from $base ($(length(P)) generators)" :
            "Lloyd steps after refining $base ($(M0.cells.n) -> $(length(P)) generators)")
logp(@sprintf("%6s %8s | %9s %9s | %4s %4s %3s | %6s %6s %5s | %7s %7s | %5s", "steps", "s/step", "res max", "res rms",
              "v5", "v7", "oth", "minang", "maxang", "#obt", "δ max", "δ rms", "Amax/Amin"))
row(k, q, s) = logp(@sprintf("%6d %8.2f | %9.2e %9.2e | %4d %4d %3d | %6.2f %6.2f %5d | %7.1e %7.1e | %5.3f",
                             k, s, q.resmax, q.resrms, q.v5, q.v7, q.vo, q.amin, q.amax, q.nobt, q.δmax, q.δrms, q.Aratio))
R = Dict{Int,Any}()
q0 = quality(voronoi_mesh(P)); R[0] = Base.structdiff(q0, (res = 0, nb = 0)); row(0, q0, 0.0)
done = 0
for k in cps[2:end]
    global P, done
    t = @elapsed ((P, _, _, vor, inds) = lloyd_steps(P, k - done))
    s = t / (k - done); done = k
    q = quality(voronoi_mesh(vor, inds, P)); R[k] = Base.structdiff(q, (res = 0, nb = 0)); row(k, q, s)
end
serialize(joinpath(TSDIR, "lloyd_$(tag)_$K.jls"), (R = R, x = [p.x for p in P], y = [p.y for p in P]))
write(joinpath(TSDIR, "lloyd_$tag.txt"), join(lines, "\n") * "\n")
