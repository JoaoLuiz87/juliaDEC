# ============================================================================
#  mesh_quality_check.jl -- quality of the SCVT family meshes/meshN.nc (and of any other mesh files given)
#
#      julia -t 4 scripts/barrier/mesh_quality_check.jl [mesh1 mesh2 ...]     ->  output/ts_data/mesh_quality.txt
#
#  Per mesh:
#    res     = |x_c - centroid_c| / h  (how far each generator is from the centroid of its Voronoi cell: 0 on an exact CVT;
#              it is the step the next Lloyd iteration would take)
#    val     = number of generators with 5, 6, 7 and other numbers of neighbours
#    angles  = smallest and largest triangle angle, number of obtuse (non-well-centred) triangles
#    δ/|e|   = offset between the midpoints of the Delaunay edge and of its Voronoi edge, relative to |e|
#    |e~|/|e|, area ratio max/min
# ============================================================================
include(joinpath(@__DIR__, "..", "stars", "three_stars.jl"))
using Printf, Statistics
empty!(lines)

function quality(M)
    N = M.cells.n; h = meshsize(M); xp, yp = M.x_period, M.y_period
    res = [norm(_pt(closest(M.cells.position[c], M.cells.centroid[c], xp, yp)) - _pt(M.cells.position[c])) for c in 1:N] ./ h
    nb = [length(M.cells.edges[c]) for c in 1:N]
    amin = Float64[]; amax = Float64[]
    for T in 1:M.vertices.n
        v = M.vertices.position[T]
        P = [_pt(closest(v, M.cells.position[c], xp, yp)) for c in M.vertices.cells[T]]
        a = [(u = P[mod1(i + 1, 3)] - P[i]; w = P[mod1(i + 2, 3)] - P[i]; acos(clamp(dot(u, w) / (norm(u) * norm(w)), -1, 1))) for i in 1:3]
        push!(amin, minimum(a)); push!(amax, maximum(a))
    end
    δ = zeros(M.edges.n); ρ = zeros(M.edges.n)
    for e in 1:M.edges.n
        a, b = edge_endpoints(M, e, true); v1, v2 = edge_endpoints(M, e, false)
        le = norm(b - a); t = (b - a) / le; n = [-t[2], t[1]]
        δ[e] = abs(dot((v1 + v2) / 2 - (a + b) / 2, n)) / le; ρ[e] = norm(v2 - v1) / le
    end
    A = collect(M.cells.area)
    return (N = N, h = h, resmax = maximum(res), resrms = sqrt(mean(res .^ 2)), res = res, nb = nb,
            v5 = count(==(5), nb), v6 = count(==(6), nb), v7 = count(==(7), nb), vo = count(x -> x < 5 || x > 7, nb),
            amin = rad2deg(minimum(amin)), amax = rad2deg(maximum(amax)), nobt = count(>(π / 2 + 1e-12), amax),
            δmax = maximum(δ), δrms = sqrt(mean(δ .^ 2)), ρmin = minimum(ρ), ρmax = maximum(ρ), Aratio = maximum(A) / minimum(A))
end

if get(ENV, "MQ_MAIN", "1") == "1"                 # MQ_MAIN=0 only loads quality() (used by lloyd_convergence_check.jl)
names = isempty(ARGS) ? ["mesh$k" for k in 1:7] : ARGS
logp("mesh quality.  res = |generator - centroid| / h ;  δ = |offset| / |e| ;  ρ = |e~| / |e| ;  A = cell area")
logp(@sprintf("%-10s %7s %8s | %9s %9s | %4s %6s %4s %3s | %6s %6s %5s | %7s %7s | %5s %5s | %5s",
              "mesh", "N", "h", "res max", "res rms", "v5", "v6", "v7", "oth", "minang", "maxang", "#obt", "δ max", "δ rms", "ρmin", "ρmax", "Amax/Amin"))
Q = Dict{String,Any}()
for mn in names
    t = @elapsed (M = VoronoiMesh("meshes/$mn.nc"))
    q = quality(M); Q[mn] = q
    logp(@sprintf("%-10s %7d %8.5f | %9.2e %9.2e | %4d %6d %4d %3d | %6.2f %6.2f %5d | %7.1e %7.1e | %5.3f %5.3f | %5.3f   (load %.0f s)",
                  mn, q.N, q.h, q.resmax, q.resrms, q.v5, q.v6, q.v7, q.vo, q.amin, q.amax, q.nobt, q.δmax, q.δrms, q.ρmin, q.ρmax, q.Aratio, t))
end
serialize(joinpath(TSDIR, "mesh_quality.jls"), Q)
write(joinpath(TSDIR, "mesh_quality.txt"), join(lines, "\n") * "\n")
end
