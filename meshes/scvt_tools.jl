# ============================================================================
#  scvt_tools.jl -- Lloyd iterations and mesh construction for large periodic SCVT meshes, using the internals of
#  VoronoiMeshes' DelaunayTriangulation extension.
#
#  Why: VoronoiMesh(points, lx, ly; ...) is slow on fine meshes because of two O(N^2) steps, not because of Lloyd:
#   1. every Lloyd step calls DelaunayTriangulation.triangulate, which in v1.6.6 tests for ghost vertices by scanning the
#      whole vertex set at each insertion: 1.5 s at 20k points, 70 s at 70k.  Fixed below by an O(1) membership test
#      (and a BRIO/Morton insertion order): 0.4 s per Lloyd step at 16k generators, 2 s at 65k, 11 s at 262k.
#   2. after Lloyd, extract_periodic_vertices_and_cells finds every Voronoi vertex by a linear search over all vertices:
#      ~1 s at 16k, ~10 minutes at 65k, hours at 262k.  fast_extract below uses a hash table instead (0.5 s at 262k).
#  Both were checked against VoronoiMeshes: identical connectivity, generators equal to 1e-15 after 5 Lloyd steps.
#
#      lloyd_steps(P, k)   -> (P', res, steps, vor, inds)   k Lloyd steps from generators P; res = max |x - centroid| / h
#      voronoi_mesh(P)     -> VoronoiMesh of the generators P, no Lloyd step
#      voronoi_mesh(vor, inds, P)                               the same from a tessellation lloyd_steps returned
#      defects(vor, N)     -> numbers of 5- and 7-sided cells
# ============================================================================
using NCDatasets, VoronoiMeshes, TensorsLite, TensorsLiteGeometry, DelaunayTriangulation, SmallCollections, Random
const DTE = Base.get_extension(VoronoiMeshes, :DelaunayTriangulationExt)
const DT = DelaunayTriangulation

# DelaunayTriangulation 1.6.6: has_ghost_vertices scans the whole vertex Set, and point location calls it at every
# insertion, so triangulate is O(N^2) (70 s for 70k random points).  In an unconstrained triangulation the only ghost
# vertex is 𝒢 = -1, so membership is an O(1) test.  (Only used here, where all triangulations are unconstrained.)
DT.has_ghost_vertices(G::DT.Graph) = DT.𝒢 ∈ DT.get_vertices(G)

_pts(P) = TensorsLite.VecArray(x = [mod(p.x, 1.0) for p in P], y = [mod(p.y, 1.0) for p in P])

"Insertion order for the incremental Delaunay algorithm: random rounds of doubling size (BRIO), each sorted along a Morton curve."
function brio_order(q; rng = MersenneTwister(1))
    xmin, xmax = extrema(first, q); ymin, ymax = extrema(last, q); s = max(xmax - xmin, ymax - ymin)
    function morton(p)
        xi = floor(UInt64, (p[1] - xmin) / s * (2^20 - 1)); yi = floor(UInt64, (p[2] - ymin) / s * (2^20 - 1)); k = UInt64(0)
        for b in 0:19; k |= ((xi >> b) & 1) << (2b) | ((yi >> b) & 1) << (2b + 1); end
        return k
    end
    key = map(morton, q); perm = randperm(rng, length(q)); out = Int[]; lo = 1; r = 1
    while lo <= length(q)
        hi = min(length(q), lo + r - 1); c = perm[lo:hi]; append!(out, c[sortperm(key[c])]); lo = hi + 1; r *= 2
    end
    return out
end

function _tessellation(pts; lx = 1.0, ly = 1.0)
    p, inds = DTE.expand_periodic_points(pts, lx, ly, (1, 1))
    q = reinterpret(NTuple{2, Float64}, p)
    tri = DT.triangulate(q; insertion_order = brio_order(q))
    return tri, DT.voronoi(tri, clip = true), inds
end

"""
k Lloyd steps from the generators P (the same step as VoronoiMeshes: centroid of every clipped periodic Voronoi cell),
no tolerance and no time limit.  Returns the new generators, res = max |x - centroid| / h for them, k, and their
tessellation (vor, inds) for voronoi_mesh.
"""
function lloyd_steps(P, k::Integer; lx = 1.0, ly = 1.0)
    pts = _pts(P); N = length(pts); c = similar(pts); pl = Vector{Float64}(undef, N)
    _, vor, inds = _tessellation(pts; lx, ly)
    for _ in 1:k
        DTE.fill_with_polygon_mass_centroids!(c, pl, N, lx, ly, vor)
        pts = _pts(c)
        _, vor, inds = _tessellation(pts; lx, ly)
    end
    DTE.fill_with_polygon_mass_centroids!(c, pl, N, lx, ly, vor)
    res = maximum(1:N) do i
        q = closest(pts[i], c[i], lx, ly); hypot(q.x - pts[i].x, q.y - pts[i].y)
    end / sqrt(lx * ly / N)
    return pts, res, k, vor, inds
end

defects(vor, N) = (n = [length(vor.polygons[c]) - 1 for c in 1:N]; (count(==(5), n), count(==(7), n)))

"""
O(N) replacement of DTE.extract_periodic_vertices_and_cells: the Voronoi vertices inside [0,lx)×[0,ly) are hashed on a
grid of spacing 1e-9, and every polygon corner is looked up there (3×3 neighbouring keys, periodic); triangle corners
are mapped to base generators through the index map `inds` of expand_periodic_points.
"""
function fast_extract(N, lx, ly, vor, inds)
    vp = vor.polygon_points
    interior = filter(i -> (0 <= vp[i][1] < lx) && (0 <= vp[i][2] < ly), eachindex(vp))
    g = 1e-9; nx = round(Int, lx / g); ny = round(Int, ly / g)
    key(x, y) = (mod(round(Int, x / g), nx), mod(round(Int, y / g), ny))
    D = Dict{Tuple{Int, Int}, Int32}(); sizehint!(D, length(interior))
    for (j, i) in enumerate(interior)
        k = key(vp[i]...); haskey(D, k) && error("two Voronoi vertices closer than 1e-9 (degenerate tessellation)"); D[k] = j
    end
    function look(x, y)
        kx, ky = key(x, y)
        for dx in (0, -1, 1), dy in (0, -1, 1)
            j = get(D, (mod(kx + dx, nx), mod(ky + dy, ny)), Int32(0))
            j != 0 && return j
        end
        error("Voronoi vertex ($x, $y) has no copy inside the periodic domain")
    end
    verts = TensorsLite.VecArray(x = [vp[i][1] for i in interior], y = [vp[i][2] for i in interior])
    T = SmallVector{16, Int32}
    voc = Vector{T}(undef, N)
    Threads.@threads for c in 1:N
        poly = vor.polygons[c]
        voc[c] = T([look(vp[i]...) for i in poly[1:end-1]])
    end
    cov = Vector{FixedVector{3, Int32}}(undef, length(interior))
    for (j, i) in enumerate(interior)
        a, b, c = vor.circumcenter_to_triangle[i]
        cov[j] = FixedVector{3, Int32}((inds[a], inds[b], inds[c]))
    end
    return verts, voc, cov
end

function voronoi_mesh(vor, inds, P; lx = 1.0, ly = 1.0)
    gens = _pts(P); N = length(gens)
    verts, voc, cov = fast_extract(N, lx, ly, vor, inds)
    M = maximum(length, voc); M > 10 && error("polygon with more than 10 sides")
    vocM = VoronoiMeshes.SmVecArray{M, Int32}(N); vocM .= voc
    pd = VoronoiMeshes.PlanarVoronoiDiagram(gens, verts, vocM, cov, ones(N), Float64(lx), Float64(ly))
    return VoronoiMesh(VoronoiMeshes.VoronoiDiagram(VoronoiMeshes.fix_diagram!(pd)))
end
function voronoi_mesh(P; lx = 1.0, ly = 1.0)
    pts = _pts(P); _, vor, inds = _tessellation(pts; lx, ly)
    return voronoi_mesh(vor, inds, pts; lx, ly)
end
