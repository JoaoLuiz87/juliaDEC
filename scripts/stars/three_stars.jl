# ============================================================================
#  three_stars.jl -- the three Hodge stars of three_stars_report.tex and the problems they are tested on.
#  Library only (no output); used by three_stars_tests.jl, three_stars_run.jl and three_stars_figures.jl.
#
#      H_DEC : diag(|e~|/|e|)                                                       exact on [P0]^2
#      H_P1  : DEC + the smallest SYMMETRIC correction exact on [P1]^2, 3-ring       (symmetric_correction, commuting_check.jl)
#      H_P2  : DEC + the smallest ROW-WISE correction exact on [P2]^2, 3-ring        (rowwise_correction, nonsymmetric_check.jl)
#
#  All three map circulations along the Delaunay edges, (R_h u)_e = \int_e u.t_e, to fluxes across the Voronoi edges,
#  (R~ *u)_e = \int_{e~} u.t_e.  Problems on the unit torus, meshes/meshN.nc (SCVT):
#      Hodge   : eps(u) = H R_h u - R~ *u for smooth periodic u
#      Poisson : d0' H d0 p = \int_{cell} f, p at the generators (= Delaunay vertices)
#      Darcy   : Hirani-Nakshatrala-Chaudhry (2015), eq. (4.6) with a body force:
#                    [ -mu H   D1' ] [ f ]   [ rho G ]     f_e = flux of v across e (normal -J t_e),
#                    [   D1     0  ] [ p ] = [  phi  ]     p_T = pressure at the circumcentre of T,
#                G_e = circulation of g along e~ (direction J t_e), phi_T = \int_T div v.
# ============================================================================
ENV["NS_MAIN"] = "0"                                # load the functions of nonsymmetric_check.jl only
include(joinpath(@__DIR__, "nonsymmetric_check.jl"))
using Serialization

const MESHES = ("mesh1", "mesh2", "mesh3", "mesh4", "mesh5", "mesh6")
const STARS = ("H_DEC", "H_P1", "H_P2")
const TSDIR = joinpath(@__DIR__, "..", "..", "output", "ts_data"); mkpath(TSDIR)

meshsize(mesh) = sqrt(mesh.x_period * mesh.y_period / mesh.cells.n)

# ---------------------------------------------------------------------------- stencils
"e and the triangles within L-1 steps of its two triangles (L = 1: 5 edges, L = 2: 13 = get_neighbors_2level_primal, L = 3: about 29)."
function tri_layers(mesh, e, L)
    S = Set{Int}(e)
    for _ in 1:L
        S = union(S, (get_neighbors_1level_primal(mesh, j) for j in S)...)
    end
    return sort(collect(S))
end
ring(mesh, e, r::Symbol) = tri_layers(mesh, e, parse(Int, String(r)[2:end]))       # :t2, :t3, ... next to ring(mesh, e, 2|3|4)
"""
The periodic Delaunay mesh of the given generators on [0,lx)×[0,ly), WITHOUT Lloyd steps.  (VoronoiMesh(points, lx, ly; max_iter=k)
always performs k+1 Lloyd steps; this calls the internal steps of VoronoiMeshes' DelaunayTriangulation extension directly.)
"""
function delaunay_mesh(xs::AbstractVector, ys::AbstractVector; lx = 1.0, ly = 1.0)
    DTE = Base.get_extension(VoronoiMeshes, :DelaunayTriangulationExt)
    pts = TensorsLite.VecArray(x = Float64.(mod.(xs, lx)), y = Float64.(mod.(ys, ly)))
    N = length(pts)
    p, _ = DTE.expand_periodic_points(pts, lx, ly, (1, 1))
    tri = DTE.triangulate(reinterpret(NTuple{2, Float64}, p))
    vor = DTE.voronoi(tri, clip = true)
    gens, verts, voc, cov = DTE.extract_periodic_vertices_and_cells(N, lx, ly, vor)
    M = maximum(length, voc)
    vocM = VoronoiMeshes.SmVecArray{M, Int32}(N); vocM .= voc
    pd = VoronoiMeshes.PlanarVoronoiDiagram(gens, verts, vocM, cov, ones(N), Float64(lx), Float64(ly))
    return VoronoiMesh(VoronoiMeshes.VoronoiDiagram(VoronoiMeshes.fix_diagram!(pd)))
end

const STENCILS = (:t2, 3, :t3, 4)
const SNAME = Dict(:t2 => "13-edge (2 triangle layers)", 3 => "3-level (vertex ring)", :t3 => "3 triangle layers", 4 => "4-ring")

"The three stars and the construction diagnostics (residual of the P1 constraints, largest P2 correction entry)."
function three_stars(mesh)
    Hd = hdiag(mesh)
    H1, info1 = symmetric_correction(mesh, 1, 3)
    H2, cm2 = rowwise_correction(mesh, Hd, 2, 3)
    return Dict("H_DEC" => Hd, "H_P1" => H1, "H_P2" => H2), (res_p1 = info1.res, cmax_p1 = info1.cmax, cmax_p2 = cm2)
end

"Λ = max_e Σ_j |H_ej| |e_j| / |e~_e|, the constant of the truncation-error bound."
function bigLambda(mesh, H)
    lp = edge_lengths(mesh, true); ld = edge_lengths(mesh, false); Ht = sparse(H')
    return maximum(sum(abs(nonzeros(Ht)[p]) * lp[rowvals(Ht)[p]] for p in nzrange(Ht, e)) / ld[e] for e in 1:size(H, 1))
end

# ---------------------------------------------------------------------------- Hodge error
const U2 = x -> [sin(TP * x[1]) * cos(TP * x[2]), -cos(TP * x[1]) * sin(TP * x[2])]
const HFIELDS = (d = (fields.d, supn.d), u2 = (U2, 1.0))

"(max, rms) relative Hodge error of the field u:  r_inf = max_e |eps_e|/(|e~| |u|inf),  r_2 = sqrt(Σ|e||e~|(eps_e/|e~|)^2 / Σ|e||e~|)/|u|inf."
function hodge_error(mesh, H, u, unorm)
    lp = edge_lengths(mesh, true); ld = edge_lengths(mesh, false)
    ε = H * reduce_primal_edges(mesh, u).values .- reduce_dual_edges(mesh, star_field(u)).values
    w = lp .* ld
    return (inf = maximum(abs.(ε) ./ ld) / unorm, l2 = sqrt(sum(w .* (ε ./ ld) .^ 2) / sum(w)) / unorm)
end

# ---------------------------------------------------------------------------- Poisson  -Δp = f
# p = sin2πx sin2πy + ½cos4πx (pfun, gradp, lapp from commuting_check.jl)
"Poisson solve with d0'Hd0 and its errors against p(x_c): nodal max and area-weighted L2 (relative), and the
discrete gradient error ||d0 e||_{H_DEC} relative (the norm of Theorem [error bounds])."
function poisson_errors(mesh, H)
    A = mesh.cells.area; d0 = d0_matrix_primal(mesh); Hd = hdiag(mesh)
    pe = [pfun(_pt(x)) for x in mesh.cells.position]
    ph = solve_poisson_primal(mesh, H, x -> -lapp(x))
    e = ph .- pe; e .-= sum(A .* e) / sum(A)                       # compare modulo constants (area-weighted)
    pc = pe .- sum(A .* pe) / sum(A)
    g = d0 * e; gp = d0 * pe
    return (sol = ph .- sum(A .* ph) / sum(A), exact = pc,
            inf = maximum(abs.(e)) / maximum(abs.(pc)), l2 = sqrt(sum(A .* e .^ 2) / sum(A .* pc .^ 2)),
            grad = sqrt(dot(g, Hd * g) / dot(gp, Hd * gp)))
end

# ---------------------------------------------------------------------------- Darcy (Hirani et al.)
"Triangles: unwrapped vertices (about the circumcentre), circumcentre, area, and D1 with every triangle CCW.
juliaDEC's d1_matrix_primal takes the vertex order of mesh.vertices.cells, which need not be CCW; here the sign of
each row is fixed from the geometry.  Returns (tris, D1, nflip)."
function triangles_ccw(mesh)
    nT = mesh.vertices.n
    I, J, V = Int[], Int[], Float64[]; tris = Vector{Any}(undef, nT); nflip = 0
    for t in 1:nT
        pts, eidx, eor = get_triangle(mesh, t)
        a, b, c = pts
        s = sign(_cross2(b - a, c - a)); s < 0 && (nflip += 1)
        for k in 1:3
            push!(I, t); push!(J, eidx[k]); push!(V, s * eor[k])
        end
        tris[t] = (pts = s > 0 ? (a, b, c) : (a, c, b), cc = _pt(mesh.vertices.position[t]), ccv = mesh.vertices.position[t],
                   area = _tri_area(a, b, c), edges = eidx)
    end
    return tris, sparse(I, J, V, nT, mesh.edges.n), nflip
end

"Data of a Darcy problem μ v + ∇p = ρ g, div v = φ on the torus, as cochains."
function darcy_data(mesh, tris; v, p, g, φ)
    f  = reduce_primal_edges(mesh, x -> _ROT90 * v(x)).values         # flux across e with normal -J t_e
    G  = reduce_dual_edges(mesh, g).values                              # circulation of g along e~ (direction J t_e)
    Φ  = [triangle_integral_ho(φ, T.pts...; qdeg = 5) for T in tris]
    pT = [p(T.cc) for T in tris]
    return (f = f, G = G, Φ = Φ, p = pT)
end

"Solve the Hirani system; one pressure is pinned and its continuity row (implied by the others) dropped."
function solve_darcy(H, D1, G, Φ, areas; μ = 1.0, ρ = 1.0)
    nT, nE = size(D1)
    K = [-μ .* sparse(H) sparse(D1'); D1 spzeros(nT, nT)]
    Φc = Φ .- areas .* (sum(Φ) / sum(areas))                         # Σ_T φ_T = 0 exactly (quadrature drift removed)
    rhs = vcat(ρ .* G, Φc)
    keep = setdiff(1:(nE + nT), nE + 1)
    x = zeros(nE + nT); x[keep] = K[keep, keep] \ rhs[keep]
    return x[1:nE], x[nE+1:end]
end

"Darcy errors: flux in the H_DEC norm (relative) and max; pressure at the circumcentres (max, L2) and Hirani's
measure (p_T constant on T against p(x), L2 over Ω, relative)."
function darcy_errors(mesh, tris, fh, ph, ex; pfun_ = nothing, vnorm = 1.0)
    lp = edge_lengths(mesh, true); hd = diag(hdiag(mesh)); At = [T.area for T in tris]
    ef = fh .- ex.f
    shift(q) = q .- sum(At .* q) / sum(At)
    ep = shift(ph .- ex.p); pc = shift(ex.p)
    out = (flux = sqrt(sum(hd .* ef .^ 2) / sum(hd .* ex.f .^ 2)), flux_inf = maximum(abs.(ef) ./ lp) / vnorm,
           p_inf = maximum(abs.(ep)) / maximum(abs.(pc)), p_l2 = sqrt(sum(At .* ep .^ 2) / sum(At .* pc .^ 2)))
    if pfun_ !== nothing                                                # Hirani's pressure measure
        phs = ph .- sum(At .* ph) / sum(At)                            # ∫ p = 0 for the exact solutions used here
        num = sum(triangle_integral_ho(x -> (phs[t] - pfun_(x))^2, tris[t].pts...; qdeg = 5) for t in eachindex(tris))
        den = sum(triangle_integral_ho(x -> pfun_(x)^2, tris[t].pts...; qdeg = 5) for t in eachindex(tris))
        out = merge(out, (p_hirani = sqrt(num / den),))
    end
    return out
end

"Velocity at each triangle from its three fluxes: f_e = (x_head - x_tail) . w with w = J v constant (least squares)."
function triangle_velocity(mesh, tris, f)
    V = zeros(length(tris), 2)
    for (t, T) in enumerate(tris)
        M = zeros(3, 2); b = zeros(3)
        for (k, e) in enumerate(T.edges)
            q1, q2 = edge_endpoints(mesh, e, true; about = T.ccv)
            M[k, :] = q2 .- q1; b[k] = f[e]
        end
        w = M \ b
        V[t, :] = -(_ROT90 * w)                                         # v = -J w
    end
    return V
end

# the Darcy problems
PD(x) = cos(TP * x[1]) * cos(TP * x[2])                                     # Hirani 6.2 on the torus (cos πx cos πy -> period 1)
VD(x) = [TP * sin(TP * x[1]) * cos(TP * x[2]), TP * cos(TP * x[1]) * sin(TP * x[2])]           # v = -∇p
PHID(x) = 2 * TP^2 * cos(TP * x[1]) * cos(TP * x[2])                       # div v = -Δp
darcy_smooth(mesh, tris) = darcy_data(mesh, tris; v = VD, p = PD, g = x -> [0.0, 0.0], φ = PHID)
darcy_patch(mesh, tris)  = darcy_data(mesh, tris; v = x -> [1.0, 0.0], p = x -> 0.0, g = x -> [1.0, 0.0], φ = x -> 0.0)

# ---------------------------------------------------------------------------- spectra
"Eigenvalues of the star relative to DEC (spectrum of H_DEC^-1 H) and of the wave/diffusion operator A^-1 d0'Hd0."
function star_spectra(mesh, H)
    Dm = Diagonal(1 ./ sqrt.(diag(hdiag(mesh))))
    d0 = d0_matrix_primal(mesh)
    return (H = eigvals(Matrix(Dm * H * Dm)), L = eigvals(Diagonal(1 ./ mesh.cells.area) * Matrix(d0' * H * d0)))
end
