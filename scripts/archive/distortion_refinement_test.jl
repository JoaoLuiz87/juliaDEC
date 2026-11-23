# ============================================================================
#  distortion_refinement_test.jl  -- numerical test of the circumradius-jump
#  explanation of the TRiSK behaviour reported by Peixoto et al. (2026)
#  (Sections "Where the offset comes from: the circumradii" and "Their
#  finite-volume results in our language" of hodge_symmetry_consistency.tex)
#
#      julia scripts/archive/-t 4 distortion_refinement_test.jl
#      julia -t 4 distortion_refinement_test.jl meshes-distort   (only build the distorted family)
#      julia -t 4 distortion_refinement_test.jl meshes-refine    (only build the refined family)
#
#  Meshes (written to juliaDEC/meshes/ and built only if missing):
#    dist_base     Lloyd-converged CVT with 11600 generators (2900 random generators
#                  + Lloyd, refined as in mesh_generator.jl: cells + edge midpoints, Lloyd again)
#    dist_sXX      dist_base with each generator coordinate displaced by
#                  clamp(N(0, (σ dx)²), -0.3dx, 0.3dx), σ = XX/100, then 10 Lloyd iterations
#    dist4k_sXX    the same on the 4096-cell SCVT mesh5 (smaller analogue)
#    ref1..ref3    locally refined CVT: density 16 for r < 0.15, 1 for r > 0.35, linear in
#                  between (r = distance to the centre); CVT spacing h ~ density^(-1/4), so the
#                  outside has half the resolution of the disc.  ~1600, 6400, 25600 cells.
#
#  NOTE on VoronoiMeshes.jl: VoronoiMesh(points, ...; max_iter = k) performs k + 1 Lloyd
#  steps (one before its loop), so "10 Lloyd iterations" is max_iter = 9.  Its default
#  max_time = 4 minutes also stops long runs; it is raised here.
# ============================================================================
using VoronoiMeshes, DelaunayTriangulation, TensorsLite, TensorsLiteGeometry, SmallCollections, LinearAlgebra, NCDatasets
using StaticArrays, SparseArrays, Statistics, Random, Printf
const ROOT = joinpath(@__DIR__, "..", "..")
cd(ROOT)                                                # NetCDF cannot open the absolute (non-ASCII) path
meshfile(name) = "meshes/$name.nc"

const SIGMAS = 0:5:30                                   # σ in hundredths of dx
const R_IN, R_OUT = 0.15, 0.35                          # refinement disc and outer radius
const DENS_IN = 16.0                                    # 16^(1/4) = 2: half resolution outside
refdensity = VoronoiMeshes.circular_refinement_function(1.0, 1.0; inner_radius = R_IN, buffer_radius = R_OUT,
                                                         inner_density = DENS_IN, outer_density = 1.0)
meanh(m) = sum(norm(closest(m.cells.position[a], m.cells.position[b], 1.0, 1.0) - m.cells.position[a])
               for (a, b) in m.edges.cells) / m.edges.n
# refinement as in mesh_generator.jl: new generators = old generators + old edge midpoints
refine(m) = vcat(m.cells.position, m.edges.position)
function save_mesh(name, m)
    VoronoiMeshes.save(meshfile(name), m)
    @printf("  wrote %-12s %6d cells  h = %.5f\n", name, m.cells.n, meanh(m)); flush(stdout)
end

# ---- the distorted family at fixed resolution
function distort_family(prefix, base; seed = 2026)
    dx = meanh(base); N = base.cells.n
    z = randn(Xoshiro(seed), N, 2)                       # one draw, rescaled for every σ
    for s in SIGMAS
        name = @sprintf("%s_s%02d", prefix, s); isfile(meshfile(name)) && continue
        σ = s / 100; pts = similar(base.cells.position)
        for i in 1:N
            p = base.cells.position[i]
            ddx = clamp(σ * dx * z[i, 1], -0.3dx, 0.3dx); ddy = clamp(σ * dx * z[i, 2], -0.3dx, 0.3dx)
            pts[i] = typeof(p)(mod(p.x + ddx, 1.0), mod(p.y + ddy, 1.0))
        end
        save_mesh(name, VoronoiMesh(pts, 1.0, 1.0; max_iter = 9, max_time = 10.0))
    end
end

function build_distorted()
    if !isfile(meshfile("dist_base"))
        Random.seed!(2026)
        c = VoronoiMesh(2900, 1.0, 1.0; max_iter = 200_000, rtol = 1e-6, max_time = 8.0)
        @printf("  coarse 2900-cell CVT: h = %.5f\n", meanh(c))
        save_mesh("dist_base", VoronoiMesh(refine(c), 1.0, 1.0; max_iter = 200_000, rtol = 1e-6, max_time = 25.0))
    end
    distort_family("dist", VoronoiMesh(meshfile("dist_base")))
    distort_family("dist4k", VoronoiMesh(meshfile("mesh5")))
end

function build_refined()
    Random.seed!(7)
    isfile(meshfile("ref1")) ||
        save_mesh("ref1", VoronoiMesh(1600, 1.0, 1.0; density = refdensity, max_iter = 200_000, rtol = 1e-6, max_time = 8.0))
    isfile(meshfile("ref2")) ||
        save_mesh("ref2", VoronoiMesh(refine(VoronoiMesh(meshfile("ref1"))), 1.0, 1.0; density = refdensity,
                                      max_iter = 200_000, rtol = 1e-6, max_time = 12.0))
    isfile(meshfile("ref3")) ||
        save_mesh("ref3", VoronoiMesh(refine(VoronoiMesh(meshfile("ref2"))), 1.0, 1.0; density = refdensity,
                                      max_iter = 200_000, rtol = 1e-6, max_time = 20.0))
end

"meshes-distort" in ARGS && (build_distorted(); exit())
"meshes-refine"  in ARGS && (build_refined(); exit())
build_distorted(); build_refined()

# ============================================================================
# analysis
# ============================================================================
using QuadGK, CairoMakie
Base.similar(bc::Base.Broadcast.Broadcasted, ::Type{Union{}}) = Array{Union{}}(undef, length.(axes(bc)))
include(joinpath(ROOT, "src", "juliaDEC.jl")); include(joinpath(ROOT, "src", "hodge_stars.jl"))
const OUT = joinpath(ROOT, "report")

const INK = colorant"#2b2b29"; const MUTED = colorant"#8a8a85"; const GRID = colorant"#e6e5e1"
const BLUE = colorant"#2a78d6"; const ORANGE = colorant"#eb6834"; const AQUA = colorant"#1baf7a"
const RED = colorant"#c23b3a"; const VIOLET = colorant"#7d55c7"
const DIV = cgrad([colorant"#1c5cab", colorant"#f0efec", colorant"#c23b3a"])
set_theme!(Theme(fontsize = 13, textcolor = INK,
    Axis = (xgridcolor = GRID, ygridcolor = GRID, spinewidth = 0.8, leftspinecolor = MUTED,
            bottomspinecolor = MUTED, topspinevisible = false, rightspinevisible = false,
            xtickcolor = MUTED, ytickcolor = MUTED, titlefont = :regular, titlesize = 13),
    Legend = (framevisible = false, labelsize = 11)))
savefig(name, f) = (CairoMakie.save(joinpath(OUT, name), f, px_per_unit = 2); println("  wrote ", name))

# ---- geometry, as in circumradius_figures.jl
function circumradii(m)
    R = zeros(m.vertices.n)
    for T in 1:m.vertices.n
        v = m.vertices.position[T]
        R[T] = norm(_pt(closest(v, m.cells.position[m.vertices.cells[T][1]], 1.0, 1.0)) - _pt(v))
    end
    return R
end
# offset delta_e (n_e = R t_e), dual length |e~|, primal length |e|, and the two triangles of e:
# T2 = the one whose circumcentre lies further along n_e (lem:circum: |e~| delta_e = (R_T2^2 - R_T1^2)/2).
# Read from the geometry rather than from D1, whose incidence is unreliable on degenerate edges.
function offsets(m)
    d = zeros(m.edges.n); L = zeros(m.edges.n); le = zeros(m.edges.n); T1 = zeros(Int, m.edges.n); T2 = zeros(Int, m.edges.n)
    for e in 1:m.edges.n
        c1, c2 = edge_endpoints(m, e, true); v1, v2 = edge_endpoints(m, e, false)
        t = (c2 - c1) / norm(c2 - c1); n = [-t[2], t[1]]; me = (c1 + c2) / 2
        d[e] = dot((v1 + v2) / 2 - me, n); L[e] = norm(v2 - v1); le[e] = norm(c2 - c1)
        a, b = m.edges.vertices[e]
        T1[e], T2[e] = dot(v1 - me, n) <= dot(v2 - me, n) ? (a, b) : (b, a)
    end
    return d, L, le, T1, T2
end
defects(m) = ([_pt(m.cells.position[c]) for c in 1:m.cells.n if length(m.cells.edges[c]) == 5],
              [_pt(m.cells.position[c]) for c in 1:m.cells.n if length(m.cells.edges[c]) == 7])
nonhex(m) = count(c -> length(m.cells.edges[c]) != 6, 1:m.cells.n)
# largest angle of every Delaunay triangle; the triangle is well-centred iff it is < 90 degrees
function max_angles(m)
    A = zeros(m.vertices.n)
    for T in 1:m.vertices.n
        v = m.vertices.position[T]
        P = [_pt(closest(v, m.cells.position[c], 1.0, 1.0)) for c in m.vertices.cells[T]]
        A[T] = maximum(acosd(clamp(dot(P[mod1(i+1, 3)] - P[i], P[mod1(i+2, 3)] - P[i]) /
                                   (norm(P[mod1(i+1, 3)] - P[i]) * norm(P[mod1(i+2, 3)] - P[i])), -1, 1)) for i in 1:3)
    end
    return A
end

# ---- the diagonal-star Laplacian, as in peixoto_figures.jl
pf(x)   = sin(2pi*x[1]) * sin(2pi*x[2]) + 0.3cos(2pi*(x[1] - 2x[2]))
lapf(x) = -8pi^2*sin(2pi*x[1])*sin(2pi*x[2]) - 6pi^2*cos(2pi*(x[1] - 2x[2]))
diag_star(m) = spdiagm(0 => edge_lengths(m, false) ./ edge_lengths(m, true))
function truncation(m, H)
    d0  = sparse(Float64.(d0_matrix_primal(m)))
    p   = [pf(_pt(m.cells.position[c])) for c in 1:m.cells.n]
    avg = discretize_scalar_field_cells(m, lapf).values ./ m.cells.area
    return ((d0' * (H * (d0 * p))) ./ m.cells.area .+ avg) ./ maximum(abs, avg)
end
# error of the Poisson solution with the same star, relative to max|p|
function poisson_error(m, H)
    ph = solve_poisson_primal(m, H, x -> -lapf(x))
    pe = [pf(_pt(m.cells.position[c])) for c in 1:m.cells.n]; pe .-= sum(pe) / length(pe)
    return ph .- pe, maximum(abs, pe)
end
gpf(x)  = [2pi*cos(2pi*x[1])*sin(2pi*x[2]) - 0.6pi*sin(2pi*(x[1]-2x[2])),
           2pi*sin(2pi*x[1])*cos(2pi*x[2]) + 1.2pi*sin(2pi*(x[1]-2x[2]))]
hesspf(x) = (s = sin(2pi*x[1])*sin(2pi*x[2]); c = cos(2pi*(x[1]-2x[2]));
             4pi^2 .* [-s-0.3c  cos(2pi*x[1])*cos(2pi*x[2])+0.6c; cos(2pi*x[1])*cos(2pi*x[2])+0.6c  -s-1.2c])
# flux error of the diagonal star on grad p (prop:p1): per edge relative to |e~| max|grad p|, and
# in the W^{-1} norm that bounds the Poisson error (prop:p4), relative to the exact flux
function flux_error(m, L, le)
    S = reduce_dual_edges(m, star_field(gpf)).values
    ε = (L ./ le) .* reduce_primal_edges(m, gpf).values .- S
    gmax = maximum(norm(gpf(_pt(m.cells.position[c]))) for c in 1:m.cells.n)
    return abs.(ε) ./ (L .* gmax), sqrt(sum(le ./ L .* ε .^ 2) / sum(le ./ L .* S .^ 2))
end
# the leading term of tau (prop:p2), tau_c = (1/|V_c|) sum_e sigma_{e,c} |e~| delta_e (t_e . Hess p(x_c) n_e),
# for any offset field delta, relative to max|Lap p|; used to split tau into smooth and rough parts
function tau_model(m, δ, L)
    out = zeros(m.cells.n); lmax = maximum(abs(lapf(_pt(m.cells.position[c]))) for c in 1:m.cells.n)
    for c in 1:m.cells.n
        M = hesspf(_pt(m.cells.position[c]))
        for e in Int.(m.cells.edges[c])
            c1, c2 = edge_endpoints(m, e, true); t = (c2 - c1) / norm(c2 - c1); n = [-t[2], t[1]]
            σ = m.edges.cells[e][1] == c ? 1 : -1
            out[c] += σ * L[e] * δ[e] * dot(t, M * n)
        end
        out[c] /= m.cells.area[c] * lmax
    end
    return out
end
# local TRiSK kinetic energy error for a constant field (prop:p3), as in peixoto_figures.jl
function ke_error(m, a)
    err = zeros(m.cells.n)
    for c in 1:m.cells.n
        K = 0.0
        for e in Int.(m.cells.edges[c])
            c1, c2 = edge_endpoints(m, e, true); v1, v2 = edge_endpoints(m, e, false)
            K += norm(c2 - c1) * norm(v2 - v1) * dot(a, (c2 - c1) / norm(c2 - c1))^2
        end
        err[c] = (K / (4m.cells.area[c]) - dot(a, a)/2) / (dot(a, a)/2)
    end
    return err
end

# ---- the skew pairing, as in obstruction_figures.jl
const U = x -> [sin(2pi*x[2]), cos(2pi*x[1])]
const V = x -> [cos(2pi*(x[1] + x[2])), sin(2pi*(x[1] - 2x[2]))]
function s_rel(m, u, v)
    Ru, Rv = reduce_primal_edges(m, u).values, reduce_primal_edges(m, v).values
    Su, Sv = reduce_dual_edges(m, star_field(u)).values, reduce_dual_edges(m, star_field(v)).values
    return abs(dot(Rv, Su) - dot(Ru, Sv)) / sqrt(abs(dot(Ru, Su) * dot(Rv, Sv)))
end

# how far from a Lloyd fixed point: generator movement in one more Lloyd step, / sqrt|V_c|
function lloyd_residual(m; density = (x -> 1.0))
    m1 = VoronoiMesh(copy(m.cells.position), 1.0, 1.0; density = density, max_iter = 0)
    mv = [norm(closest(m.cells.position[i], m1.cells.position[i], 1.0, 1.0) - m.cells.position[i]) / sqrt(m.cells.area[i])
          for i in 1:m.cells.n]
    return maximum(mv), sqrt(mean(mv .^ 2))
end

wrms(x, w) = sqrt(sum(w .* x .^ 2) / sum(w))
radius(p) = hypot(mod(p[1], 1.0) - 0.5, mod(p[2], 1.0) - 0.5)

# every number used below, for one mesh
function analyse(mn)
    m = VoronoiMesh(meshfile(mn)); h = meanh(m)
    R = circumradii(m); d, L, le, T1, T2 = offsets(m)
    rho = R .^ 2 ./ 2
    J  = abs.(rho[T2] .- rho[T1])                           # |R2^2 - R1^2| / 2 per edge
    idv = maximum(abs, rho[T2] .- rho[T1] .- L .* d) / maximum(abs, L .* d)
    D1 = sparse(Float64.(d1_matrix_primal(m)))
    idD1 = count(abs.(D1' * rho .- L .* d) .> 1e-8 * maximum(abs, L .* d))   # edges where the code's D1 disagrees
    r  = abs.(d) ./ L
    tau = truncation(m, diag_star(m)); perr, pmax = poisson_error(m, diag_star(m))
    eps, epsW = flux_error(m, L, le); ke = ke_error(m, [0.8, -0.35])
    ang = max_angles(m); p5, p7 = defects(m)
    res = (name = mn, m = m, N = m.cells.n, h = h, d = d, L = L, le = le, R = R, T1 = T1, T2 = T2, tau = tau, perr = perr ./ pmax,
           eps = eps, ke = ke, J = 2J, Jmax = 2maximum(J) / h^3, Jrms = 2sqrt(mean(J .^ 2)) / h^3,
           rmax = maximum(r), rrms = wrms(r, L), tmax = maximum(abs, tau), trms = sqrt(mean(tau .^ 2)),
           emax = maximum(eps), eW = epsW, kmax = maximum(abs, ke),
           pmaxerr = maximum(abs, perr) / pmax, prms = sqrt(mean(perr .^ 2)) / pmax, s = s_rel(m, U, V),
           nobt = count(>(90), ang), amax = maximum(ang), nbad = count(e -> !diamond_is_well_centred(m, e), 1:m.edges.n),
           n5 = length(p5), n7 = length(p7), nother = nonhex(m) - length(p5) - length(p7), idv = idv, idD1 = idD1)
    @printf("  %-11s N=%5d h=%.5f | |R2²-R1²|/h³ max %6.2f rms %5.3f | |δ|/|ẽ| max %.3f rms %.4f | flux err max %.1e W⁻¹ %.1e | τ max %.2e rms %.2e | KE max %.1e | Poisson max %.1e rms %.1e | s_h %.1e | 5/7/other %3d/%3d/%d | obtuse %d (max angle %.1f°) | identity %.0e (D1 off on %d edges)\n",
            mn, res.N, h, res.Jmax, res.Jrms, res.rmax, res.rrms, res.emax, res.eW, res.tmax, res.trms, res.kmax, res.pmaxerr, res.prms, res.s,
            res.n5, res.n7, res.nother, res.nobt, res.amax, idv, idD1)
    flush(stdout)
    return res
end

# smooth part of the potential rho = R^2/2 on the refined meshes: its radial profile (bin means
# over triangles by circumcentre radius, linearly interpolated).  The smooth offset is
# (rho_bar(T2) - rho_bar(T1)) / |e~| (the grading term of the note), the rough one is the remainder.
function smooth_split(A)
    m = A.m; rT = [radius(m.vertices.position[T]) for T in 1:m.vertices.n]
    rho = A.R .^ 2 ./ 2; dr = 1.5sqrt(2 / (sqrt(3) * m.cells.n))      # ~1.5 mean spacings
    edges_r = 0:dr:(maximum(rT) + dr); nb = length(edges_r) - 1
    bm = [(s = rho[(rT .>= edges_r[k]) .& (rT .< edges_r[k+1])]; isempty(s) ? NaN : mean(s)) for k in 1:nb]
    bc = [(edges_r[k] + edges_r[k+1]) / 2 for k in 1:nb]
    ok = .!isnan.(bm); bc, bm = bc[ok], bm[ok]
    interp(x) = x <= bc[1] ? bm[1] : x >= bc[end] ? bm[end] :
                (k = searchsortedlast(bc, x); bm[k] + (bm[k+1] - bm[k]) * (x - bc[k]) / (bc[k+1] - bc[k]))
    rhobar = interp.(rT)
    ds = (rhobar[A.T2] .- rhobar[A.T1]) ./ A.L
    return ds, A.d .- ds
end
region(r) = r < R_IN ? 1 : r <= R_OUT ? 2 : 3
const REGNAMES = ("disc r<$(R_IN)", "band", "outside r>$(R_OUT)")

# ============================================================================
# (1) the distorted families
# ============================================================================
println("Lloyd residual (max, rms of the next Lloyd step / sqrt|V_c|):")
for mn in ("dist_base", "mesh5")
    a, b = lloyd_residual(VoronoiMesh(meshfile(mn))); @printf("  %-9s %.1e  %.1e\n", mn, a, b)
end
println("distorted family, 11600 cells (h = mean Delaunay edge length; δ rms weighted by |ẽ|)")
DIST = [analyse(@sprintf("dist_s%02d", s)) for s in SIGMAS]
println("distorted family, 4096 cells")
DIST4 = [analyse(@sprintf("dist4k_s%02d", s)) for s in SIGMAS]
println("growth factor sigma = 0.30 / sigma = 0:")
for (lab, F) in (("11600", DIST), ("4096", DIST4))
    g(k) = getfield(F[end], k) / getfield(F[1], k)
    @printf("  %-6s |R2²-R1²| max %.2f rms %.2f | |δ|/|ẽ| max %.2f rms %.2f | flux err max %.2f W⁻¹ %.2f | τ max %.2f rms %.2f | KE max %.2f | Poisson max %.2f rms %.2f | s_h %.2f\n",
            lab, g(:Jmax), g(:Jrms), g(:rmax), g(:rrms), g(:emax), g(:eW), g(:tmax), g(:trms), g(:kmax), g(:pmaxerr), g(:prms), g(:s))
end
# where do the large offsets sit at the largest distortion? (share of edges above a threshold, and of
# the largest offsets that touch a non-hexagonal cell)
for (lab, F) in (("11600", DIST), ("4096", DIST4))
    for A in (F[1], F[end])
        m = A.m; r = abs.(A.d) ./ A.L
        nearDefect = [any(c -> length(m.cells.edges[c]) != 6, m.edges.cells[e]) for e in 1:m.edges.n]
        top = sortperm(r, rev = true)[1:ceil(Int, 0.01 * m.edges.n)]
        @printf("  %-6s %-11s edges with |δ|/|ẽ| > 0.1: %5.2f%%   top 1%% of offsets touching a non-hexagon: %5.1f%% (non-hexagon edges: %4.1f%% of all)\n",
                lab, A.name, 100count(>(0.1), r) / length(r), 100count(nearDefect[top]) / length(top), 100count(nearDefect) / length(r))
    end
end

# ============================================================================
# (2) the locally refined family: where do the offsets sit?
# ============================================================================
println("refined family (offset statistics per region; h_e = |e| locally)")
REF = [analyse(mn) for mn in ("ref1", "ref2", "ref3")]
for A in REF
    m = A.m; a, b = lloyd_residual(m; density = refdensity)
    re = [radius(m.edges.position[e]) for e in 1:m.edges.n]; rc = [radius(m.cells.position[c]) for c in 1:m.cells.n]
    ds, dr = smooth_split(A); rg = region.(re); rgc = region.(rc)
    ts, tr, tt = tau_model(m, ds, A.L), tau_model(m, dr, A.L), tau_model(m, A.d, A.L)
    hin = mean(A.le[re .< R_IN - 0.03]); hout = mean(A.le[re .> R_OUT + 0.03])
    @printf("  %s: N=%d  h_in %.4f  h_out %.4f (ratio %.2f)  Lloyd residual %.1e/%.1e  leading-term model of τ: max|τ - model| / max|τ| = %.2f\n",
            A.name, A.N, hin, hout, hout / hin, a, b, maximum(abs, A.tau .- tt) / maximum(abs, A.tau))
    for k in 1:3
        E = rg .== k; C = rgc .== k
        nd = count(c -> length(m.cells.edges[c]) != 6, findall(C))
        no = count(abs.(A.d[E]) .> A.L[E] ./ 2)               # e does not cross its dual edge (lem:bisect fails)
        @printf("    %-16s cells %5d non-hex %5.1f%% | |δ|/|ẽ| max %.3f rms %.4f [smooth rms %.4f max %.3f | rough rms %.4f] | |R2²-R1²|/|e|³ max %5.2f rms %.3f | flux err max %.1e rms %.1e | τ max %.1e rms %.1e [smooth part max %.1e, rough part max %.1e] | KE max %.1e rms %.1e | Poisson err max %.1e | non-wc edges %d\n",
                REGNAMES[k], count(C), 100nd / count(C), maximum(abs.(A.d[E]) ./ A.L[E]), wrms(abs.(A.d[E]) ./ A.L[E], A.L[E]),
                wrms(ds[E] ./ A.L[E], A.L[E]), maximum(abs.(ds[E]) ./ A.L[E]), wrms(dr[E] ./ A.L[E], A.L[E]),
                maximum(A.J[E] ./ A.le[E] .^ 3), sqrt(mean((A.J[E] ./ A.le[E] .^ 3) .^ 2)),
                maximum(A.eps[E]), sqrt(mean(A.eps[E] .^ 2)),
                maximum(abs, A.tau[C]), sqrt(mean(A.tau[C] .^ 2)), maximum(abs, ts[C]), maximum(abs, tr[C]),
                maximum(abs, A.ke[C]), sqrt(mean(A.ke[C] .^ 2)), maximum(abs, A.perr[C]), no)
    end
end

# ============================================================================
# (3) the figure
# ============================================================================
# per radial bin: weighted rms and max of x
function radial_profile(x, r, w; dr = 0.025)
    edges_r = 0:dr:0.725; bc = Float64[]; rm = Float64[]; mx = Float64[]
    for k in 1:length(edges_r)-1
        s = findall((r .>= edges_r[k]) .& (r .< edges_r[k+1])); length(s) < 5 && continue
        push!(bc, (edges_r[k] + edges_r[k+1]) / 2); push!(rm, wrms(x[s], w[s])); push!(mx, maximum(abs, x[s]))
    end
    return bc, rm, mx
end
let f = Figure(size = (1350, 900))
    σs = collect(SIGMAS) ./ 100
    mk!(ax, x, y, c, m) = scatter!(ax, x, y, color = c, marker = m, markersize = 9, strokecolor = :white, strokewidth = 1)
    # (a), (b): offsets and circumradius jumps against σ, both resolutions
    for (j, (kmax, krms, yl, ttl)) in enumerate([(:rmax, :rrms, "|δₑ| / |ẽ|", "(a) midpoint offset  (numbers: obtuse triangles, 11600 cells)"),
                                                 (:Jmax, :Jrms, "|R₂² − R₁²| / h³", "(b) circumradius jump")])
        ax = Axis(f[1, j], xlabel = "σ  (displacement std / dx)", ylabel = yl, yscale = log10, title = ttl)
        for (F, ls, lab) in ((DIST, :solid, "11600 cells"), (DIST4, :dash, "4096 cells"))
            ymax = [getfield(A, kmax) for A in F]; yrms = [getfield(A, krms) for A in F]
            lines!(ax, σs, ymax, color = ORANGE, linestyle = ls, linewidth = 2.5, label = "max, $lab")
            lines!(ax, σs, yrms, color = BLUE, linestyle = ls, linewidth = 2.5, label = "rms, $lab")
            ls == :solid && (mk!(ax, σs, ymax, ORANGE, :utriangle); mk!(ax, σs, yrms, BLUE, :circle))
            if j == 1 && ls == :solid
                text!(ax, σs, ymax, text = string.([A.nobt for A in F]), align = (:center, :bottom), offset = (0, 7), fontsize = 10, color = INK)
            end
        end
        j == 1 && (axislegend(ax, position = :lt, labelsize = 10); ylims!(ax, 6e-3, 400))
    end
    # (c): growth relative to σ = 0, 11600 cells
    ax = Axis(f[1, 3], xlabel = "σ  (displacement std / dx)", ylabel = "value / value at σ = 0", yscale = log10,
              title = "(c) growth with distortion (11600 cells)")
    for (k, lab, c, m, ls) in [(:Jrms, "rms |R₂² − R₁²|", BLUE, :circle, :solid), (:tmax, "max |τ_c| (Laplacian truncation)", RED, :rect, :solid),
                               (:kmax, "max local kinetic-energy error", ORANGE, :utriangle, :solid),
                               (:eW, "flux error, W⁻¹ norm", AQUA, :diamond, :solid), (:prms, "Poisson solution error (rms)", INK, :circle, :dash),
                               (:s, "|s_h(u, v)|", VIOLET, :xcross, :solid)]
        y = [getfield(A, k) for A in DIST] ./ getfield(DIST[1], k)
        lines!(ax, σs, y, color = c, linewidth = 2.5, linestyle = ls, label = lab); mk!(ax, σs, y, c, m)
    end
    hlines!(ax, [1.0], color = MUTED, linewidth = 1)
    axislegend(ax, position = :lt, labelsize = 10)

    # (d): the offsets on the refined mesh ref2 (colour scale clipped: obtuse triangles give |δ|/|ẽ| up to ~2)
    A = REF[2]; m = A.m; dl = A.d ./ A.L; lim = 0.2
    ax = Axis(f[2, 1], aspect = DataAspect(), title = @sprintf("(d) δₑ/|ẽ| on the refined mesh (%d cells)", A.N))
    hidedecorations!(ax); hidespines!(ax)
    segs = Point2f[]; cols = Float64[]
    for e in sortperm(abs.(dl))                             # large offsets drawn last, on top
        c1, c2 = edge_endpoints(m, e, true); push!(segs, Point2f(c1...), Point2f(c2...)); push!(cols, dl[e], dl[e])
    end
    linesegments!(ax, segs, color = clamp.(cols, -lim, lim), colormap = DIV, colorrange = (-lim, lim), linewidth = 1.1)
    for rr in (R_IN, R_OUT)
        lines!(ax, [Point2f(0.5 + rr * cos(t), 0.5 + rr * sin(t)) for t in range(0, 2pi, length = 200)], color = INK, linestyle = :dash, linewidth = 1)
    end
    p5, p7 = defects(m)
    isempty(p5) || scatter!(ax, first.(p5), last.(p5), marker = :diamond, color = INK, markersize = 5, strokecolor = :white, strokewidth = 0.4)
    isempty(p7) || scatter!(ax, first.(p7), last.(p7), marker = :star5, color = INK, markersize = 6, strokecolor = :white, strokewidth = 0.4)
    xlims!(ax, 0, 1); ylims!(ax, 0, 1)
    Colorbar(f[2, 1][1, 2], colormap = DIV, limits = (-lim, lim), width = 8, ticklabelsize = 10)

    # (e), (f): radial profiles of the offset (total and smooth part) and of the truncation error
    axe = Axis(f[2, 2], xlabel = "distance r from the centre of the disc", ylabel = "rms of |δₑ|/|ẽ| per ring",
               title = "(e) offsets by ring: total (solid), smooth part (dashed)")
    axf = Axis(f[2, 3], xlabel = "distance r from the centre of the disc", ylabel = "max |τ_c| per ring", yscale = log10,
               title = "(f) Laplacian truncation by ring: total (solid),\nleading term of the smooth part (dashed)")
    for ax in (axe, axf); vspan!(ax, [R_IN], [R_OUT], color = (MUTED, 0.12)); end
    for (A, c, m_) in zip(REF, (AQUA, BLUE, VIOLET), (:circle, :rect, :utriangle))
        m = A.m; ds, _ = smooth_split(A)
        re = [radius(m.edges.position[e]) for e in 1:m.edges.n]; rc = [radius(m.cells.position[c]) for c in 1:m.cells.n]
        bc, rm, _ = radial_profile(A.d ./ A.L, re, A.L); lines!(axe, bc, rm, color = c, linewidth = 2.2, label = "$(A.N) cells")
        mk!(axe, bc, rm, c, m_)
        bc, rm, _ = radial_profile(ds ./ A.L, re, A.L); lines!(axe, bc, rm, color = c, linewidth = 1.8, linestyle = :dash)
        bc, _, mx = radial_profile(A.tau, rc, ones(length(rc))); lines!(axf, bc, mx, color = c, linewidth = 2.2, label = "$(A.N) cells")
        mk!(axf, bc, mx, c, m_)
        bc, _, mx = radial_profile(tau_model(m, ds, A.L), rc, ones(length(rc))); lines!(axf, bc, mx, color = c, linewidth = 1.8, linestyle = :dash)
    end
    axislegend(axe, position = :rt, labelsize = 10)
    Label(f[3, :], "Top: Lloyd-converged CVT (11600 cells; dashed: the 4096-cell SCVT mesh5), every generator coordinate displaced by clamp(N(0, (σ dx)²), ±0.3dx), " *
          "then 10 Lloyd iterations; h = dx = mean Delaunay edge length; rms of |δ|/|ẽ| weighted by |ẽ|.\n" *
          "Bottom: CVT density 16 for r < $(R_IN), 1 for r > $(R_OUT), linear in between (h_out ≈ 1.95 h_in); dashed circles and grey bands: transition band; " *
          "(d) colour scale clipped at ±$(lim).  ◆ pentagons, ★ heptagons.\nSmooth part of δ: jump of the ring average of R²/2 divided by |ẽ| " *
          "(the grading term κ²h∂ₙh).  τ: (1/|V_c|)(d₀ᵀH₀d₀p)_c + mean of Δp over V_c, relative to max|Δp|; colours in (f) as in (e).\n" *
          "Dashed in (f): τ_c = Σₑ σ|ẽ|δₑ(tₑ·∇²p nₑ)/|V_c| (exact for quadratic p) with δ replaced by its smooth part.",
          fontsize = 11, color = MUTED, halign = :left, justification = :left, tellwidth = false)
    savefig("fig_distortion_refinement.png", f)
end
println("done")
