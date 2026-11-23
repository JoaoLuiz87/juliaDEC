# ============================================================================
#  commuting_check.jl -- the commuting identity  H R_h u = R~_h *u  on the SCVT meshes.
#
#      julia -t 4 scripts/stars/commuting_check.jl      (writes output/gs_data/commuting_check.txt)
#
#  Part 1  the LIFT and SANDWICH stars of hodge_stars.jl (and DEC for reference):
#          symmetry, definiteness, spectrum of d0' H d0, commuting defect on constants and
#          smooth fields, and the truncation error of the Laplacian in the max norm.
#  Part 2  basis invariance: a random change of basis of [P1]^2 on one stencil leaves both
#          stars unchanged.
#  Part 3  does ANY symmetric H with a k-ring sparsity pattern satisfy the commuting identity
#          exactly on [P_k]^2, row by row?  We look for the smallest symmetric correction of DEC,
#              H = Hd + C,  C = C',   (H A^e)_m = b^e_m   for every edge e and basis field m,
#          with A^e_jm = int_{e_j} f_m . t,  b^e_m = int_{*e} (*f_m) . t  in the frame of e.
#          Solvable  <=>  least-squares residual ~ 0.
# ============================================================================
using VoronoiMeshes, TensorsLite, TensorsLiteGeometry, SmallCollections, LinearAlgebra, NCDatasets
using StaticArrays, SparseArrays, Printf, Random
Base.similar(bc::Base.Broadcast.Broadcasted, ::Type{Union{}}) = Array{Union{}}(undef, length.(axes(bc)))
const ROOT = joinpath(@__DIR__, "..", ".."); cd(ROOT)
include(joinpath(ROOT, "src", "juliaDEC.jl")); include(joinpath(ROOT, "src", "hodge_stars.jl"))
const OUT = joinpath(@__DIR__, "..", "..", "output", "gs_data"); mkpath(OUT)
lines = String[]
logp(s) = (println(s); flush(stdout); push!(lines, s))

# ---------------------------------------------------------------- test fields
const TP = 2pi
fields = (a0 = x -> [1.0, 0.0],
          b0 = x -> [0.0, 1.0],
          u1 = x -> [sin(TP * x[1]) * cos(TP * x[2]), cos(TP * x[1]) * sin(TP * x[2])],
          d  = x -> [sin(TP * x[1]) + cos(TP * x[2]), sin(TP * x[2]) + 0.5cos(TP * x[1])])
supn = (a0 = 1.0, b0 = 1.0, u1 = 1.0, d = maximum(norm(fields.d([i / 300, j / 300])) for i in 0:299 for j in 0:299))
# potential for the Laplacian test
pfun(x)  = sin(TP * x[1]) * sin(TP * x[2]) + 0.5cos(2TP * x[1])
gradp(x) = [TP * cos(TP * x[1]) * sin(TP * x[2]) - TP * sin(2TP * x[1]), TP * sin(TP * x[1]) * cos(TP * x[2])]
lapp(x)  = -2TP^2 * sin(TP * x[1]) * sin(TP * x[2]) - 2TP^2 * cos(2TP * x[1])

hdiag(mesh) = spdiagm(edge_lengths(mesh, false) ./ edge_lengths(mesh, true))

"max_e |(H R u - R~ *u)_e| / (|*e| ||u||)"
function commuting_defect(mesh, H, k)
    α = reduce_primal_edges(mesh, fields[k]).values
    β = reduce_dual_edges(mesh, star_field(fields[k])).values
    return maximum(abs.(H * α .- β) ./ edge_lengths(mesh, false)) / supn[k]
end

"Laplacian truncation error relative to the exact cell average:  max_v |(d0'(H R grad p - R~ *grad p))_v| / |v~| / max|Δp|"
function laplacian_truncation(mesh, H)
    d0 = d0_matrix_primal(mesh)
    α = reduce_primal_edges(mesh, gradp).values
    β = reduce_dual_edges(mesh, star_field(gradp)).values
    τ = (d0' * (H * α .- β)) ./ mesh.cells.area
    return maximum(abs.(τ)) / (4TP^2)            # 4TP² = sup |Δp|
end

"(μ = λmin of D^-1/2 sym(H) D^-1/2 with D = DEC,  # Re λ(d0'Hd0) < 0,  max |Im λ| / max |λ|)"
function spectrum_stats(mesh, H)
    Hd = hdiag(mesh); Dm = Diagonal(1 ./ sqrt.(diag(Hd)))
    Hs = Matrix((H + H') / 2)
    μ = eigmin(Symmetric(Dm * Hs * Dm))
    d0 = d0_matrix_primal(mesh)
    λ = eigvals(Matrix(d0' * H * d0))
    λmax = maximum(abs.(λ))
    return μ, count(x -> real(x) < -1e-9 * λmax, λ), maximum(abs.(imag.(λ))) / λmax
end

const PARTS = get(ENV, "PARTS", "123")      # e.g. PARTS=3 runs only Part 3

# ============================================================================
if occursin('1', PARTS)
logp("Part 1: the two stars of hodge_stars.jl against  H R_h u = R~_h *u")
logp("   asym = ||H-H'||_F/||H||_F ;  mu = lambda_min(sym H) relative to DEC ;  neg = #Re λ(d0'Hd0) < 0 ;  Im = max|Im λ|/max|λ|")
logp("   r(u) = max_e |(H R u - R~ *u)_e| / (|*e| ||u||) ;  tau = Laplacian truncation (max norm, relative)")
p1 = Dict{Any,Any}()
for mn in ("mesh2", "mesh3", "mesh4", "mesh5")
    mesh = VoronoiMesh("meshes/$mn.nc")
    h = sqrt(mesh.x_period * mesh.y_period / mesh.cells.n)
    stars = [("DEC diagonal", hdiag(mesh)),
             ("lift  [P1]^2", hodge_star_lift_matrix(mesh; only_linear = true)),
             ("lift  [P2]^2", hodge_star_lift_matrix(mesh; only_linear = false)),
             ("sandw [P1]^2", hodge_star_sandwich_matrix(mesh; only_linear = true)),
             ("sandw [P2]^2", hodge_star_sandwich_matrix(mesh; only_linear = false))]
    logp(@sprintf("%s  (h = %.4f, %d edges)", mn, h, mesh.edges.n))
    for (name, H) in stars
        asym = norm(H - H') / norm(H)
        spec = mn == "mesh5" ? "      (skipped on mesh5)       " :
               (s = spectrum_stats(mesh, H); @sprintf("mu %+.3f  neg %4d  Im %.1e", s...))
        r = Dict(k => commuting_defect(mesh, H, k) for k in (:a0, :b0, :u1, :d))
        τ = laplacian_truncation(mesh, H)
        p1[(mn, name)] = (h = h, r = r, τ = τ)
        logp(@sprintf("   %-13s asym %.2e  %s   r(const) %.2e  r(u1) %.2e  r(d) %.2e   tau %.2e",
                      name, asym, spec, max(r[:a0], r[:b0]), r[:u1], r[:d], τ))
    end
end
logp("\n   observed orders (mesh2->3->4->5) of r(d) and tau:")
for name in ("DEC diagonal", "lift  [P1]^2", "lift  [P2]^2", "sandw [P1]^2", "sandw [P2]^2")
    ms = ("mesh2", "mesh3", "mesh4", "mesh5")
    rd = [p1[(m, name)].r[:d] for m in ms]; τs = [p1[(m, name)].τ for m in ms]
    logp(@sprintf("   %-13s r(d): %s     tau: %s", name,
                  join([@sprintf("%5.2f", log2(rd[i] / rd[i+1])) for i in 1:3], " "),
                  join([@sprintf("%5.2f", log2(τs[i] / τs[i+1])) for i in 1:3], " ")))
end
end

# ============================================================================
occursin('2', PARTS) && let mesh = VoronoiMesh("meshes/mesh3.nc"), e = 17
    logp("\nPart 2: change of basis of [P1]^2 on one stencil (mesh3, edge 17), f -> f B with B random")
    L = lsq_lift(mesh, e; only_linear = true)
    M = diamond_mass_matrix(mesh, e, L.x0, true, L.s)
    b = star_basis_on_edge(mesh, e, L.x0, true, L.s, false, nothing)
    Random.seed!(1); B = randn(6, 6) + 3I
    A2 = L.A * B; Π2 = pinv(A2); M2 = B' * M * B; b2 = B' * b
    logp(@sprintf("   sandwich  ||Π'ᵀM'Π' - ΠᵀMΠ|| / ||ΠᵀMΠ|| = %.1e", norm(Π2' * M2 * Π2 - L.Pi' * M * L.Pi) / norm(L.Pi' * M * L.Pi)))
    logp(@sprintf("   stabiliser ||A'Π' - AΠ||                 = %.1e", norm(A2 * Π2 - L.Q)))
    logp(@sprintf("   lift row  ||b'ᵀΠ' - bᵀΠ|| / ||bᵀΠ||      = %.1e", norm(b2' * Π2 - b' * L.Pi) / norm(b' * L.Pi)))
    # what DOES change the star: the choice of left inverse (weighted least squares)
    Lw = lsq_lift(mesh, e; only_linear = true, metric = true)
    logp(@sprintf("   other left inverse (metric=true):  ||ΠwᵀMΠw - ΠᵀMΠ|| / ||ΠᵀMΠ|| = %.2e  (ΠwA = I: %.1e)",
                  norm(Lw.Pi' * M * Lw.Pi - L.Pi' * M * L.Pi) / norm(L.Pi' * M * L.Pi), norm(Lw.Pi * Lw.A - I)))
end

# ============================================================================
#  Part 3: smallest symmetric correction of DEC that satisfies the commuting identity on [P_k]^2
# ============================================================================
const GLX = [-0.8611363115940526, -0.3399810435848563, 0.3399810435848563, 0.8611363115940526]
const GLW = [0.3478548451374538, 0.6521451548625461, 0.6521451548625461, 0.3478548451374538]
nb_of(k) = k == 0 ? 2 : k == 1 ? 6 : 12

"∫_{p1→p2} f_m · dx for the basis about x0 scaled by s (rotate: ∫ (*f_m) · dx).  Exact for degree ≤ 7."
function seg_moments(p1, p2, x0, s, k; rotate = false)
    N = nb_of(k); out = zeros(N); t = p2 .- p1
    for (ξ, w) in zip(GLX, GLW)
        F = basis_at(p1 .+ (ξ + 1) / 2 .* t, x0, k <= 1, s)[:, 1:N]
        rotate && (F = _ROT90 * F)
        out .+= (w / 2) .* (F' * t)
    end
    return out
end

function ring(mesh, e, r)
    r == 2 && return unique(vcat(e, get_neighbors_2level_primal(mesh, e)))
    r == 3 && return unique(vcat(e, get_neighbors_3level_primal(mesh, e)))
    # r == 4: union of the 2-level stencils of the 2-level neighbours
    return unique(vcat([get_neighbors_2level_primal(mesh, j) for j in ring(mesh, e, 2)]...))
end

"""
Smallest (Frobenius) symmetric C on the symmetrised r-ring pattern with (Hd + C) exact on [P_k]^2 row by row.
Returns (H, info).  info.res is the relative least-squares residual of the constraints.
"""
function symmetric_correction(mesh, k, r; ε = 1e-14, B = nothing)        # B: optional nb_of(k) x m basis of a subspace of [P_k]^2
    n = mesh.edges.n; N = B === nothing ? nb_of(k) : size(B, 2)
    proj(v) = B === nothing ? v : B' * v
    lp = edge_lengths(mesh, true); ld = edge_lengths(mesh, false); hd = ld ./ lp
    S = [ring(mesh, e, r) for e in 1:n]
    nb = [Set(S[e]) for e in 1:n]
    for e in 1:n, j in S[e]; push!(nb[j], e); end
    nbv = [sort(collect(nb[e])) for e in 1:n]
    pid = Dict{Tuple{Int,Int},Int}()
    for e in 1:n, j in nbv[e]
        key = minmax(e, j); haskey(pid, key) || (pid[key] = length(pid) + 1)
    end
    np = length(pid)
    I_, J_, V_ = Int[], Int[], Float64[]; rhs = zeros(n * N)
    for e in 1:n
        xv = mesh.edges.position[e]; x0 = _pt(xv); s = lp[e]
        q1, q2 = edge_endpoints(mesh, e, false; about = xv)
        b = proj(seg_moments(q1, q2, x0, s, k; rotate = true))
        for j in nbv[e]
            p1_, p2_ = edge_endpoints(mesh, j, true; about = xv)
            a = proj(seg_moments(p1_, p2_, x0, s, k))
            j == e && (b .-= hd[e] .* a)
            for m in 1:N
                push!(I_, (e - 1) * N + m); push!(J_, pid[minmax(e, j)]); push!(V_, a[m] / ld[e])
            end
        end
        rhs[(e-1)*N+1:e*N] .= b ./ ld[e]
    end
    G = sparse(I_, J_, V_, n * N, np)
    w = [k_[1] == k_[2] ? 1.0 : 2.0 for k_ in sort(collect(keys(pid)), by = x -> pid[x])]
    K = G * Diagonal(1 ./ w) * G'
    δ = ε * sum(diag(K)) / size(K, 1)
    λ = cholesky(Symmetric(K + δ * I)) \ rhs
    c = (1 ./ w) .* (G' * λ)
    res = norm(rhs) == 0 ? 0.0 : norm(G * c - rhs) / norm(rhs)     # k = 0: DEC is already exact, rhs = 0
    Ii, Jj, Vv = Int[], Int[], Float64[]
    for (key, p) in pid
        push!(Ii, key[1]); push!(Jj, key[2]); push!(Vv, c[p])
        key[1] != key[2] && (push!(Ii, key[2]); push!(Jj, key[1]); push!(Vv, c[p]))
    end
    C = sparse(Ii, Jj, Vv, n, n)
    H = hdiag(mesh) + C
    return H, (res = res, cmax = maximum(abs.(c)) / maximum(hd), rows = n * N, unknowns = np,
               per_row = sum(length, nbv) / n)
end

occursin('3', PARTS) && let
logp("\nPart 3: smallest symmetric correction H = Hd + C, C = C', exact on [P_k]^2 row by row")
logp("   res = relative least-squares residual of the constraints (0 <=> a symmetric exact star exists on that pattern)")
logp("   |C|/|Hd| = largest correction entry relative to the largest DEC entry")
p3 = Dict{Any,Any}()
for (k, r) in ((0, 2), (1, 2), (1, 3), (2, 3), (2, 4)), mn in ("mesh2", "mesh3", "mesh4", "mesh3_regular", "mesh4_regular")
    (k, r) == (2, 4) && mn in ("mesh4", "mesh4_regular") && continue
    mesh = VoronoiMesh("meshes/$mn.nc")
    H, info = symmetric_correction(mesh, k, r)
    hl = sqrt(mesh.x_period * mesh.y_period / mesh.cells.n)
    reg = endswith(mn, "regular")
    spec = (reg || mesh.edges.n > 4000) ? "" : (s = spectrum_stats(mesh, H); @sprintf("mu %+.3f", s[1]))
    rd = commuting_defect(mesh, H, :d); ru = commuting_defect(mesh, H, :u1); τ = laplacian_truncation(mesh, H)
    p3[(k, r, mn)] = (h = hl, rd = rd, τ = τ, res = info.res)
    logp(@sprintf("   k=%d ring %d  %-14s %6d eqs %6d unknowns (%.1f/row)  res %.1e  |C|/|Hd| %.2e  r(u1) %.2e  r(d) %.2e  tau %.2e  %s",
                  k, r, mn, info.rows, info.unknowns, info.per_row, info.res, info.cmax, ru, rd, τ, spec))
end
end

# ============================================================================
#  Part 4: refinement of the two decisive cases, and a Poisson solve
#    [P1]^2 on the 3-ring pattern (meshes 2-5): does the correction stay bounded, is the order 2?
#    [P2]^2 on the 4-ring pattern (meshes 2-4): the skew forces the correction to grow like 1/h.
# ============================================================================
occursin('4', PARTS) && let
logp("\nPart 4: refinement of the [P1]^2 / 3-ring and [P2]^2 / 4-ring symmetric stars; Poisson -Δp = f, max nodal error")
for (k, r, ms) in ((1, 3, ("mesh2", "mesh3", "mesh4", "mesh5")), (2, 4, ("mesh2", "mesh3", "mesh4")))
    prev = nothing
    for mn in ms
        mesh = VoronoiMesh("meshes/$mn.nc")
        t0 = time()
        H, info = symmetric_correction(mesh, k, r)
        spec = mesh.edges.n > 4000 ? "mu  (n/a)" : @sprintf("mu %+.3f", spectrum_stats(mesh, H)[1])
        pe = [pfun(_pt(x)) for x in mesh.cells.position]; pe .-= sum(pe) / length(pe)
        eP(Hx) = maximum(abs.(solve_poisson_primal(mesh, Hx, x -> -lapp(x)) .- pe)) / maximum(abs.(pe))
        cur = (rd = commuting_defect(mesh, H, :d), ru = commuting_defect(mesh, H, :u1), τ = laplacian_truncation(mesh, H),
               pS = eP(H), pD = eP(hdiag(mesh)))
        ords = prev === nothing ? "" : @sprintf("   orders r(d) %.2f  tau %.2f  poisson %.2f", log2(prev.rd / cur.rd), log2(prev.τ / cur.τ), log2(prev.pS / cur.pS))
        logp(@sprintf("   k=%d ring %d  %-6s res %.1e  |C|/|Hd| %.2e  %s  r(u1) %.2e  r(d) %.2e  tau %.2e  poisson: sym %.2e  DEC %.2e  (%.0f s)%s",
                      k, r, mn, info.res, info.cmax, spec, cur.ru, cur.rd, cur.τ, cur.pS, cur.pD, time() - t0, ords))
        prev = cur
    end
end
end

# PARTS="" only loads the functions
isempty(lines) || write(joinpath(OUT, PARTS == "123" ? "commuting_check.txt" : "commuting_check_part$(PARTS).txt"), join(lines, "\n") * "\n")
