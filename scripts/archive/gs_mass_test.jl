# ============================================================================
#  gs_mass_test.jl -- why B_T = R_Tᵀ M_T R_T and not R_Tᵀ R_T: the Galerkin star with the mass matrix M_T replaced
#  by another 6 x 6 weight X_T, on the SCVT meshes 2-5.
#
#      julia scripts/archive/gs_mass_test.jl          (writes output/gs_data/mass_test.txt)
#
#  X_T = M_T      the Galerkin star of galerkin_star.jl (reference)
#  X_T = I        plain R_Tᵀ R_T
#  X_T = Ā I      R_Tᵀ R_T times the mean triangle area Ā = |Ω| / #triangles
#  X_T = |T| I    "lumped": the area of T kept, its second moments dropped
#  For each X the corrections Z_T are the minimal ones (in the X_T norm) that satisfy the moment conditions of the
#  star B_T = R_Tᵀ X_T R_T.  Row e of that star on a linear u is Σ_T R_T[:, e]ᵀ X_T c_T(u), c_T(u) = M_T⁻¹ ∫_T φ u, so
#  the mixed masses N_{T,e} of galerkin_star.jl become X_T M_T⁻¹ N_{T,e}.  Solvability is reported as the part of the
#  right-hand side along the kernel of G: its 4 constant kernel vectors and the 8 linear ones.
# ============================================================================
using VoronoiMeshes, TensorsLite, TensorsLiteGeometry, SmallCollections, LinearAlgebra, NCDatasets
using StaticArrays, SparseArrays, Printf
Base.similar(bc::Base.Broadcast.Broadcasted, ::Type{Union{}}) = Array{Union{}}(undef, length.(axes(bc)))
const ROOT = joinpath(@__DIR__, "..", ".."); cd(ROOT)
include(joinpath(ROOT, "src", "juliaDEC.jl")); include(joinpath(ROOT, "src", "hodge_stars.jl")); include(joinpath(ROOT, "src", "galerkin_star.jl"))
const OUT = joinpath(@__DIR__, "..", "..", "output", "gs_data"); mkpath(OUT)
lines = String[]
logp(s) = (println(s); flush(stdout); push!(lines, s))
u1(x) = [sin(2pi * x[1]) * cos(2pi * x[2]), cos(2pi * x[1]) * sin(2pi * x[2])]

"the Galerkin star with weight X_T in place of M_T (kind = :M, :I, :mean, :area)"
function star_X(mesh, kind; s = nothing, tau = 1.0, reg = 1e-13)
    E = mesh.edges.n; nT = mesh.vertices.n
    s === nothing && (s = sum(edge_length(mesh, e, true) for e in 1:E) / E)
    cells = [gs_cell(mesh, t, s) for t in 1:nT]
    edata = [gs_edge(mesh, e, s) for e in 1:E]
    Ā = mesh.x_period * mesh.y_period / nT; I6 = Matrix(1.0I, 6, 6)
    Xs = [kind == :M ? C.M : kind == :I ? I6 : kind == :mean ? Ā * I6 : _tri_area(C.tri...) * I6 for C in cells]
    ccsum = sum(X[1, 1] for X in Xs) / (mesh.x_period * mesh.y_period)   # Σ_T X_T[1,1] / |Ω|
    βall = vcat([ed.β for ed in edata]...); rhs = copy(βall)
    blocks = Dict{Tuple{Int,Int},Matrix{Float64}}()
    Ns = Vector{Vector{Matrix{Float64}}}(undef, nT)
    for (k, C) in enumerate(cells)
        Nt = [Xs[k] * (C.M \ Nq) for Nq in C.Nm]; Ns[k] = Nt
        XN = [Xs[k] \ n for n in Nt]
        for (q1, e1) in enumerate(C.S)
            rhs[_b6(e1)] .-= Nt[q1]' * C.R0[:, q1]
            for (q2, e2) in enumerate(C.S)
                w = dot(view(C.W, q1, :), view(C.W, q2, :)); abs(w) < 1e-15 && continue
                blk = w * (Nt[q1]' * XN[q2])
                haskey(blocks, (e1, e2)) ? (blocks[(e1, e2)] .+= blk) : (blocks[(e1, e2)] = blk)
            end
        end
    end
    Ii, Jj, Vv = Int[], Int[], Float64[]
    for ((e1, e2), blk) in blocks, jj in 1:6, ii in 1:6
        push!(Ii, 6 * (e1 - 1) + ii); push!(Jj, 6 * (e2 - 1) + jj); push!(Vv, blk[ii, jj])
    end
    G = sparse(Ii, Jj, Vv, 6E, 6E); G = (G + G') / 2
    Kv = gs_kernel_vectors(edata, s)
    Qa = Matrix(qr(Kv).Q)[:, 1:12]; Qc = Matrix(qr(Kv[:, 1:4]).Q)[:, 1:4]
    oa = norm(Qa' * rhs); oc = norm(Qc' * rhs)
    obs_const = oc / norm(βall); obs_lin = sqrt(max(oa^2 - oc^2, 0.0)) / norm(βall)
    r = rhs - Qa * (Qa' * rhs)
    μ = cholesky(Symmetric(G + reg * maximum(abs, diag(G)) * I)) \ r
    Hi, Hj, Hv = Int[], Int[], Float64[]; resid = -copy(βall); dev = 0.0
    for (k, C) in enumerate(cells)
        X = Xs[k]; Z = zeros(6, size(C.W, 2))
        for (q, e) in enumerate(C.S)
            Z .+= (X \ Ns[k][q]) * μ[_b6(e)] * C.W[q, :]'
        end
        R = C.R0 + Z * C.W'
        dev = max(dev, sqrt(tr(Z' * X * Z) / tr(C.R0' * X * C.R0)))
        for (q, e) in enumerate(C.S)
            resid[_b6(e)] .+= Ns[k][q]' * R[:, q]
        end
        B = R' * X * R
        if tau > 0
            Pr = I - C.P * R; Sm = Pr' * Diagonal(1 ./ C.lens .^ 2) * Pr
            B .+= (tau * tr(B) / tr(Sm)) .* Sm
        end
        B = (B + B') / 2
        for (q1, e1) in enumerate(C.S), (q2, e2) in enumerate(C.S)
            push!(Hi, e1); push!(Hj, e2); push!(Hv, B[q1, q2])
        end
    end
    H = sparse(Hi, Hj, Hv, E, E); H = (H + H') / 2
    H, (ccsum = ccsum, obs_const = obs_const, obs_lin = obs_lin, moment = norm(resid) / norm(βall), dev = dev, s = s)
end

kinds = (:M, :I, :mean, :area)
names = Dict(:M => "M_T", :I => "I", :mean => "Abar*I", :area => "|T|*I")
logp("== the Galerkin star with X_T in place of M_T")
logp(@sprintf("%-6s %-7s %8s %9s %9s %9s %8s %6s %10s %10s %9s", "mesh", "X_T", "ΣX11/|Ω|", "obs const", "obs lin",
              "moment", "rowP1", "corr", "u1 max", "u1 rms", "λmin/DEC"))
table = Dict{Symbol,Vector{Float64}}(k => Float64[] for k in kinds)
for mn in ("mesh2", "mesh3", "mesh4", "mesh5")
    mesh = VoronoiMesh("meshes/$mn.nc")
    Hd = spdiagm(edge_lengths(mesh, false) ./ edge_lengths(mesh, true))
    a = gs_hodge_error(mesh, Hd, u1)
    logp(@sprintf("%-6s %-7s %8s %9s %9s %9s %8.1e %6s %10.3e %10.3e %9.4f", mn, "DEC", "", "", "", "",
                  gs_row_exactness(mesh, Hd), "", a..., 1.0))
    for k in kinds
        H, inf = star_X(mesh, k)
        a = gs_hodge_error(mesh, H, u1); μ = gs_smallest_eigenvalue(H; D = Hd)
        push!(table[k], a[1])
        logp(@sprintf("%-6s %-7s %8.3f %9.1e %9.1e %9.1e %8.1e %6.2f %10.3e %10.3e %9.4f", mn, names[k], inf.ccsum,
                      inf.obs_const, inf.obs_lin, inf.moment, gs_row_exactness(mesh, H), inf.dev, a..., μ))
        H = nothing; GC.gc()
    end
end
logp("\nobserved orders of the u1 max error (h halves)")
for k in kinds
    v = table[k]
    logp(@sprintf("  %-7s %s", names[k], join([@sprintf("%6.2f", log2(v[i] / v[i+1])) for i in 1:length(v)-1], " ")))
end

logp("\n== dependence on the basis scale s (mesh3): s0 = mean edge length")
mesh = VoronoiMesh("meshes/mesh3.nc")
s0 = sum(edge_length(mesh, e, true) for e in 1:mesh.edges.n) / mesh.edges.n
for k in kinds
    H1, _ = star_X(mesh, k; s = s0); H2, _ = star_X(mesh, k; s = 2s0)
    a1 = gs_hodge_error(mesh, H1, u1); a2 = gs_hodge_error(mesh, H2, u1)
    logp(@sprintf("  %-7s ‖H(2s0) - H(s0)‖/‖H(s0)‖ = %.1e    u1 max: %.3e (s0)  %.3e (2s0)", names[k],
                  norm(nonzeros(H2 - H1)) / norm(nonzeros(H1)), a1[1], a2[1]))
end
write(joinpath(OUT, "mass_test.txt"), join(lines, "\n") * "\n")
