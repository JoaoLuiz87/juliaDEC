# ============================================================================
#  compatibility_check.jl -- which degrees k admit a symmetric star exact on [P_k]^2 (Section 3.5 of
#  commuting_symmetric_note.tex).
#
#      julia -t 4 scripts/barrier/compatibility_check.jl      (writes output/gs_data/compatibility_check.txt)
#
#  Theorem: for every matrix polynomial κ with κ(-z) = -κ(z)ᵀ and deg κ ≤ k, the multipliers
#      φ_e(y) = ∫_{x∈e} κ(y-x)ᵀ t_e ds_x          (one field of [P_k]^2 per edge)
#  lie in the left kernel of the constraint matrix G of symmetric_correction, and solvability needs
#      I(κ) = Σ_e ∫_ẽ ∫_e t_eᵀ κ(y-x) t_e = 0.
#  Families: deg 0: J ; deg 1: z_c K^c, K^c symmetric (6) ; deg 2: q(z) J (3).  Sizes 1, 7, 10.
#  Part A  left-kernel dimension of G versus the κ-family (is the family the whole kernel?).
#  Part B  the compatibility tensors of degree 1 (zero by the circumcentric identity) and degree 3.
# ============================================================================
ENV["PARTS"] = ""                                   # load the functions of commuting_check.jl only
include(joinpath(@__DIR__, "..", "stars", "commuting_check.jl"))

"G and rhs of symmetric_correction (same rows, same scaling)"
function constraint_system(mesh, k, r)
    n = mesh.edges.n; N = nb_of(k)
    lp = edge_lengths(mesh, true); ld = edge_lengths(mesh, false); hd = ld ./ lp
    S = [ring(mesh, e, r) for e in 1:n]
    nb = [Set(S[e]) for e in 1:n]
    for e in 1:n, j in S[e]; push!(nb[j], e); end
    nbv = [sort(collect(nb[e])) for e in 1:n]
    pid = Dict{Tuple{Int,Int},Int}()
    for e in 1:n, j in nbv[e]
        key = minmax(e, j); haskey(pid, key) || (pid[key] = length(pid) + 1)
    end
    I_, J_, V_ = Int[], Int[], Float64[]; rhs = zeros(n * N)
    for e in 1:n
        xv = mesh.edges.position[e]; x0 = _pt(xv); s = lp[e]
        q1, q2 = edge_endpoints(mesh, e, false; about = xv)
        b = seg_moments(q1, q2, x0, s, k; rotate = true)
        for j in nbv[e]
            p1_, p2_ = edge_endpoints(mesh, j, true; about = xv)
            a = seg_moments(p1_, p2_, x0, s, k)
            j == e && (b .-= hd[e] .* a)
            for m in 1:N
                push!(I_, (e - 1) * N + m); push!(J_, pid[minmax(e, j)]); push!(V_, a[m] / ld[e])
            end
        end
        rhs[(e-1)*N+1:e*N] .= b ./ ld[e]
    end
    return sparse(I_, J_, V_, n * N, length(pid)), rhs
end

const Jm = [0.0 -1.0; 1.0 0.0]
function kappas(k)
    κs = Function[z -> Jm]
    k >= 1 && for c in 1:2, K in ([1.0 0; 0 0], [0.0 0; 0 1], [0.0 1; 1 0])
        push!(κs, z -> z[c] * K)
    end
    k >= 2 && for q in (z -> z[1]^2, z -> z[1] * z[2], z -> z[2]^2)
        push!(κs, z -> q(z) * Jm)
    end
    return κs
end

"multiplier vector of one κ, scaled like the rows of G: coefficients of φ_e in the basis of row e, times |ẽ|"
function multiplier(mesh, k, κ)
    n = mesh.edges.n; N = nb_of(k); lp = edge_lengths(mesh, true); ld = edge_lengths(mesh, false)
    λ = zeros(n * N); rng = MersenneTwister(3)
    for e in 1:n
        xv = mesh.edges.position[e]; x0 = _pt(xv); s = lp[e]
        p1, p2 = edge_endpoints(mesh, e, true; about = xv); te = (p2 .- p1) ./ norm(p2 .- p1)
        φ(y) = sum((w / 2) * lp[e] * (κ(y .- (p1 .+ (ξ + 1) / 2 .* (p2 .- p1))))' * te for (ξ, w) in zip(GLX, GLW))
        ys = [x0 .+ s .* (2 .* rand(rng, 2) .- 1) for _ in 1:12]          # φ_e is in the span: the fit is exact
        A = vcat([basis_at(y, x0, k <= 1, s)[:, 1:N] for y in ys]...)
        λ[(e-1)*N+1:e*N] .= (A \ vcat([φ(y) for y in ys]...)) .* ld[e]
    end
    return λ
end

logp("Part A: left kernel of the constraint matrix G (symmetric, row-exact on [P_k]^2) versus the κ-family")
logp("   null = rows - rank(G) ;  fam = size and rank of the κ-family ;  ker = max ||Gᵀλ||/(||λ|| σmax) ;  comp = max |λ·rhs|/(||λ|| ||rhs||)")
for (mn, k, r) in (("mesh2", 1, 3), ("mesh3", 1, 3), ("mesh2", 2, 4), ("mesh2_regular", 2, 3))
    mesh = VoronoiMesh("meshes/$mn.nc")
    G, rhs = constraint_system(mesh, k, r)
    σ = svdvals(Matrix(G))
    nullL = size(G, 1) - count(>(1e-9 * σ[1]), σ)
    Λ = hcat([multiplier(mesh, k, κ) for κ in kappas(k)]...)
    svΛ = svdvals(Λ)
    ker = maximum(norm(G' * Λ[:, i]) / (norm(Λ[:, i]) * σ[1]) for i in 1:size(Λ, 2))
    comp = maximum(abs(dot(Λ[:, i], rhs)) / (norm(Λ[:, i]) * norm(rhs)) for i in 1:size(Λ, 2))
    logp(@sprintf("   %-14s k=%d ring %d  %5d rows %5d unknowns   null %3d   fam %2d (rank %2d)   ker %.1e   comp %.1e",
                  mn, k, r, size(G)..., nullL, size(Λ, 2), count(>(1e-8 * svΛ[1]), svΛ), ker, comp))
    if k == 2
        λb = multiplier(mesh, k, z -> (z[1]^2 + z[2]^2) * Matrix(1.0I, 2, 2))
        logp(@sprintf("      contrast κ = |z|^2 I (κ(-z) ≠ -κ(z)ᵀ):  ||Gᵀλ||/(||λ|| σmax) = %.2f", norm(G' * λb) / (norm(λb) * σ[1])))
    end
end

logp("\nPart B: compatibility tensors  T1 = Σ_e ∫_ẽ∫_e (y-x) ⊗ t_e ⊗ t_e  and  T3 = Σ_e ∫_ẽ∫_e (y-x)^{⊗3} ⊗ t_e ⊗ t_e")
logp("   |T3| is relative to Σ_e |e| |ẽ| |e|^3; T1 = Σ_e |e||ẽ|δ_e n_e ⊗ t_e ⊗ t_e is zero by the circumcentric identity")
for mn in ("mesh2", "mesh3", "mesh4", "mesh3_regular")
    mesh = VoronoiMesh("meshes/$mn.nc")
    T1 = zeros(2, 2, 2); T3 = zeros(2, 2, 2, 2, 2); scale = 0.0
    for e in 1:mesh.edges.n
        xv = mesh.edges.position[e]
        p1, p2 = edge_endpoints(mesh, e, true; about = xv); q1, q2 = edge_endpoints(mesh, e, false; about = xv)
        le, ld = norm(p2 .- p1), norm(q2 .- q1); t = (p2 .- p1) ./ le
        scale += le * ld * le^3
        for (ξ, w) in zip(GLX, GLW), (η, v) in zip(GLX, GLW)
            z = (q1 .+ (η + 1) / 2 .* (q2 .- q1)) .- (p1 .+ (ξ + 1) / 2 .* (p2 .- p1)); W = (w / 2) * le * (v / 2) * ld
            for c in 1:2, i in 1:2, j in 1:2
                T1[c, i, j] += W * z[c] * t[i] * t[j]
                for a in 1:2, b in 1:2
                    T3[a, b, c, i, j] += W * z[a] * z[b] * z[c] * t[i] * t[j]
                end
            end
        end
    end
    logp(@sprintf("   %-14s |T1| = %.1e   |T3| / Σ|e||ẽ||e|^3 = %.2e", mn, norm(T1), norm(T3) / scale))
end
write(joinpath(OUT, "compatibility_check.txt"), join(lines, "\n") * "\n")
