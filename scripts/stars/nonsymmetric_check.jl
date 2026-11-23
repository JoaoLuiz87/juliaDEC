# ============================================================================
#  nonsymmetric_check.jl -- past the second-order ceiling with ONE unknown per edge by giving up exact symmetry
#  (route "symmetry" of Section 5 of commuting_symmetric_note.tex).
#
#      julia -t 4 scripts/stars/nonsymmetric_check.jl      (writes output/gs_data/nonsymmetric_check.txt)
#
#  H = H0 + C, C row by row the smallest (Frobenius) correction with (H0 + C) exact on [P_k]^2 on the r-ring of the
#  edge.  No symmetry constraint, so the rows decouple (one small least-norm problem per edge) and the skew of the
#  pairing does not force C to grow.  Bases H0: DEC, and the symmetric [P1]^2 star of Section 4.
#  The price is the antisymmetric part: d0'Hd0 gets complex eigenvalues, and in a wave equation x'' = -Lx the mode
#  with eigenvalue λ grows like exp(|Im √λ| t).  gamma = max |Im √λ| / √max|λ| is that growth rate in units of the
#  fastest frequency (L = A^-1 d0'Hd0, A = cell areas); 1/(2π gamma) is the e-folding time in fastest-wave periods.
# ============================================================================
ENV["PARTS"] = ""                                   # load the functions of commuting_check.jl only
include(joinpath(@__DIR__, "commuting_check.jl"))

"Smallest row-wise C on the r-ring (not symmetrised) with (H0 + C) exact on [P_k]^2."
function rowwise_correction(mesh, H0, k, r)
    n = mesh.edges.n; N = nb_of(k)
    lp = edge_lengths(mesh, true); ld = edge_lengths(mesh, false)
    H0t = sparse(H0')                               # column e of H0t = row e of H0
    Ii, Jj, Vv = Int[], Int[], Float64[]; cmax = 0.0
    for e in 1:n
        xv = mesh.edges.position[e]; x0 = _pt(xv); s = lp[e]
        q1, q2 = edge_endpoints(mesh, e, false; about = xv)
        b = seg_moments(q1, q2, x0, s, k; rotate = true)
        mom(j) = (q = edge_endpoints(mesh, j, true; about = xv); seg_moments(q[1], q[2], x0, s, k))
        # residual of the base row (its whole pattern, which may reach past the ring)
        for p in nzrange(H0t, e)
            b .-= nonzeros(H0t)[p] .* mom(rowvals(H0t)[p])
        end
        Se = ring(mesh, e, r)
        A = reduce(hcat, [mom(j) for j in Se])
        c = pinv(A) * b
        append!(Ii, fill(e, length(Se))); append!(Jj, Se); append!(Vv, c)
        cmax = max(cmax, maximum(abs.(c)))
    end
    C = sparse(Ii, Jj, Vv, n, n)
    return H0 + C, cmax / maximum(diag(hdiag(mesh)))
end

"Gershgorin lower bound on λmin(D^-1/2 sym(H) D^-1/2): g > 0 certifies that sym(H) is SPD, i.e. d0'Hd0 is coercive"
function gersh_sym(mesh, H)
    Dm = Diagonal(1 ./ sqrt.(diag(hdiag(mesh)))); S = sparse(Dm * ((H + H') / 2) * Dm)
    return minimum(2 * S[e, e] - sum(abs, nonzeros(S)[nzrange(S, e)]) for e in 1:size(S, 1))
end

"(μ = λmin of D^-1/2 sym(H) D^-1/2,  max|Im λ|/max|λ| of A^-1 d0'Hd0,  gamma = max|Im √λ| / √max|λ|)"
function wave_stats(mesh, H)
    Hd = hdiag(mesh); Dm = Diagonal(1 ./ sqrt.(diag(Hd)))
    μ = eigmin(Symmetric(Matrix(Dm * ((H + H') / 2) * Dm)))
    d0 = d0_matrix_primal(mesh)
    λ = eigvals(Matrix(Diagonal(1 ./ mesh.cells.area) * (d0' * H * d0)))
    λmax = maximum(abs.(λ))
    return μ, maximum(abs.(imag.(λ))) / λmax, maximum(abs.(imag.(sqrt.(complex.(λ))))) / sqrt(λmax)
end

if get(ENV, "NS_MAIN", "1") == "1"                 # NS_MAIN=0 only loads the functions (used by sw_linear_check.jl)
logp("Past the ceiling by giving up exact symmetry: H = H0 + C, C the smallest row-wise correction exact on [P_k]^2")
logp("   asym = ||H-H'||_F/||H||_F ;  |C|/|Hd| = largest correction entry / largest DEC entry ;  mu = lambda_min(sym H) rel. to DEC")
logp("   Im = max|Im λ|/max|λ| and gamma = max|Im √λ|/√max|λ| for L = A^-1 d0'Hd0 (spectrum on meshes 2-4 only)")
logp("   g = Gershgorin lower bound on mu (g > 0 certifies sym(H) SPD, so the Laplacian d0'Hd0 is coercive)")
logp("   r(u) = max_e |(H R u - R~ *u)_e| / (|*e| ||u||) ;  tau = Laplacian truncation ;  poisson = max nodal error (relative)")
res = Dict{Any,Any}()
names = ("DEC diagonal", "sym [P1] (3-ring)", "DEC + ns[P2] 3-ring", "symP1 + ns[P2] 3-ring")
for mn in ("mesh2", "mesh3", "mesh4", "mesh5")
    mesh = VoronoiMesh("meshes/$mn.nc")
    h = sqrt(mesh.x_period * mesh.y_period / mesh.cells.n)
    Hd = hdiag(mesh)
    Hs, _ = symmetric_correction(mesh, 1, 3)
    stars = Any[(names[1], Hd, 0.0), (names[2], Hs, NaN)]
    H, cm = rowwise_correction(mesh, Hd, 2, 3); push!(stars, (names[3], H, cm))
    H, cm = rowwise_correction(mesh, Hs, 2, 3); push!(stars, (names[4], H, cm))
    logp(@sprintf("%s  (h = %.4f, %d edges)", mn, h, mesh.edges.n))
    pe = [pfun(_pt(x)) for x in mesh.cells.position]; pe .-= sum(pe) / length(pe)
    for (name, H, cm) in stars
        asym = norm(H - H') / norm(H)
        spec = mn == "mesh5" ? "                 (n/a on mesh5)              " :
               (s = wave_stats(mesh, H); @sprintf("mu %+.3f  Im %.1e  gamma %.1e", s...))
        rd = commuting_defect(mesh, H, :d); ru = commuting_defect(mesh, H, :u1); τ = laplacian_truncation(mesh, H)
        pS = maximum(abs.(solve_poisson_primal(mesh, H, x -> -lapp(x)) .- pe)) / maximum(abs.(pe))
        res[(mn, name)] = (h = h, rd = rd, τ = τ, p = pS)
        logp(@sprintf("   %-22s asym %.2e  |C|/|Hd| %s  g %.3f  %s  r(u1) %.2e  r(d) %.2e  tau %.2e  poisson %.2e",
                      name, asym, isnan(cm) ? "  (sym)  " : @sprintf("%.2e", cm), gersh_sym(mesh, H), spec, ru, rd, τ, pS))
    end
end
logp("\n   observed orders (mesh2->3->4->5) of r(d), tau and poisson:")
for name in names
    ms = ("mesh2", "mesh3", "mesh4", "mesh5")
    o(f) = join([@sprintf("%5.2f", log2(f(res[(ms[i], name)]) / f(res[(ms[i+1], name)]))) for i in 1:3], " ")
    logp(@sprintf("   %-22s r(d): %s    tau: %s    poisson: %s", name, o(x -> x.rd), o(x -> x.τ), o(x -> x.p)))
end

# for H exact on [P_k]^2:  (R u)'(H - H')(R v) = s_h(u,v) + <R u, eps(v)> - <R v, eps(u)>,  so a_h -> s_h when k >= 2
logp("\n   the antisymmetric part carries the skew: a_h = (R_h u2)'(H - H')(R_h d) for DEC + ns[P2], against s_h(u2,d)")
u2(x) = [sin(TP * x[1]) * cos(TP * x[2]), -cos(TP * x[1]) * sin(TP * x[2])]
for mn in ("mesh2", "mesh3", "mesh4", "mesh5")
    mesh = VoronoiMesh("meshes/$mn.nc")
    h = sqrt(mesh.x_period * mesh.y_period / mesh.cells.n)
    α(f) = reduce_primal_edges(mesh, f).values
    β(f) = reduce_dual_edges(mesh, star_field(f)).values
    s = dot(α(u2), β(fields.d)) - dot(α(fields.d), β(u2))
    H, _ = rowwise_correction(mesh, hdiag(mesh), 2, 3)
    a = dot(α(u2), (H - H') * α(fields.d))
    logp(@sprintf("   %s  s_h %+.4e  a_h %+.4e  a_h/s_h %.4f  |a_h - s_h|/h^2 %.1e", mn, s, a, a / s, abs(a - s) / h^2))
end
write(joinpath(OUT, "nonsymmetric_check.txt"), join(lines, "\n") * "\n")
end
