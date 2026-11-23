# ============================================================================
#  paper_runs.jl -- the numbers of paper/hodge_barrier_paper.tex that were not computed before
#
#      PART=E julia -t 4 scripts/elliptic/paper_runs.jl   Poisson P1-P3, Darcy D1-D3, meshes 1-7      -> ts_data/paper_elliptic.{jls,txt}
#      PART=S   spectrum of A^-1 D0' H D0 (M0 = A) on meshes 1-7, replication check                     -> ts_data/paper_spectra.{jls,txt}
#      PART=T   construction cost of H_P1 / H_P2, direct and Krylov solves                              -> ts_data/paper_costs.{jls,txt}
#      PART=Y   symmetric [P2]^2 star on the 4-ring, mesh 6                                             -> ts_data/paper_symp2.{jls,txt}
# ============================================================================
include(joinpath(@__DIR__, "..", "stars", "stars_cache.jl"))
using LinearAlgebra, SparseArrays, Printf, Serialization, Random
const PART = get(ENV, "PART", "E")
const OUTP = String[]
say(s) = (println(s); flush(stdout); push!(OUTP, s))
const MESHES7 = ("mesh1", "mesh2", "mesh3", "mesh4", "mesh5", "mesh6", "mesh7")
function getstars(mn, mesh)
    mn == "mesh1" || return cached_stars(mn; mesh)
    S, _ = three_stars(mesh)
    return Dict{String,Any}(k => sparse(v) for (k, v) in S)
end
d0mat(mesh) = sparse(Float64.(d0_matrix_primal(mesh)))

# ---------------------------------------------------------------------------- test problems (as in thesis_tests.jl)
function bumpder(x; x0, σ)
    a = 1 / (TP^2 * σ^2); X = TP * (x[1] - x0[1]); Y = TP * (x[2] - x0[2]); b = exp(a * (cos(X) + cos(Y) - 2))
    g = b .* [-a * TP * sin(X), -a * TP * sin(Y)]
    lap = b * (a^2 * TP^2 * (sin(X)^2 + sin(Y)^2) - a * TP^2 * (cos(X) + cos(Y)))
    return b, g, lap
end
const PTESTS = (
    (name = "P1 smooth", p = x -> sin(TP * x[1]) * sin(TP * x[2]) + 0.5cos(2TP * x[1]),
     lap = x -> -2TP^2 * sin(TP * x[1]) * sin(TP * x[2]) - 2TP^2 * cos(2TP * x[1])),
    (name = "P2 localised", p = x -> bumpder(x; x0 = (0.4, 0.6), σ = 0.06)[1], lap = x -> bumpder(x; x0 = (0.4, 0.6), σ = 0.06)[3]),
    (name = "P3 several scales", p = x -> sin(TP * x[1]) * cos(2TP * x[2]) + 0.3sin(3TP * (x[1] + x[2])),
     lap = x -> -5TP^2 * sin(TP * x[1]) * cos(2TP * x[2]) - 0.3 * 18TP^2 * sin(3TP * (x[1] + x[2]))))
gsw(x) = TP .* [cos(TP * x[1]) * sin(TP * x[2]), -sin(TP * x[1]) * cos(TP * x[2])]     # divergence free, as large as ∇p
const DTESTS = (
    (name = "D1 potential flow", p = PD, v = VD, g = x -> [0.0, 0.0], φ = PHID),
    (name = "D2 with body force", p = PD, v = x -> VD(x) .+ gsw(x), g = gsw, φ = PHID),
    (name = "D3 localised", p = x -> bumpder(x; x0 = (0.5, 0.5), σ = 0.10)[1], v = x -> -bumpder(x; x0 = (0.5, 0.5), σ = 0.10)[2],
     g = x -> [0.0, 0.0], φ = x -> -bumpder(x; x0 = (0.5, 0.5), σ = 0.10)[3]))
supnorm(f) = maximum(norm(f([i / 200, j / 200])) for i in 0:199 for j in 0:199)

# ============================================================================ E: elliptic problems on meshes 1-7
if PART == "E"
RE = Dict{Any,Any}()
for mn in MESHES7
    mesh = VoronoiMesh("meshes/$mn.nc"); S = getstars(mn, mesh); A = mesh.cells.area; h = meshsize(mesh)
    RE[(mn, :h)] = h; RE[(mn, :size)] = (cells = mesh.cells.n, edges = mesh.edges.n, tris = mesh.vertices.n)
    say(@sprintf("%s  h = 1/%d  (%d cells, %d edges, %d triangles)", mn, round(Int, 1 / h), mesh.cells.n, mesh.edges.n, mesh.vertices.n))
    pts = [_pt(x) for x in mesh.cells.position]; D0 = d0mat(mesh)
    tris, D1, _ = triangles_ccw(mesh); At = [T.area for T in tris]
    for s in STARS
        H = S[s]; r = Dict{Symbol,Any}()
        r[:asym] = norm(H - H') / norm(H); r[:Λ] = bigLambda(mesh, H); r[:g] = gersh_sym(mesh, H); r[:nnz] = nnz(H) / size(H, 1)
        r[:hd] = hodge_error(mesh, H, HFIELDS.d...); r[:hu2] = hodge_error(mesh, H, HFIELDS.u2...)
        r[:nnzL] = nnz(D0' * H * D0) / mesh.cells.n
        RE[(mn, s)] = r
        say(@sprintf("   %-5s asym %.1e  Λ %.2f  g %.3f  nnz/row H %.1f, L %.1f  r_inf(d) %.2e r_2(d) %.2e r_inf(u2) %.2e", s, r[:asym],
                     r[:Λ], r[:g], r[:nnz], r[:nnzL], r[:hd].inf, r[:hd].l2, r[:hu2].inf))
    end
    for (i, T) in enumerate(PTESTS)
        pe = [T.p(x) for x in pts]; pe .-= sum(A .* pe) / sum(A); line = "   " * rpad(T.name, 20)
        for s in STARS
            ph = solve_poisson_primal(mesh, S[s], x -> -T.lap(x)); ph .-= sum(A .* ph) / sum(A); e = ph .- pe
            RE[(mn, :P, i, s)] = (inf = maximum(abs.(e)) / maximum(abs.(pe)), l2 = sqrt(sum(A .* e .^ 2) / sum(A .* pe .^ 2)))
            mn in ("mesh1", "mesh2", "mesh4") && (RE[(mn, :Psol, i, s)] = ph; RE[(mn, :Pex, i)] = pe)
            line *= @sprintf("  %s %.2e/%.2e", s, RE[(mn, :P, i, s)].inf, RE[(mn, :P, i, s)].l2)
        end
        say(line)
    end
    for (i, T) in enumerate(DTESTS)
        ex = darcy_data(mesh, tris; v = T.v, p = T.p, g = T.g, φ = T.φ); vn = supnorm(T.v); line = "   " * rpad(T.name, 20)
        for s in STARS
            t = @elapsed ((fh, ph) = solve_darcy(S[s], D1, ex.G, ex.Φ, At))
            d = darcy_errors(mesh, tris, fh, ph, ex; pfun_ = i == 1 ? PD : nothing, vnorm = vn)
            RE[(mn, :D, i, s)] = merge(d, (time = t,))
            mn in ("mesh1", "mesh2", "mesh3", "mesh4") && (RE[(mn, :Dsol, i, s)] = (p = ph, V = triangle_velocity(mesh, tris, fh), f = fh))
            line *= @sprintf("  %s flux %.2e max %.2e p %.2e", s, d.flux, d.flux_inf, d.p_inf)
        end
        if mn in ("mesh1", "mesh2", "mesh3", "mesh4")
            RE[(mn, :Dex, i)] = (p = ex.p, V = reduce(vcat, [T.v(Tr.cc)' for Tr in tris]), f = ex.f, Vrec = triangle_velocity(mesh, tris, ex.f))
        end
        say(line)
    end
    serialize(joinpath(TSDIR, "paper_elliptic.jls"), RE)
end
function ptable(title, f)
    say("\n" * title)
    say("   star   " * join([@sprintf("%10s", m) for m in MESHES7], "") * "     orders")
    for s in STARS
        v = [f(m, s) for m in MESHES7]
        say(@sprintf("   %-6s %s     %s", s, join([@sprintf("%10.2e", x) for x in v], ""),
                     join([@sprintf("%5.2f", log2(v[i] / v[i+1])) for i in 1:length(v)-1], " ")))
    end
end
ptable("Hodge r_inf(d)", (m, s) -> RE[(m, s)][:hd].inf)
ptable("Hodge r_2(d)", (m, s) -> RE[(m, s)][:hd].l2)
ptable("Hodge r_inf(u2)", (m, s) -> RE[(m, s)][:hu2].inf)
ptable("Λ", (m, s) -> RE[(m, s)][:Λ])
ptable("g", (m, s) -> RE[(m, s)][:g])
for i in 1:3
    ptable("Poisson $(PTESTS[i].name), max", (m, s) -> RE[(m, :P, i, s)].inf)
    ptable("Poisson $(PTESTS[i].name), L2", (m, s) -> RE[(m, :P, i, s)].l2)
end
for i in 1:3
    ptable("Darcy $(DTESTS[i].name), flux H_DEC-norm", (m, s) -> RE[(m, :D, i, s)].flux)
    ptable("Darcy $(DTESTS[i].name), flux max", (m, s) -> RE[(m, :D, i, s)].flux_inf)
    ptable("Darcy $(DTESTS[i].name), p max (circumcentres)", (m, s) -> RE[(m, :D, i, s)].p_inf)
    ptable("Darcy $(DTESTS[i].name), p L2 (circumcentres)", (m, s) -> RE[(m, :D, i, s)].p_l2)
end
ptable("Darcy D1, Hirani's pressure measure", (m, s) -> RE[(m, :D, 1, s)].p_hirani)
ptable("Darcy D1, solve time (s)", (m, s) -> RE[(m, :D, 1, s)].time)
write(joinpath(TSDIR, "paper_elliptic.txt"), join(OUTP, "\n") * "\n")
end

# ============================================================================ S: spectra
"Shift-invert Arnoldi for L x = λ M x near σ: m steps with full reorthogonalisation. Returns (λ, relative residuals)."
function arnoldi_si(L, M, σ; m = 160, seed = 1)
    F = lu(L - σ * M); n = size(L, 1); rng = MersenneTwister(seed)
    V = zeros(n, m + 1); Hm = zeros(m + 1, m)
    v = randn(rng, n); V[:, 1] = v / norm(v)
    for j in 1:m
        w = F \ (M * V[:, j])
        for pass in 1:2, i in 1:j
            c = dot(view(V, :, i), w); Hm[i, j] += c; w .-= c .* view(V, :, i)
        end
        Hm[j+1, j] = norm(w); V[:, j+1] = w / Hm[j+1, j]
    end
    μ, Y = eigen(Hm[1:m, 1:m])
    λ = σ .+ 1 ./ μ; X = V[:, 1:m] * Y
    res = [norm(L * X[:, i] - λ[i] * (M * X[:, i])) / (abs(λ[i]) * norm(M * X[:, i])) for i in 1:m]
    return λ, res
end
"‖B‖₂ of a real antisymmetric operator (given by its action) by power iteration on -B²."
function antisym_norm(Bf, n; it = 300)
    x = normalize(randn(MersenneTwister(2), n)); ν = 0.0
    for _ in 1:it
        y = -Bf(Bf(x)); ν = norm(y); x = y / ν
    end
    return sqrt(ν)
end
"Statistics of the complex part of a list of eigenvalues normalised by λmax."
function cstats(λ, λmax)
    nr = findall(x -> abs(imag(x)) > 1e-9 * λmax, λ)
    return (imax = maximum(abs.(imag.(λ))) / λmax, nonreal = length(nr), frac = length(nr) / length(λ),
            band = isempty(nr) ? (NaN, NaN) : extrema(abs.(λ[nr])) ./ λmax)
end

if PART == "S"
RS = Dict{Any,Any}()
say("Spectrum of A^-1 D0' H D0 (M0 = A).  imax = max|Im λ|/λmax; β = Bendixson bound ‖A^-1/2 D0' H_a D0 A^-1/2‖₂;")
say("asym2 = ‖Hd^-1/2 H_a Hd^-1/2‖₂, asymF = ‖H_a‖_F/‖H‖_F.  Dense eigvals up to mesh 6 (H_P2), shift-invert Arnoldi on meshes 5-7.")
for mn in MESHES7
    mesh = VoronoiMesh("meshes/$mn.nc"); S = getstars(mn, mesh); A = mesh.cells.area; N = mesh.cells.n; h = meshsize(mesh)
    D0 = d0mat(mesh); Ad = spdiagm(A); hd = diag(hdiag(mesh))
    for s in (N <= 4096 ? STARS : ("H_P2",))
        H = S[s]; L = sparse(D0' * H * D0); Ha = (H - H') / 2
        r = Dict{Symbol,Any}()
        Bf(x) = (y = D0 * (x ./ sqrt.(A)); (D0' * (Ha * y)) ./ sqrt.(A))
        r[:β] = s == "H_P2" ? antisym_norm(Bf, N) : 0.0
        r[:asym2] = s == "H_P2" ? antisym_norm(x -> (Ha * (x ./ sqrt.(hd))) ./ sqrt.(hd), mesh.edges.n) : 0.0
        r[:asymF] = norm(Ha) / norm(H)
        if N <= 16384
            t = @elapsed (λ = eigvals(Matrix(Diagonal(1 ./ A) * L)))
            λmax = maximum(real.(λ)); r[:λmax] = λmax; r[:dense] = cstats(λ, λmax); r[:λ] = λ
            λs = sort(λ; by = real)[2:min(201, N)]
            r[:low] = maximum(abs.(imag.(λs)) ./ abs.(λs))
            say(@sprintf("%s %-5s dense (%.0f s): λmax %.4e  imax %.2e  nonreal %d (%.1f%%) band [%.2f, %.2f]  β/λmax %.2e  asym2 %.3f asymF %.4f  lowest 200: max|Im|/|λ| %.1e",
                         mn, s, t, λmax, r[:dense].imax, r[:dense].nonreal, 100r[:dense].frac, r[:dense].band..., r[:β] / λmax,
                         r[:asym2], r[:asymF], r[:low]))
        end
        if s == "H_P2" && N >= 4096
            λmax = get(r, :λmax, NaN)
            if isnan(λmax)                                     # largest eigenvalue: power estimate, then shift-invert above it
                x = normalize(randn(MersenneTwister(3), N)); ν = 0.0
                for _ in 1:300
                    y = (L * x) ./ A; ν = norm(y); x = y / ν
                end
                λt, rt = arnoldi_si(L, Ad, 1.2ν; m = 80)
                λmax = any(rt .< 1e-9) ? maximum(real.(λt[rt .< 1e-9])) : ν; r[:λmax] = λmax
            end
            found = ComplexF64[]
            ta = @elapsed for f in 0.10:0.05:0.90
                λa, ra = arnoldi_si(L, Ad, f * λmax; m = 120)
                append!(found, λa[ra .< 1e-9])
            end
            found = unique(x -> round(x / λmax, digits = 9), found)
            r[:arnoldi] = cstats(found, λmax); r[:arnoldi_n] = length(found); r[:found] = found
            λl, rl = arnoldi_si(L, Ad, 20.0; m = 120)
            low = λl[rl .< 1e-9]; low = low[abs.(low) .> 1.0]
            r[:low_arn] = isempty(low) ? NaN : maximum(abs.(imag.(low)) ./ abs.(low)); r[:low_n] = length(low)
            r[:low_first] = sort(real.(low))[1:min(8, length(low))]
            say(@sprintf("%s %-5s Arnoldi (%.0f s): λmax %.4e  %d eigenvalues found in [0.1, 0.9]λmax, imax %.2e, nonreal %d (%.1f%%), band [%.2f, %.2f];  β/λmax %.2e  asym2 %.3f asymF %.4f;  %d low modes: max|Im|/|λ| %.1e",
                         mn, s, ta, λmax, r[:arnoldi_n], r[:arnoldi].imax, r[:arnoldi].nonreal, 100r[:arnoldi].frac, r[:arnoldi].band...,
                         r[:β] / λmax, r[:asym2], r[:asymF], r[:low_n], r[:low_arn]))
        end
        RS[(mn, s)] = r; RS[(mn, :h)] = h
        serialize(joinpath(TSDIR, "paper_spectra.jls"), RS)
    end
end
# homogeneous family generated by SCVT mesh 3: the spectrum of M_1 reappears on M_n scaled by n^2
say("\nReplications M_n of SCVT mesh 3 (H_P2 rebuilt on each M_n)")
m3 = VoronoiMesh("meshes/mesh3.nc"); P3 = [_pt(x) for x in m3.cells.position]
λ1 = ComplexF64[]
for n in (1, 2, 3)
    xs = [(p[1] + i) / n for p in P3 for i in 0:n-1 for j in 0:n-1]
    ys = [(p[2] + j) / n for p in P3 for i in 0:n-1 for j in 0:n-1]
    Mn = delaunay_mesh(xs, ys); D0 = d0mat(Mn)
    H2, _ = rowwise_correction(Mn, hdiag(Mn), 2, 3)
    λ = eigvals(Matrix(Diagonal(1 ./ Mn.cells.area) * (D0' * H2 * D0))); λmax = maximum(real.(λ))
    st = cstats(λ, λmax)
    if n == 1
        global λ1 = λ[abs.(imag.(λ)) .> 1e-9 * λmax]
        dist = NaN
    else
        dist = maximum(minimum(abs.(λ .- n^2 * μ)) / abs(n^2 * μ) for μ in λ1)
    end
    RS[(:rep, n)] = (cells = Mn.cells.n, st = st, λmax = λmax, dist = dist, asymF = norm(H2 - H2') / norm(H2) / 2)
    say(@sprintf("   n = %d: %5d cells  λmax %.4e  max|Im λ|/λmax %.3e  nonreal %.1f%%  max over complex λ of M_1 of dist(n²λ, spec M_n)/|n²λ| %.1e",
                 n, Mn.cells.n, λmax, st.imax, 100st.frac, dist))
end
serialize(joinpath(TSDIR, "paper_spectra.jls"), RS)
write(joinpath(TSDIR, "paper_spectra.txt"), join(OUTP, "\n") * "\n")
end

# ============================================================================ T: costs
function pcg(Af, b, Minv; tol = 1e-10, maxit = 50000)
    x = zero(b); r = copy(b); z = Minv .* r; p = copy(z); rz = dot(r, z); nb = norm(b)
    for it in 1:maxit
        Ap = Af(p); α = rz / dot(p, Ap); x .+= α .* p; r .-= α .* Ap
        norm(r) <= tol * nb && return x, it
        z = Minv .* r; rz2 = dot(r, z); p .= z .+ (rz2 / rz) .* p; rz = rz2
    end
    return x, maxit
end
function bicgstab(Af, b, Minv; tol = 1e-10, maxit = 50000)
    x = zero(b); r = copy(b); rh = copy(r); ρ = α = ω = 1.0; v = zero(b); p = zero(b); nb = norm(b)
    for it in 1:maxit
        ρn = dot(rh, r); β = (ρn / ρ) * (α / ω); ρ = ρn
        p .= r .+ β .* (p .- ω .* v)
        y = Minv .* p; v = Af(y); α = ρ / dot(rh, v)
        s = r .- α .* v
        if norm(s) <= tol * nb
            x .+= α .* y; return x, it
        end
        z = Minv .* s; t = Af(z); ω = dot(t, s) / dot(t, t)
        x .+= α .* y .+ ω .* z; r = s .- ω .* t
        norm(r) <= tol * nb && return x, it
    end
    return x, maxit
end

if PART == "T"
RT = Dict{Any,Any}()
say("Construction cost (s, single run after warm-up on mesh1) and the size of the global system of H_P1")
for mn in MESHES7
    mesh = VoronoiMesh("meshes/$mn.nc"); Hd = hdiag(mesh)
    t1 = @elapsed ((H1, info) = symmetric_correction(mesh, 1, 3))
    t2 = @elapsed ((H2, _) = rowwise_correction(mesh, Hd, 2, 3))
    RT[(mn, :build)] = (p1 = t1, p2 = t2, rows = info.rows, unknowns = info.unknowns, edges = mesh.edges.n)
    say(@sprintf("   %s  %6d edges   H_P1 %8.2f s (global system %d x %d)   H_P2 %8.2f s", mn, mesh.edges.n, t1, info.rows,
                 info.unknowns, t2))
    # solves of the Poisson problem P1 with each star
    S = Dict("H_DEC" => Hd, "H_P1" => H1, "H_P2" => H2); D0 = d0mat(mesh); n = mesh.cells.n
    b = discretize_scalar_field_cells(mesh, x -> -lapp(x)).values; b .-= sum(b) / n
    keep = 2:n
    for s in STARS
        L = sparse(D0' * S[s] * D0); Lk = L[keep, keep]
        tl = @elapsed lu(Lk)
        tc = s == "H_P2" ? NaN : (@elapsed cholesky(Symmetric(Lk)))
        Minv = 1 ./ diag(L)
        tcg, icg = s == "H_P2" ? (NaN, 0) : (t = @elapsed ((_, it) = pcg(x -> L * x, b, Minv)); (t, it))
        tbi = @elapsed ((_, ibi) = bicgstab(x -> L * x, b, Minv))
        RT[(mn, :solve, s)] = (lu = tl, chol = tc, cg = icg, cg_t = tcg, bicg = ibi, bicg_t = tbi, nnzL = nnz(L) / n)
        say(@sprintf("      %-5s L: %5.1f nnz/row   LU %7.3f s  Cholesky %7.3f s   PCG %5d it (%6.2f s)   BiCGStab %5d it (%6.2f s)",
                     s, nnz(L) / n, tl, tc, icg, tcg, ibi, tbi))
    end
    serialize(joinpath(TSDIR, "paper_costs.jls"), RT)
end
write(joinpath(TSDIR, "paper_costs.txt"), join(OUTP, "\n") * "\n")
end

# ============================================================================ Y: the symmetric [P2]^2 star on the 4-ring, mesh 6
function lambda_min_sym(mesh, H)
    Dm = Diagonal(1 ./ sqrt.(diag(hdiag(mesh)))); S = sparse(Symmetric(Dm * ((H + H') / 2) * Dm))
    spd(t) = issuccess(cholesky(S - t * I; check = false))
    lo = -1.0; while spd(lo) == false && lo > -1e7; lo *= 4; end
    hi = 1.5
    while hi - lo > 1e-3 * max(1.0, abs(lo))
        t = (lo + hi) / 2; spd(t) ? (lo = t) : (hi = t)
    end
    return lo
end
if PART == "Y"
RY = Dict{Any,Any}()
for mn in split(get(ENV, "YMESH", "mesh6"), ",")
    mesh = VoronoiMesh("meshes/$mn.nc")
    t = @elapsed ((H, info) = symmetric_correction(mesh, 2, 4))
    hd = hodge_error(mesh, H, HFIELDS.d...); hu = hodge_error(mesh, H, HFIELDS.u2...)
    P = poisson_errors(mesh, H)
    μ = lambda_min_sym(mesh, H)
    RY[mn] = (time = t, res = info.res, Λ = bigLambda(mesh, H), g = gersh_sym(mesh, H), μ = μ, hd = hd, hu = hu, poisson = P.inf,
              per_row = info.per_row, unknowns = info.unknowns)
    say(@sprintf("%s sym [P2]^2 4-ring (%.0f s): res %.1e  Λ %.2f  g %.2f  μ %.3f  r_2(d) %.2e  r_inf(u2)+r_inf(d) %.2e  Poisson max %.2e  %.1f entries/row",
                 mn, t, info.res, RY[mn].Λ, RY[mn].g, μ, hd.l2, hu.inf + hd.inf, P.inf, info.per_row))
    serialize(joinpath(TSDIR, "paper_symp2.jls"), RY)
end
write(joinpath(TSDIR, "paper_symp2.txt"), join(OUTP, "\n") * "\n")
end
