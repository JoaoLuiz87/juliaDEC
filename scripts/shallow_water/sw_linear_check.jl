# ============================================================================
#  sw_linear_check.jl -- the non-rotating linear shallow-water test (open point "TRiSK" of commuting_symmetric_note.tex):
#  does the non-symmetric [P2]^2 star of nonsymmetric_check.jl pay off in a wave problem, and what does its
#  antisymmetric part cost?
#
#      julia -t 4 scripts/shallow_water/sw_linear_check.jl      (writes output/gs_data/sw_linear_check.txt)
#
#  Unknowns: h at the generators (point values) and v = circulations along the Delaunay edges (TC12's prognostic):
#      M0 dh/dt = H0 d0' H v       (mass of each Voronoi cell; d0'F = minus the outward flux sum, F = H v)
#         dv/dt = -g d0 h          (exact: the circulation of grad h along e)
#  with g = H0 = 1.  M0 = diag(cell areas) is TRiSK's 0-form star (point values -> cell integrals, exact on P1 only).
#  The only approximations are H (circulations -> fluxes) and M0.  The energy E = g/2 h'Ah + H0/2 v'sym(H)v is
#  conserved exactly when H = H' and M0 = A.  Waves have frequencies ±√λ, λ an eigenvalue of A^-1 d0'Hd0; when
#  Im √λ ≠ 0 one of the pair grows like exp(|Im √λ| t).
# ============================================================================
ENV["NS_MAIN"] = "0"                                # load the functions of nonsymmetric_check.jl only
include(joinpath(@__DIR__, "..", "stars", "nonsymmetric_check.jl"))

const GRAV = 1.0; const DEPTH = 1.0
const MODES = ((1, 1, 1.0, 0.0), (2, -1, 0.5, 0.7), (0, 1, 0.3, 1.9))     # (m, n, amplitude, phase)
const AMP = sum(m[3] for m in MODES)                                      # bound on |h| and on |u| (g/c = 1)

"Exact travelling-wave solution h(x,t), u(x,t) = sum of a cos(k.x - ωt + φ), u = (g a/ω) k cos(...), ω = c|k|."
function wave(mesh)
    ks = [(2π * m / mesh.x_period, 2π * n / mesh.y_period, a, φ) for (m, n, a, φ) in MODES]
    c = sqrt(GRAV * DEPTH)
    hf(x, t) = sum(a * cos(kx * x[1] + ky * x[2] - c * hypot(kx, ky) * t + φ) for (kx, ky, a, φ) in ks)
    uf(x, t) = sum((GRAV * a / (c * hypot(kx, ky))) * cos(kx * x[1] + ky * x[2] - c * hypot(kx, ky) * t + φ) .* [kx, ky]
                   for (kx, ky, a, φ) in ks)
    return hf, uf
end

stars(mesh) = (Hd = hdiag(mesh);
               [("DEC", Hd), ("sym P1", symmetric_correction(mesh, 1, 3)[1]), ("ns P2", rowwise_correction(mesh, Hd, 2, 3)[1])])

"Point values at the generators -> integrals over the Voronoi cells: diag(A) plus the smallest row-wise correction on
the 2-ring of cells that makes it exact on P_k (k = 3: 10 conditions on about 19 cells).  Not symmetric."
function mass0_corrected(mesh; k = 3)
    nc = mesh.cells.n; A = mesh.cells.area; Lx, Ly = mesh.x_period, mesh.y_period
    nbr = [Set{Int}() for _ in 1:nc]
    for e in 1:mesh.edges.n
        c1, c2 = mesh.edges.cells[e]; push!(nbr[c1], c2); push!(nbr[c2], c1)
    end
    mons = [(i, d - i) for d in 0:k for i in d:-1:0]
    bq, wq = tri_rule(5)
    Ii, Jj, Vv = Int[], Int[], Float64[]
    for c in 1:nc
        x0 = _pt(mesh.cells.position[c]); s = sqrt(A[c])
        r1 = union(Set([c]), nbr[c]); S = sort(collect(union(r1, (nbr[j] for j in r1)...)))
        mono(p) = [((p[1] - x0[1]) / s)^α[1] * ((p[2] - x0[2]) / s)^α[2] for α in mons]
        b = zeros(length(mons))
        for (q1, q2, q3) in voronoi_cell_triangles(mesh, c)
            Ar = _tri_area(q1, q2, q3)
            for (bb, ww) in zip(bq, wq)
                b .+= (ww * Ar) .* mono(bb[1] .* q1 .+ bb[2] .* q2 .+ bb[3] .* q3)
            end
        end
        Q = reduce(hcat, [mono(_pt(closest(mesh.cells.position[c], mesh.cells.position[j], Lx, Ly))) for j in S])
        b .-= A[c] .* Q[:, findfirst(==(c), S)]
        append!(Ii, fill(c, length(S))); append!(Jj, S); append!(Vv, pinv(Q) * b)
        push!(Ii, c); push!(Jj, c); push!(Vv, A[c])
    end
    return sparse(Ii, Jj, Vv, nc, nc)               # the two (c,c) entries are summed
end

"x' = Mb x for x = [h; v] with M0 = diag(A)."
function sw_block(mesh, H)
    d0 = 1.0 * d0_matrix_primal(mesh); A = mesh.cells.area; nc, ne = mesh.cells.n, mesh.edges.n
    return [spzeros(nc, nc) DEPTH .* (Diagonal(1 ./ A) * d0' * H); -GRAV .* d0 spzeros(ne, ne)]
end

energy(mesh, Hs, x) = (nc = mesh.cells.n; h = x[1:nc]; v = x[nc+1:end];
                       GRAV / 2 * dot(h, mesh.cells.area .* h) + DEPTH / 2 * dot(v, Hs * v))

function rk4(f, x, dt, nsteps)
    for _ in 1:nsteps
        k1 = f(x); k2 = f(x .+ (dt / 2) .* k1); k3 = f(x .+ (dt / 2) .* k2); k4 = f(x .+ dt .* k3)
        x = x .+ (dt / 6) .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4)
    end
    return x
end

"Implicit midpoint: neutral, |amplification| = 1 exactly for imaginary eigenvalues, > 1 exactly for growing ones."
function midpoint_run(Mb, x, dt, nsteps, every, E)
    F = lu(I - (dt / 2) * Mb); B = I + (dt / 2) * Mb
    x = Vector{Float64}(x); E0 = E(x); dev = 0.0
    for n in 1:nsteps
        x = F \ (B * x)
        n % every == 0 && (dev = max(dev, abs(E(x) / E0 - 1)))
    end
    return x, E(x) / E0 - 1, dev
end

"λ of M0^-1 d0'Hd0 (M0 = A by default), the frequencies ω = |Re √λ|, growth rates σ = |Im √λ|, and ω_max = √max|λ|."
function wave_spectrum(mesh, H; M0 = nothing)
    d0 = d0_matrix_primal(mesh); L = Matrix(d0' * H * d0)
    λ = M0 === nothing ? eigvals(Diagonal(1 ./ mesh.cells.area) * L) : eigvals(Matrix(M0) \ L)
    r = sqrt.(complex.(λ))
    return λ, abs.(real.(r)), abs.(imag.(r)), sqrt(maximum(abs.(λ)))
end

"β = ||A^-1/2 d0' skew(H) d0 A^-1/2||_2, the Bendixson bound on |Im λ| of A^-1 d0'Hd0."
function skew_bound(mesh, H)
    d0 = d0_matrix_primal(mesh); Ds = Diagonal(1 ./ sqrt.(mesh.cells.area))
    return opnorm(Ds * Matrix(d0' * ((H - H') / 2) * d0) * Ds)
end

R4(z) = 1 + z + z^2 / 2 + z^3 / 6 + z^4 / 24
"Largest RK4 amplification over the wave eigenvalues ±i√λ at the step dt."
rk4_amp(λ, dt) = maximum(max(abs(R4(dt * im * s)), abs(R4(-dt * im * s))) for s in sqrt.(complex.(λ)))

# ============================================================================
if get(ENV, "SW_MAIN", "1") == "1"                  # SW_MAIN=0 only loads the functions (used by sw_swept_check.jl)
logp("Linear shallow water, f = 0, g = H0 = 1, on the SCVT meshes. Stars: DEC, the symmetric [P1]^2 star (3-ring),")
logp("and DEC + the smallest row-wise correction exact on [P2]^2 (ns P2, not symmetric).")

logp("\nPart 1: spectrum of the wave operator. sigma = |Im √λ| is the growth rate of a wave of frequency omega = |Re √λ|")
logp("   sig/om_max = largest growth rate over the fastest frequency;  at om = frequency of that mode / om_max")
logp("   resolved: largest sigma/omega over waves with omega < om_max/4, and the e-folding time 1/(2π sigma/omega) in their own periods")
logp("   RK4: largest amplification - 1 over all waves, at om_max*dt = 0.1 / 0.5 / 1 / 1.5 / 2 / 2.5 (> 0: some wave grows)")
logp("   Im: max|Im λ|/max|λ| ;  beta: Bendixson bound ||A^-1/2 d0' skew(H) d0 A^-1/2|| / max|λ| (M0 = A only)")
logp("   rows '+C0' use the corrected 0-form star M0 = A + C0 of Part 3 (non-symmetric) instead of A")
spec = Dict{Any,Any}()
for mn in ("mesh3", "mesh4", "mesh5")
    mesh = VoronoiMesh("meshes/$mn.nc")
    logp(@sprintf("%s  (%d cells, %d edges)", mn, mesh.cells.n, mesh.edges.n))
    M0c = mass0_corrected(mesh)
    st = stars(mesh)
    for (name, H, M0) in vcat([(n_, H_, nothing) for (n_, H_) in st], [(n_ * "+C0", H_, M0c) for (n_, H_) in st if n_ != "sym P1"])
        λ, ω, σ, ωm = wave_spectrum(mesh, H; M0 = M0)
        spec[(mn, name)] = (λ = λ, ωm = ωm)
        i = argmax(σ)
        res_ = [σ[j] / ω[j] for j in eachindex(λ) if 1e-8 * ωm < ω[j] < ωm / 4]
        rmax = maximum(res_)
        amps = [rk4_amp(λ, c / ωm) - 1 for c in (0.1, 0.5, 1.0, 1.5, 2.0, 2.5)]
        bnd = M0 === nothing ? @sprintf("beta %.1e", skew_bound(mesh, H) / ωm^2) : "             "
        logp(@sprintf("   %-10s om_max %6.1f  sig/om_max %.1e at om %.2f  Im %.1e  %s  resolved: sig/om %.1e (e-fold %s periods)  RK4: %s",
                      name, ωm, σ[i] / ωm, ω[i] / ωm, maximum(abs.(imag.(λ))) / ωm^2, bnd, rmax,
                      rmax > 0 ? @sprintf("%.0f", 1 / (2π * rmax)) : "inf", join([@sprintf("%+.0e", a) for a in amps], " ")))
    end
end

logp("\nPart 2: energy. Implicit midpoint (neutral in time), so every change of E comes from the space discretisation")
logp("   smooth: the travelling waves, dt = 0.01 up to t = 20 (about 28 periods of the longest wave); max_t |E/E0 - 1|")
logp("   noise : random h and v, dt = 0.25/om_max for 500 periods of the fastest wave; E(T)/E0 against exp(2 sigma_max T)")
logp("           and the same noise with RK4 at om_max*dt = 1.5")
for mn in ("mesh3", "mesh4", "mesh5")
    mesh = VoronoiMesh("meshes/$mn.nc"); nc = mesh.cells.n
    hf, uf = wave(mesh)
    x0 = Vector{Float64}(vcat([hf(_pt(p), 0.0) for p in mesh.cells.position], reduce_primal_edges(mesh, x -> uf(x, 0.0)).values))
    logp(mn)
    for (name, H) in stars(mesh)
        Mb = sw_block(mesh, H); Hs = (H + H') / 2; E(x) = energy(mesh, Hs, x)
        _, dE, dev = midpoint_run(Mb, x0, 0.01, 2000, 20, E)
        line = @sprintf("   %-7s smooth: max|E/E0-1| %.1e (at t=20: %+.1e)", name, dev, dE)
        if mn != "mesh5"
            ωm = spec[(mn, name)].ωm; σm = maximum(abs.(imag.(sqrt.(complex.(spec[(mn, name)].λ)))))
            T = 500 * 2π / ωm
            Random.seed!(3); xn = vcat(randn(nc), randn(mesh.edges.n) .* sqrt.(mesh.cells.area[1] ./ diag(Hs)))
            dt = 0.25 / ωm; ns = ceil(Int, T / dt)
            _, dEn, _ = midpoint_run(Mb, xn, T / ns, ns, ns, E)
            nr = ceil(Int, T / (1.5 / ωm))
            xr = rk4(x -> Mb * x, xn, T / nr, nr)
            line *= @sprintf("   noise: E(T)/E0 %.3e (bound exp(2 sig T) = %.3e)   RK4 1.5: E(T)/E0 %.3e",
                             1 + dEn, exp(2σm * T), E(xr) / E(xn))
        end
        logp(line)
    end
end

logp("\nPart 3: accuracy at t = 1 (RK4, dt = 1e-3). e(h) = max|h - h_exact|/max, e(v) = max_e |v - R_h u|/(|e| max|u|)")
logp("   M0 = A      : TRiSK's 0-form star (exact on P1 only); dE = E(1)/E(0) - 1")
logp("   M0 = A + C0 : point values -> cell integrals exact on P3 (2-ring, not symmetric)")
acc = Dict{Any,Any}()
for mn in ("mesh2", "mesh3", "mesh4", "mesh5")
    mesh = VoronoiMesh("meshes/$mn.nc"); nc = mesh.cells.n
    hf, uf = wave(mesh)
    pts = [_pt(p) for p in mesh.cells.position]
    x0 = Vector{Float64}(vcat(hf.(pts, 0.0), reduce_primal_edges(mesh, x -> uf(x, 0.0)).values))
    hT = hf.(pts, 1.0); vT = Vector{Float64}(reduce_primal_edges(mesh, x -> uf(x, 1.0)).values)
    lp = edge_lengths(mesh, true)
    d0 = 1.0 * d0_matrix_primal(mesh)
    M0c = mass0_corrected(mesh)
    hint = discretize_scalar_field_cells(mesh, x -> hf(x, 0.0)).values
    m0res(M) = maximum(abs.(M * x0[1:nc] .- hint) ./ mesh.cells.area) / AMP
    logp(@sprintf("%s  (h = %.4f)   0-form star defect max|M0 h - ∫h|/|cell|: A %.1e   A + C0 %.1e",
                  mn, sqrt(mesh.x_period * mesh.y_period / nc), m0res(Diagonal(mesh.cells.area)), m0res(M0c)))
    F0c = lu(M0c)
    for (name, H) in stars(mesh)
        nsteps = 1000; dt = 1.0 / nsteps
        Mb = sw_block(mesh, H); E(x) = energy(mesh, (H + H') / 2, x)
        xa = rk4(x -> Mb * x, x0, dt, nsteps)
        fc(x) = vcat(DEPTH .* (F0c \ (d0' * (H * x[nc+1:end]))), -GRAV .* (d0 * x[1:nc]))
        xc = rk4(fc, x0, dt, nsteps)
        err(x) = (maximum(abs.(x[1:nc] .- hT)) / AMP, maximum(abs.(x[nc+1:end] .- vT) ./ lp) / AMP)
        ea, ec = err(xa), err(xc)
        acc[(mn, name)] = (ea = ea, ec = ec)
        logp(@sprintf("   %-7s M0 = A: e(h) %.2e  e(v) %.2e  dE %+.1e     M0 = A + C0: e(h) %.2e  e(v) %.2e",
                      name, ea..., E(xa) / E(x0) - 1, ec...))
    end
end
logp("\n   observed orders (mesh2->3->4->5) of e(h):")
for name in ("DEC", "sym P1", "ns P2")
    ms = ("mesh2", "mesh3", "mesh4", "mesh5")
    o(f) = join([@sprintf("%5.2f", log2(f(acc[(ms[i], name)]) / f(acc[(ms[i+1], name)]))) for i in 1:3], " ")
    logp(@sprintf("   %-7s M0 = A: %s     M0 = A + C0: %s", name, o(a -> a.ea[1]), o(a -> a.ec[1])))
end
write(joinpath(OUT, "sw_linear_check.txt"), join(lines, "\n") * "\n")
end   # SW_MAIN
