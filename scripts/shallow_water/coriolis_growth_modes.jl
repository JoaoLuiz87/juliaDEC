# ============================================================================
#  coriolis_growth_modes.jl -- the growing modes of the linear shallow-water scheme with the consistent Coriolis operator P.
#  (1) check of the plane-wave formula of k(u,w) on a lattice, against the direct sum of coriolis_barrier_check.jl:
#        u = a cos(κ·x), w = b cos(κ·x):   k = ½ Σ_e |e||ẽ| sinc(κ·n|ẽ|/2) sinc(κ·t|e|/2) [ (a·t)(Jb·t) + (b·t)(Ja·t) ]   (t ⟂ n, ẽ centred on e)
#  (2) scale of the fastest growing mode (dense, meshes 3 and 4)
#  (3) the same growth rate on meshes 5 and 6: shift-invert Arnoldi on  K x = λ B x,  B = diag(M0, I),  around slow shifts
#
#      julia -t 4 scripts/shallow_water/coriolis_growth_modes.jl      -> output/ts_data/coriolis_growth.txt
# ============================================================================
include(joinpath(@__DIR__, "sw_linear_three.jl"))
OUTC = String[]; sc(s) = (println(s); flush(stdout); push!(OUTC, s))
sinc1(x) = abs(x) < 1e-8 ? 1.0 : sin(x) / x
rot(a) = Jr(a)
sc("(1) plane-wave formula of k on lattices, against the direct sum (|Ω| = 1, divided by h²)")
for mn in ("mesh3_regular", "mesh5_regular", "mesh6_regular")
    mesh = VoronoiMesh("meshes/$mn.nc"); N = mesh.cells.n; h2 = 1 / N
    sc(@sprintf("   %s  N %d  periods %.4f × %.4f", mn, N, mesh.x_period, mesh.y_period))
    for (a, b, m) in (([1.0, 0.0], [1.0, 0.0], (2, -1)), ([1.0, 0.0], [0.0, 1.0], (1, 0)), ([1.0, 0.0], [0.0, 1.0], (0, 1)), ([0.6, 0.8], [-0.8, 0.6], (1, 1)))
        κv = 2π .* [m[1] / mesh.x_period, m[2] / mesh.y_period]; u(x) = a .* cos(dot(κv, x)); w(x) = b .* cos(dot(κv, x))
        kd = 0.0; kf = 0.0
        for e in 1:mesh.edges.n
            p1, p2 = edge_endpoints(mesh, e, true); p1 = _p2(p1); p2 = _p2(p2); t = collect((p2 .- p1) ./ norm(p2 .- p1)); n = [-t[2], t[1]]
            q1, q2 = edge_endpoints(mesh, e, false); q1 = _p2(q1); q2 = _p2(q2)
            Ru = seg_int(x -> u(x), q1, q2, t); Rw = seg_int(x -> w(x), q1, q2, t)
            Cu = seg_int(x -> rot(u(x)), p1, p2, t); Cw = seg_int(x -> rot(w(x)), p1, p2, t)
            kd += Ru * Cw + Rw * Cu
            le, ld = norm(p2 .- p1), norm(q2 .- q1)
            kf += 0.5 * le * ld * sinc1(dot(κv, n) * ld / 2) * sinc1(dot(κv, t) * le / 2) * (dot(a, t) * dot(rot(b), t) + dot(b, t) * dot(rot(a), t))
        end
        sc(@sprintf("      a=%s b=%s κ=2π%s   direct %+.6e   formula %+.6e   k/h² %+.4f", a, b, m, kd, kf, kd / h2))
    end
end
sc("\n(2) the fastest growing mode: scale.  λ_η = 2π/√(ηᴴD0ᵀHD0η/ηᴴAη)/h and λ_ζ = 2π/√(Σζ²A_tri/|v|²_H)/h, wavelengths in units of h (grid scale ≈ 2)")
function blocks(O, M, C)
    nc, ne = O.nc, O.ne
    K = [spzeros(nc, nc) H0J .* (O.D0t * O.H); -GJ .* O.D0 -FJ .* C]
    B = [sparse(M) spzeros(nc, ne); spzeros(ne, nc) sparse(1.0I, ne, ne)]
    return sparse(K), sparse(B)
end
function arnoldi_si(K, B, σ; m = 60)
    F = lu(ComplexF64.(K) .- σ .* ComplexF64.(B)); n = size(K, 1)
    V = zeros(ComplexF64, n, m + 1); Hm = zeros(ComplexF64, m + 1, m); r = randn(ComplexF64, n); V[:, 1] = r ./ norm(r)
    for j in 1:m
        y = F \ (B * V[:, j])
        for _ in 1:2, i in 1:j; c = dot(V[:, i], y); Hm[i, j] += c; y .-= c .* V[:, i]; end
        Hm[j+1, j] = norm(y); V[:, j+1] = y ./ Hm[j+1, j]
    end
    E = eigen(Hm[1:m, 1:m]); λ = σ .+ 1 ./ E.values
    res = [abs(Hm[m+1, m] * E.vectors[m, i]) / abs(E.values[i]) for i in 1:m]
    return λ, res, V[:, 1:m] * E.vectors
end
function scales(O, mesh, x)
    h = O.L / sqrt(O.nc); η = x[1:O.nc]; v = x[O.nc+1:end]; HD = spdiagm(0 => [O.HD[e] for e in 1:O.ne])
    g = O.D0 * η; kη = sqrt(real(dot(g, HD * g)) / real(dot(η, O.A .* η)))
    ke = real(dot(v, HD * v))
    ζ = O.D1 * v; kζ = sqrt(real(sum(abs2.(ζ) ./ O.AT)) / ke); dv = O.D0t * (HD * v); kδ = sqrt(real(sum(abs2.(dv) ./ O.A)) / ke)
    pe = GJ * real(dot(η, O.A .* η)); kin = H0J * ke
    return 2π / kη / h, 2π / kζ / h, kin / (kin + pe), kδ / kζ
end
for mn in ("mesh3", "mesh4", "mesh5", "mesh6")
    mesh = VoronoiMesh("meshes/$mn.nc"); S = cached_stars(mn; mesh); O0 = sw_ops(mesh, S["H_DEC"]); P = perp_p2(mesh, O0.W * O0.H)
    M0c = mass0_corrected(mesh) .* O0.L^2; AT = O0.AT; D1 = O0.D1
    for (st, Mlab, M) in (("H_DEC", "A", spdiagm(0 => O0.A)), ("H_P1", "M0c", M0c), ("H_P2", "A", spdiagm(0 => O0.A)), ("H_P2", "M0c", M0c))
        O = sw_ops(mesh, S[st]); Ox = (; nc = O.nc, ne = O.ne, L = O.L, D0 = O.D0, D0t = O.D0t, A = O.A, D1 = D1, AT = AT, HD = Vector(diag(S["H_DEC"]))); K, B = blocks(O, M, P)
        best = (-Inf, 0.0im, nothing, 0.0)
        shifts = mn in ("mesh3", "mesh4", "mesh5") ? [complex(6e-6, s * FJ) for s in (0.0, 0.11, 0.25, 0.5)] : [complex(6e-6, s * FJ) for s in (0.05, 0.11)]
        for σ in shifts
            λ, res, X = arnoldi_si(K, B, σ)
            for i in eachindex(λ)
                res[i] < 1e-6 && real(λ[i]) > best[1] && (best = (real(λ[i]), λ[i], X[:, i], res[i]))
            end
        end
        best[3] === nothing && (sc("$mn $st $Mlab: nothing converged"); continue)
        a, b, c, d = scales(Ox, mesh, best[3])
        sc(@sprintf("   %s %-5s P, M0 = %-4s max Re λ (slow shifts) %+.3e s⁻¹ (%.2f days)  |Im|/f %.3f  wavelength: η %.1f h, vorticity %.1f h; kinetic fraction %.2f; div/curl %.2f  (res %.0e)",
                    mn, st, Mlab, best[1], 1 / best[1] / 86400, abs(imag(best[2])) / FJ, a, b, c, d, best[4]))
    end
end
write(joinpath(TSDIR, "coriolis_growth.txt"), join(OUTC, "\n") * "\n")
