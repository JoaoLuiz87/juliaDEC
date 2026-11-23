# ============================================================================
#  sw_linear_rot.jl -- the linear tests of Peixoto, Alves, Fumagalli & Antonietti (PolyDG paper, §5.2) with H_DEC and H_P2
#
#      julia -t 4 scripts/shallow_water/sw_linear_rot.jl      ->  output/ts_data/sw_linear_rot.{txt,jls}
#
#  Linear rotating shallow water about a state of rest of depth H (f-plane), Earth-scale square [0,L]², L = 2π·6371.22 km:
#      ∂t η + H div u = 0,      ∂t u + f J u + g ∇η = 0.
#  Discretisation (TRiSK in DEC variables; η at the generators, circulations v on the Delaunay edges):
#      M0 η̇ = H D0ᵀ (Hstar v),      v̇ = −f W (Hstar v) − g D0 η
#  M0 = diag(A) (TRiSK) or the corrected 0-form star (exact on cubics, sw_linear_check.jl). D0 η is exact for point values.
#  Energy  E = ½ g Σ A η² + ½ H vᵀ H_s v  (H_s the symmetric part): conserved by the semi-discrete system if Hstar is symmetric,
#  M0 = diag(A) and W antisymmetric (the Coriolis term does no work); the implicit midpoint rule keeps it to round-off.
#
#  Tests (k = 2π·3/L, ω² = f² + gHk²):
#    plane wave with rotation (geostrophic adjustment): rest, η = η0 (sin kx + sin ky), η0 = 1 m, T = 1.5 days
#    Poincaré wave: two inertia-gravity waves (along x and y), velocity amplitude 1 m/s, η̂ = u0 H k/ω ≈ 22.7 m, T = 2π/ω
#    geostrophic balance: ψ = (u0/k1) sin k1x sin k1y, k1 = 2π/L, u = J∇ψ, η = (f/g)ψ (≈ 95 m), T = 1 day
# ============================================================================
include(joinpath(@__DIR__, "sw_jet.jl"))
ENV["SW_MAIN"] = "0"; include(joinpath(@__DIR__, "sw_linear_check.jl"))
using SparseArrays, LinearAlgebra, Printf, Serialization

const KW = 2π * 3 / LDOM
const OMW = sqrt(FJ^2 + GJ * H0J * KW^2)
const K1 = 2π / LDOM
Jr(a) = [-a[2], a[1]]

"Exact solutions: (η(x,t), u(x,t)) for the three tests."
function exact_lin(test)
    k, ω, f, g, H = KW, OMW, FJ, GJ, H0J
    dirs = ([1.0, 0.0], [0.0, 1.0])
    if test == :plane
        η0 = 1.0
        η(x, t) = sum(η0 * (f^2 / ω^2 + (1 - f^2 / ω^2) * cos(ω * t)) * sin(k * dot(n, x)) for n in dirs)
        u(x, t) = sum((-(g * k * η0 / ω) * sin(ω * t) * cos(k * dot(n, x))) .* n .+
                      ((g * k * f * η0 / ω^2) * (1 - cos(ω * t)) * cos(k * dot(n, x))) .* Jr(n) for n in dirs)
        return η, u, 1.5DAY
    elseif test == :poincare
        u0 = 1.0; ηh = u0 * H * k / ω
        ηp(x, t) = sum(ηh * cos(k * dot(n, x) - ω * t) for n in dirs)
        up(x, t) = sum((u0 * cos(k * dot(n, x) - ω * t)) .* n .+ ((f * ηh / (H * k)) * sin(k * dot(n, x) - ω * t)) .* Jr(n) for n in dirs)
        return ηp, up, 2π / ω
    else
        u0 = 1.0; ψ0 = u0 / K1
        ηb(x, t) = (f / g) * ψ0 * sin(K1 * x[1]) * sin(K1 * x[2])
        ub(x, t) = u0 .* [-sin(K1 * x[1]) * cos(K1 * x[2]), cos(K1 * x[1]) * sin(K1 * x[2])]
        return ηb, ub, 1.0DAY
    end
end

const GL8L = ((-0.9602898564975363, 0.1012285362903763), (-0.7966664774136267, 0.2223810344533745), (-0.5255324099163290, 0.3137066458778873),
              (-0.1834346424956498, 0.3626837833783620), (0.1834346424956498, 0.3626837833783620), (0.5255324099163290, 0.3137066458778873),
              (0.7966664774136267, 0.2223810344533745), (0.9602898564975363, 0.1012285362903763))
"∫ along the segment p1 → p2 (metres) of w(x)·d, d a fixed unit vector."
seg_int(w, p1, p2, d) = sum(gw / 2 * dot(w(p1 .+ (ξ + 1) / 2 .* (p2 .- p1)), d) for (ξ, gw) in GL8L) * norm(p2 .- p1)
function cochains(O, η, u, t)
    L = O.L; mesh = O.mesh
    h = [η(x, t) for x in O.xc]
    v = zeros(O.ne); fl = zeros(O.ne); cJ = zeros(O.ne)
    for e in 1:O.ne
        p1, p2 = edge_endpoints(mesh, e, true); p1 = _p2(p1) .* L; p2 = _p2(p2) .* L; te = (p2 .- p1) ./ norm(p2 .- p1)
        q1, q2 = edge_endpoints(mesh, e, false); q1 = _p2(q1) .* L; q2 = _p2(q2) .* L
        v[e] = seg_int(x -> u(x, t), p1, p2, te)                    # circulation along e
        fl[e] = seg_int(x -> u(x, t), q1, q2, te)                   # exact volume flux across ẽ (direction t_e), R̃⋆u
        cJ[e] = seg_int(x -> Jr(u(x, t)), p1, p2, te)               # circulation of J u (exact Coriolis direction)
    end
    return h, v, fl, cJ
end

"The linear operator for a star and a 0-form star; returns (rhs!, L_sparse or nothing)."
function lin_ops(O, M0kind)
    nc, ne = O.nc, O.ne
    if M0kind == :A
        Minv = spdiagm(0 => 1 ./ O.A)
        Lmat = [spzeros(nc, nc) H0J .* Minv * O.D0t * O.H; -GJ .* O.D0 -FJ .* O.W * O.H]
        return x -> Lmat * x, Lmat, nothing
    else
        M0c = lu(mass0_corrected(O.mesh) .* O.L^2)
        f = x -> (v = view(x, nc+1:nc+ne); U = O.H * v; vcat(H0J .* (M0c \ (O.D0t * U)), -FJ .* (O.W * U) .- GJ .* (O.D0 * x[1:nc])))
        return f, nothing, M0c
    end
end
energy_lin(O, x, Hs) = (h = x[1:O.nc]; v = x[O.nc+1:end]; 0.5GJ * sum(O.A .* h .^ 2) + 0.5H0J * dot(v, Hs * v))
function rk4_lin(f, x, dt, n; every = 0, cb = nothing)
    for k in 1:n
        k1 = f(x); k2 = f(x .+ (dt / 2) .* k1); k3 = f(x .+ (dt / 2) .* k2); k4 = f(x .+ dt .* k3)
        x = x .+ (dt / 6) .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4)
        every > 0 && k % every == 0 && cb(k, x)
    end
    return x
end

# ---------------------------------------------------------------------------- run
if abspath(PROGRAM_FILE) == @__FILE__
    OUTL = String[]; sl(s) = (println(s); flush(stdout); push!(OUTL, s))
    RL = Dict{Any,Any}()
    TESTSL = (:plane, :poincare, :balance)
    meshes = isempty(ARGS) ? ["mesh2", "mesh3", "mesh4", "mesh5", "mesh6", "mesh7"] : ARGS
    sl(@sprintf("linear rotating shallow water: k = 2π·3/L, ω = %.4e s⁻¹ = %.3f f, f²/ω² = %.3f, period %.2f h", OMW, OMW / FJ, FJ^2 / OMW^2, 2π / OMW / 3600))
    for mn in meshes
        mesh = VoronoiMesh("meshes/$mn.nc"); S = cached_stars(mn; mesh)
        for st in ("H_DEC", "H_P2")
            O = sw_ops(mesh, S[st]); Hs = (O.H + O.H') ./ 2
            for test in TESTSL
                η, u, T = exact_lin(test)
                h0, v0, fl0, _ = cochains(O, η, u, 0.0); hT, vT, _, _ = cochains(O, η, u, T)
                # truncation errors at t = T/3 (exact fields, exact time derivatives by central differences)
                t3 = T / 3; δ = 1.0
                h3, v3, fl3, cJ3 = cochains(O, η, u, t3); hp, vp, _, _ = cochains(O, η, u, t3 + δ); hm, vm, _, _ = cochains(O, η, u, t3 - δ)
                dh = (hp .- hm) ./ 2δ; U3 = O.H * v3
                RL[(mn, st, test, :flux)] = maximum(abs, U3 .- fl3) / maximum(abs, fl3)
                RL[(mn, st, test, :corW)] = maximum(abs, O.W * fl3 .- cJ3) / maximum(abs, cJ3)       # W alone (exact flux)
                RL[(mn, st, test, :cor)] = maximum(abs, O.W * U3 .- cJ3) / maximum(abs, cJ3)         # W ∘ H
                for M0k in (:A, :C)
                    rhs, Lmat, M0c = lin_ops(O, M0k)
                    if test != :balance
                        τh = (M0k == :A ? (H0J .* (O.D0t * U3)) ./ O.A : H0J .* (M0c \ (O.D0t * U3))) .- dh
                        RL[(mn, st, test, M0k, :cont)] = maximum(abs, τh) / maximum(abs, dh)
                    end
                    c = sqrt(GJ * H0J); dtmax = 0.1 * minimum(O.le) / c; n = ceil(Int, T / dtmax); dt = T / n
                    x0 = vcat(h0, v0); E0 = energy_lin(O, x0, Hs); Emax = Ref(0.0)
                    xT = rk4_lin(rhs, x0, dt, n; every = max(1, n ÷ 50), cb = (k, x) -> (Emax[] = max(Emax[], abs(energy_lin(O, x, Hs) / E0 - 1))))
                    eh = xT[1:O.nc] .- hT; ev = (xT[O.nc+1:end] .- vT) ./ O.le
                    w = O.le .* (O.H isa Diagonal ? 1 : 1)
                    rh = sqrt(sum(O.A .* eh .^ 2) / sum(O.A .* h0 .^ 2))
                    ueT = vT ./ O.le; dia = O.le .* (abs.(S["H_DEC"]) * ones(O.ne)) .* O.le ./ 2   # diamond areas |e||ẽ|/2 (|ẽ| = H_DEC |e|)
                    rv = sqrt(sum(dia .* ev .^ 2) / sum(dia .* ueT .^ 2))                    # relative to the exact velocity at T
                    RL[(mn, st, test, M0k)] = (rh = rh, rv = rv, h = sqrt(1 / O.nc), dE = energy_lin(O, xT, Hs) / E0 - 1, dEmax = Emax[], dt = dt, n = n)
                    sl(@sprintf("%s %-5s M0=%s %-8s depth %.3e  velocity %.3e  energy change %+.2e (max %.1e)  [dt %.0f s, %d steps]",
                                mn, st, M0k == :A ? "A " : "Ac", test, rh, rv, RL[(mn, st, test, M0k)].dE, Emax[], dt, n))
                end
                sl(@sprintf("   truncation at T/3, %s %s %s: flux Hv %.2e | Coriolis W(R̃⋆u) %.2e, W(Hv) %.2e | continuity M0=A %s, corrected %s", mn, st, test,
                            RL[(mn, st, test, :flux)], RL[(mn, st, test, :corW)], RL[(mn, st, test, :cor)],
                            test == :balance ? "-" : @sprintf("%.2e", RL[(mn, st, test, :A, :cont)]), test == :balance ? "-" : @sprintf("%.2e", RL[(mn, st, test, :C, :cont)])))
            end
            # energy with the implicit midpoint rule (M0 = A), meshes 4 and 5: semi-discrete conservation
            if mn in ("mesh4", "mesh5")
                η, u, T = exact_lin(:plane); h0, v0, _, _ = cochains(O, η, u, 0.0); _, Lmat, _ = lin_ops(O, :A)
                n = 400; dt = 10 * T / n; Id = sparse(1.0I, size(Lmat)...)
                F = lu(Id .- (dt / 2) .* Lmat); B = Id .+ (dt / 2) .* Lmat; x = vcat(h0, v0); E0 = energy_lin(O, x, Hs); Es = Float64[]
                for k in 1:n; x = F \ (B * x); push!(Es, energy_lin(O, x, Hs) / E0 - 1); end
                RL[(mn, st, :midpoint)] = Es
                sl(@sprintf("   implicit midpoint, %s %s, plane wave, 15 days (%d steps): energy change at end %+.2e, max %.2e", mn, st, n, Es[end], maximum(abs, Es)))
            end
            serialize(joinpath(TSDIR, "sw_linear_rot.jls"), RL)
        end
    end
    write(joinpath(TSDIR, "sw_linear_rot.txt"), join(OUTL, "\n") * "\n")
    println("done")
end
