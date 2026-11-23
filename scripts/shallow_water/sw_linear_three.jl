# ============================================================================
#  sw_linear_three.jl -- the linear rotating shallow-water tests of Peixoto, Alves, Fumagalli & Antonietti (2026, §5.2)
#  with the three stars H_DEC, H_P1, H_P2, in two schemes:
#
#    trsk        TRiSK/TRSK in DEC variables (as in sw_linear_rot.jl): Coriolis −f W(H v), M0 = diag(A).
#                Energy E = ½ g ηᵀAη + ½ H0 vᵀ H_s v is conserved by the semi-discrete system when H is symmetric.
#    consistent  the same equations with the two other approximations made high order, so that only H limits the order:
#                Coriolis −f P v, P the row-by-row least change of W·H_DEC exact on [P2]² (3-ring), and M0 the 0-form star
#                exact on cubics (mass0_corrected).  Not energy conserving for any H.
#
#      julia -t 4 scripts/shallow_water/sw_linear_three.jl [meshes...]    -> output/ts_data/sw_three.{txt,jls}
#      julia -t 4 scripts/shallow_water/sw_linear_three.jl energy         -> long implicit-midpoint energy runs (trsk), sw_three_energy.jls
#      julia scripts/shallow_water/sw_linear_three.jl figures             -> fig_sw_conv.png, fig_sw_energy.png
#  Tests, exact solutions, cochains and RK4 (CFL 0.1 on the shortest edge) as in sw_linear_rot.jl.
# ============================================================================
include(joinpath(@__DIR__, "sw_linear_rot.jl"))
using CairoMakie

"P: circulations v -> ∫_e (J u)·t_e, row-wise least change of P0 exact on [P2]² on the r-ring (cf. rowwise_correction)."
function perp_p2(mesh, P0; k = 2, r = 3)
    n = mesh.edges.n; lp = edge_lengths(mesh, true); P0t = sparse(P0')
    Ii, Jj, Vv = Int[], Int[], Float64[]
    for e in 1:n
        xv = mesh.edges.position[e]; x0 = _pt(xv); s = lp[e]
        p1, p2 = edge_endpoints(mesh, e, true; about = xv)
        b = seg_moments(p1, p2, x0, s, k; rotate = true)                   # ∫_e (J f_m)·dx
        mom(j) = (q = edge_endpoints(mesh, j, true; about = xv); seg_moments(q[1], q[2], x0, s, k))
        for p in nzrange(P0t, e); b .-= nonzeros(P0t)[p] .* mom(rowvals(P0t)[p]); end
        Se = ring(mesh, e, r); Am = reduce(hcat, [mom(j) for j in Se])
        c = pinv(Am) * b
        append!(Ii, fill(e, length(Se))); append!(Jj, Se); append!(Vv, c)
    end
    return P0 + sparse(Ii, Jj, Vv, n, n)
end

"right-hand side for a scheme: returns (rhs, Lmat or nothing)"
function scheme_ops(O, scheme, P, M0c)
    nc, ne = O.nc, O.ne
    if scheme == :trsk
        Minv = spdiagm(0 => 1 ./ O.A)
        Lmat = [spzeros(nc, nc) H0J .* Minv * O.D0t * O.H; -GJ .* O.D0 -FJ .* O.W * O.H]
        return x -> Lmat * x, Lmat
    else
        f = x -> (v = view(x, nc+1:nc+ne); vcat(H0J .* (M0c \ (O.D0t * (O.H * v))), -FJ .* (P * v) .- GJ .* (O.D0 * view(x, 1:nc))))
        return f, nothing
    end
end

const STARS3 = ("H_DEC", "H_P1", "H_P2")
const TESTS3 = (:plane, :poincare, :balance)
const OUT3 = joinpath(TSDIR, "sw_three.jls")

function run_conv(meshes)
    R = isfile(OUT3) ? deserialize(OUT3) : Dict{Any,Any}()
    OUTL = String[]; sl(s) = (println(s); flush(stdout); push!(OUTL, s))
    sl(@sprintf("linear rotating shallow water (Peixoto et al. 2026, §5.2): ω = %.4e s⁻¹ = %.3f f, period %.2f h", OMW, OMW / FJ, 2π / OMW / 3600))
    for mn in meshes
        mesh = VoronoiMesh("meshes/$mn.nc"); S = cached_stars(mn; mesh)
        O0 = sw_ops(mesh, S["H_DEC"])
        t0 = time(); P = perp_p2(mesh, O0.W * O0.H); M0c = lu(mass0_corrected(mesh) .* O0.L^2)
        sl(@sprintf("%s: %d cells, %d edges; P and M0 built in %.0f s", mn, O0.nc, O0.ne, time() - t0))
        # truncation of the Coriolis operators on the Poincaré wave at T/3
        η, u, T = exact_lin(:poincare); _, v3, fl3, cJ3 = cochains(O0, η, u, T / 3)
        sl(@sprintf("   Coriolis truncation (Poincaré, max rel.): W·R̃⋆u %.2e,  P·R_h u %.2e", maximum(abs, O0.W * fl3 .- cJ3) / maximum(abs, cJ3),
                    maximum(abs, P * v3 .- cJ3) / maximum(abs, cJ3)))
        for st in STARS3
            O = sw_ops(mesh, S[st]); Hs = (O.H + O.H') ./ 2
            for scheme in (:trsk, :consistent), test in TESTS3
                key = (mn, st, scheme, test); haskey(R, key) && (r = R[key]; sl(@sprintf("%s %-5s %-10s %-8s depth %.3e  velocity %.3e  ΔE/E %+.2e (max %.1e)  [cached]", mn, st, scheme, test, r.rh, r.rv, r.dE, r.dEmax)); continue)
                η, u, T = exact_lin(test)
                h0, v0, _, _ = cochains(O, η, u, 0.0); hT, vT, _, _ = cochains(O, η, u, T)
                rhs, _ = scheme_ops(O, scheme, P, M0c)
                c = sqrt(GJ * H0J); dtmax = 0.1 * minimum(O.le) / c; n = ceil(Int, T / dtmax); dt = T / n
                x0 = vcat(h0, v0); E0 = energy_lin(O, x0, Hs); Emax = Ref(0.0); Et = Float64[]; tt = Float64[]
                xT = rk4_lin(rhs, x0, dt, n; every = max(1, n ÷ 60),
                             cb = (k, x) -> (ε = energy_lin(O, x, Hs) / E0 - 1; Emax[] = max(Emax[], abs(ε)); push!(Et, ε); push!(tt, k * dt)))
                eh = xT[1:O.nc] .- hT; ev = (xT[O.nc+1:end] .- vT) ./ O.le
                rh = sqrt(sum(O.A .* eh .^ 2) / sum(O.A .* hT .^ 2))
                dia = O.le .* (abs.(S["H_DEC"]) * ones(O.ne)) .* O.le ./ 2
                rv = sqrt(sum(dia .* ev .^ 2) / sum(dia .* (vT ./ O.le) .^ 2))
                R[key] = (rh = rh, rv = rv, hinf = maximum(abs, eh) / maximum(abs, hT), dE = energy_lin(O, xT, Hs) / E0 - 1, dEmax = Emax[],
                          Et = Et, tt = tt, dt = dt, n = n, h = LDOM / sqrt(O.nc))
                sl(@sprintf("%s %-5s %-10s %-8s depth %.3e  velocity %.3e  ΔE/E %+.2e (max %.1e)  [dt %.0f s, %d steps]", mn, st, scheme, test,
                            rh, rv, R[key].dE, Emax[], dt, n))
                serialize(OUT3, R)
            end
        end
    end
    sl("\nobserved orders of the depth error between consecutive meshes")
    ms = [m for m in ("mesh2", "mesh3", "mesh4", "mesh5", "mesh6", "mesh7") if haskey(R, (m, "H_DEC", :trsk, :plane))]
    for scheme in (:trsk, :consistent), test in TESTS3, st in STARS3
        v = [R[(m, st, scheme, test)].rh for m in ms if haskey(R, (m, st, scheme, test))]
        length(v) < 2 && continue
        sl(@sprintf("   %-10s %-8s %-5s ", scheme, test, st) * join([@sprintf("%5.2f", log2(v[i] / v[i+1])) for i in 1:length(v)-1], " ") *
           @sprintf("   (finest %.2e)", v[end]))
    end
    write(joinpath(TSDIR, "sw_three.txt"), join(OUTL, "\n") * "\n")
end

"long runs with the implicit midpoint rule (trsk scheme, energy exactly conserved for symmetric H): plane wave, 30 days"
function run_energy(; meshes = ("mesh4", "mesh5", "mesh6"), days = 30, cfl = 0.4)
    f = joinpath(TSDIR, "sw_three_energy.jls"); RE = isfile(f) ? deserialize(f) : Dict{Any,Any}()
    for mn in meshes
        mesh = VoronoiMesh("meshes/$mn.nc"); S = cached_stars(mn; mesh)
        for st in STARS3
            haskey(RE, (mn, st)) && continue
            O = sw_ops(mesh, S[st]); Hs = (O.H + O.H') ./ 2
            η, u, _ = exact_lin(:plane); h0, v0, _, _ = cochains(O, η, u, 0.0)
            _, Lmat = scheme_ops(O, :trsk, nothing, nothing)
            c = sqrt(GJ * H0J); T = days * DAY; dt0 = cfl * minimum(O.le) / c; n = ceil(Int, T / dt0); dt = T / n
            Id = sparse(1.0I, size(Lmat)...); F = lu(Id .- (dt / 2) .* Lmat); B = Id .+ (dt / 2) .* Lmat
            x = vcat(h0, v0); E0 = energy_lin(O, x, Hs); Es = Float64[]; ts = Float64[]; every = max(1, n ÷ 300)
            for k in 1:n
                x = F \ (B * x)
                k % every == 0 && (push!(Es, energy_lin(O, x, Hs) / E0 - 1); push!(ts, k * dt / DAY))
            end
            RE[(mn, st)] = (Es = Es, ts = ts, dt = dt, n = n)
            println(@sprintf("%s %-5s implicit midpoint, plane wave, %d days (%d steps, dt %.0f s): ΔE/E at end %+.2e, max %.2e", mn, st, days, n, dt,
                             Es[end], maximum(abs, Es))); flush(stdout)
            serialize(f, RE)
        end
    end
end

function figures()
    R = deserialize(OUT3)
    ms = [m for m in ("mesh2", "mesh3", "mesh4", "mesh5", "mesh6", "mesh7") if haskey(R, (m, "H_DEC", :trsk, :plane))]
    hs = [R[(m, "H_DEC", :trsk, :plane)].h / 1e3 for m in ms]
    col = Dict("H_DEC" => colorant"#7a7a7a", "H_P1" => colorant"#2b6cb0", "H_P2" => colorant"#c23b3a")
    mk = Dict("H_DEC" => :circle, "H_P1" => :xcross, "H_P2" => :rect)
    tname = Dict(:plane => "plane wave with rotation (1.5 days)", :poincare => "Poincaré wave (one period)", :balance => "geostrophic balance (1 day)")
    sname = Dict(:trsk => "TRSK: TRiSK Coriolis, M0 = A", :consistent => "consistent: Coriolis and M0 exact on P2/P3")
    f = Figure(size = (1500, 860))
    for (i, scheme) in enumerate((:trsk, :consistent)), (j, test) in enumerate(TESTS3)
        ax = Axis(f[i, j]; xscale = log10, yscale = log10, xreversed = true, xlabel = i == 2 ? "h (km)" : "", titlesize = 13,
                  xticks = (hs, [string(round(Int, x)) for x in hs]),
                  ylabel = j == 1 ? "relative depth error (rms)\n" * sname[scheme] : "", title = i == 1 ? tname[test] : "")
        for st in STARS3
            v = [R[(m, st, scheme, test)].rh for m in ms if haskey(R, (m, st, scheme, test))]
            scatterlines!(ax, hs[1:length(v)], v; color = col[st], marker = mk[st], markersize = 10, linewidth = 2, label = st)
        end
        for (k, y0) in ((2, 1.0), (3, 1.0), (4, 1.0))
            yb = [R[(ms[end-2], "H_DEC", scheme, test)].rh]
            lines!(ax, [hs[end-2], hs[end]], yb[1] .* 0.3 .* ([hs[end-2], hs[end]] ./ hs[end-2]) .^ k; color = :black, linestyle = :dot, linewidth = 1)
            text!(ax, hs[end], yb[1] * 0.3 * (hs[end] / hs[end-2])^k; text = " h" * Dict(2 => "²", 3 => "³", 4 => "⁴")[k], fontsize = 11)
        end
        (i, j) == (1, 1) && axislegend(ax; position = :lb, framevisible = false)
    end
    CairoMakie.save(joinpath(@__DIR__, "..", "..", "output", "figures", "fig_sw_conv.png"), f; px_per_unit = 2)
    fe = joinpath(TSDIR, "sw_three_energy.jls")
    if isfile(fe)
        RE = deserialize(fe); g = Figure(size = (1500, 480))
        for (j, mn) in enumerate(("mesh4", "mesh5", "mesh6"))
            ax = Axis(g[1, j]; yscale = log10, xlabel = "days", ylabel = j == 1 ? "|E(t)/E(0) − 1|" : "", title = "$mn, TRSK, implicit midpoint", titlesize = 13)
            for st in STARS3
                haskey(RE, (mn, st)) || continue
                r = RE[(mn, st)]; lines!(ax, r.ts, max.(abs.(r.Es), 1e-17); color = col[st], linewidth = 2, label = st)
            end
            ylims!(ax, 1e-17, 1e1); j == 1 && axislegend(ax; position = :rc, framevisible = false)
        end
        CairoMakie.save(joinpath(@__DIR__, "..", "..", "output", "figures", "fig_sw_energy.png"), g; px_per_unit = 2)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    if !isempty(ARGS) && ARGS[1] == "energy"
        run_energy()
    elseif !isempty(ARGS) && ARGS[1] == "figures"
        figures()
    else
        run_conv(isempty(ARGS) ? ["mesh2", "mesh3", "mesh4", "mesh5", "mesh6", "mesh7"] : ARGS)
    end
end
