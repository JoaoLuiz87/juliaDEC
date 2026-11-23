# ============================================================================
#  sw_ablation.jl -- which operator limits the linear shallow-water model?  Four schemes for the linear rotating shallow-water tests
#  of Peixoto et al. (2026, §5.2) (exact solutions, cochains, RK4 at CFL 0.1 as in sw_linear_three.jl):
#
#      trsk        W (TRiSK Coriolis),  M0 = A                      (sw_linear_three.jl, cached in ts_data/sw_three.jls)
#      consistent  P (exact on [P2]²),  M0ᶜ (exact on cubics)        (idem)
#      W_Mc        W,                   M0ᶜ                          only the mass matrix upgraded
#      P_A         P,                   M0 = A                       only the Coriolis operator upgraded
#
#      julia -t 4 scripts/shallow_water/sw_ablation.jl            -> ts_data/sw_ablation.{jls,txt}
#      julia scripts/shallow_water/sw_ablation.jl figures         -> fig_sw_ablation.png
# ============================================================================
include(joinpath(@__DIR__, "sw_linear_three.jl"))
const OUTA = joinpath(TSDIR, "sw_ablation.jls")

"right-hand side for the mixed schemes (cf. scheme_ops in sw_linear_three.jl)"
function ablation_rhs(O, scheme, P, M0c)
    nc, ne = O.nc, O.ne
    if scheme == :P_A
        Minv = spdiagm(0 => 1 ./ O.A)
        Lmat = [spzeros(nc, nc) H0J .* Minv * O.D0t * O.H; -GJ .* O.D0 -FJ .* P]
        return x -> Lmat * x
    else                                                                                  # :W_Mc
        return x -> (v = view(x, nc+1:nc+ne); U = O.H * v;
                     vcat(H0J .* (M0c \ (O.D0t * U)), -FJ .* (O.W * U) .- GJ .* (O.D0 * view(x, 1:nc))))
    end
end

function run_ablation(meshes)
    R = isfile(OUTA) ? deserialize(OUTA) : Dict{Any,Any}()
    OUTL = String[]; sl(s) = (println(s); flush(stdout); push!(OUTL, s))
    for mn in meshes
        mesh = VoronoiMesh("meshes/$mn.nc"); S = cached_stars(mn; mesh); O0 = sw_ops(mesh, S["H_DEC"])
        P = perp_p2(mesh, O0.W * O0.H); M0c = lu(mass0_corrected(mesh) .* O0.L^2)
        for st in STARS3
            O = sw_ops(mesh, S[st]); Hs = (O.H + O.H') ./ 2
            for scheme in (:W_Mc, :P_A), test in TESTS3
                key = (mn, st, scheme, test); haskey(R, key) && continue
                η, u, T = exact_lin(test)
                h0, v0, _, _ = cochains(O, η, u, 0.0); hT, vT, _, _ = cochains(O, η, u, T)
                rhs = ablation_rhs(O, scheme, P, M0c)
                c = sqrt(GJ * H0J); dtmax = 0.1 * minimum(O.le) / c; n = ceil(Int, T / dtmax); dt = T / n
                x0 = vcat(h0, v0); E0 = energy_lin(O, x0, Hs); Emax = Ref(0.0)
                xT = rk4_lin(rhs, x0, dt, n; every = max(1, n ÷ 60), cb = (k, x) -> (Emax[] = max(Emax[], abs(energy_lin(O, x, Hs) / E0 - 1))))
                eh = xT[1:O.nc] .- hT
                rh = sqrt(sum(O.A .* eh .^ 2) / sum(O.A .* hT .^ 2))
                R[key] = (rh = rh, dE = energy_lin(O, xT, Hs) / E0 - 1, dEmax = Emax[], h = LDOM / sqrt(O.nc), dt = dt, n = n)
                sl(@sprintf("%s %-5s %-5s %-8s depth %.3e  ΔE/E %+.2e", mn, st, scheme, test, rh, R[key].dE)); serialize(OUTA, R)
            end
        end
    end
    write(joinpath(TSDIR, "sw_ablation.txt"), join(OUTL, "\n") * "\n")
end

function fig_ablation()
    RA = deserialize(OUTA); R3 = deserialize(OUT3)
    get4(mn, st, scheme, test) = scheme in (:trsk, :consistent) ? get(R3, (mn, st, scheme, test), nothing) : get(RA, (mn, st, scheme, test), nothing)
    ms = ["mesh$i" for i in 4:7]; hs = [LDOM / sqrt(VoronoiMesh("meshes/$m.nc").cells.n) / 1e3 for m in ms]
    cols = Dict(:trsk => colorant"#7a7a7a", :W_Mc => colorant"#2b6cb0", :P_A => colorant"#d08a1e", :consistent => colorant"#c23b3a")
    nm = Dict(:trsk => "W, A (TRSK)", :W_Mc => "W, M0ᶜ", :P_A => "P, A", :consistent => "P, M0ᶜ (consistent)")
    tname = Dict(:plane => "plane wave with rotation", :poincare => "Poincaré wave", :balance => "geostrophic balance")
    f = Figure(size = (1500, 480))
    for (j, test) in enumerate(TESTS3)
        ax = Axis(f[1, j]; xscale = log10, yscale = log10, xreversed = true, xlabel = "h (km)", titlesize = 13,
                  xticks = (hs, [string(round(Int, x)) for x in hs]), ylabel = j == 1 ? "relative depth error (rms)" : "", title = tname[test] * ", H_P2")
        for scheme in (:trsk, :W_Mc, :P_A, :consistent)
            v = [(r = get4(m, "H_P2", scheme, test); r === nothing ? NaN : r.rh) for m in ms]
            any(isfinite, v) && scatterlines!(ax, hs, v; color = cols[scheme], markersize = 10, linewidth = 2.2, label = nm[scheme])
        end
        d = [(r = get4(m, "H_DEC", :trsk, test); r === nothing ? NaN : r.rh) for m in ms]
        scatterlines!(ax, hs, d; color = :black, linestyle = :dash, marker = :xcross, markersize = 10, label = "H_DEC, W, A (TRSK)")
        j == 1 && axislegend(ax; position = :lb, framevisible = false, labelsize = 12)
    end
    CairoMakie.save(joinpath(@__DIR__, "..", "..", "output", "figures", "fig_sw_ablation.png"), f; px_per_unit = 2)
end

if abspath(PROGRAM_FILE) == @__FILE__
    if !isempty(ARGS) && ARGS[1] == "figures"
        fig_ablation()
    else
        run_ablation(isempty(ARGS) ? ["mesh4", "mesh5", "mesh6", "mesh7"] : ARGS); fig_ablation()
    end
end
