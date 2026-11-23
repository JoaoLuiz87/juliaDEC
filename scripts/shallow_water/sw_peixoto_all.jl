# ============================================================================
#  sw_peixoto_all.jl -- the nonlinear tests of Peixoto, Alves, Fumagalli & Antonietti (2026, §5.1-5.4) with the three stars
#  H_DEC, H_P1, H_P2 in the nonlinear TRiSK model of sw_jet.jl (DEC variables; the star enters the volume flux Hv, hence the mass
#  flux and the PV flux, and the kinetic energy).  The linear tests of §5.2 are in sw_linear_three.jl.
#
#      julia -t 4 scripts/shallow_water/sw_peixoto_all.jl zonal vortex distort small jet figures
#
#    zonal    zonal geostrophic balance, 1 day: quasi-uniform SCVT meshes 2-7 and near-regular hexagonal meshes 2-7 (_regular)
#    vortex   stationary vortex in gradient-wind balance (f = 1.5645e-5 s⁻¹), 10 days, both families
#    distort  zonal balance on the distorted set dist_s00 ... dist_s30 (11600 cells, distortion 0 to 0.30)
#    small    small-amplitude limit: the nonlinear model from the linear plane wave with rotation, amplitudes 1, 10, 100 m,
#             16384 cells (mesh6), 1.5 days, against the linear exact solution
#    jet      the barotropically unstable jet, 20 days, APVM 0.5 on meshes 5-7 and inviscid on mesh 6, for H_P1
#             (H_DEC and H_P2 were run by sw_jet.jl / sw_jet_more.jl and are reused)
#    figures  fig_sw_nonlinear.png, fig_sw_jet.png, fig_sw_jet_diag.png
#  Not reproducible here: the locally refined family of the paper (its mesh repository is not public).
# ============================================================================
include(joinpath(@__DIR__, "sw_jet.jl"))
using CairoMakie

const STARS_A = ("H_DEC", "H_P1", "H_P2")
const OUTP = joinpath(TSDIR, "sw_peixoto.jls")
RP = isfile(OUTP) ? deserialize(OUTP) : Dict{Any,Any}()
LOGP = String[]; sp(s) = (println(s); flush(stdout); push!(LOGP, s))
savep() = (serialize(OUTP, RP); open(joinpath(TSDIR, "sw_peixoto.txt"), "a") do io; foreach(l -> println(io, l), LOGP); end; empty!(LOGP))

# ---------------------------------------------------------------- stationary vortex (Peixoto et al. §5.1, after their ref. [86])
const FVX = 1.5645e-5; const DHV = 417.0; const R0V = 0.379 * LDOM
hvx(r) = r < R0V ? H0J - DHV * cos(π * r / (2R0V))^8 : H0J
dhvx(r) = r < R0V ? DHV * 8 * cos(π * r / (2R0V))^7 * sin(π * r / (2R0V)) * π / (2R0V) : 0.0
vθ(r) = r < 1e-6 ? 0.0 : (r / 2) * (-FVX + sqrt(FVX^2 + 4GJ * dhvx(r) / r))          # v²/r + f v = g dh/dr
function mindisp(x, c, L)
    d = x .- c; return d .- L .* round.(d ./ L)
end
uvx(x, L) = (d = mindisp(x, [L / 2, L / 2], L); r = norm(d); r < 1e-6 ? [0.0, 0.0] : (vθ(r) / r) .* [-d[2], d[1]])
const GL8P = ((-0.9602898564975363, 0.1012285362903763), (-0.7966664774136267, 0.2223810344533745), (-0.5255324099163290, 0.3137066458778873),
              (-0.1834346424956498, 0.3626837833783620), (0.1834346424956498, 0.3626837833783620), (0.5255324099163290, 0.3137066458778873),
              (0.7966664774136267, 0.2223810344533745), (0.9602898564975363, 0.1012285362903763))
"circulations of u (physical coordinates) along the Delaunay edges, 8-point Gauss"
function circ_of(O, u)
    v = zeros(O.ne)
    for e in 1:O.ne
        p1, p2 = edge_endpoints(O.mesh, e, true); p1 = _p2(p1) .* O.L; p2 = _p2(p2) .* O.L; t = p2 .- p1
        v[e] = sum(w / 2 * dot(u(p1 .+ (ξ + 1) / 2 .* t), t) for (ξ, w) in GL8P)
    end
    return v
end
vortex_initial(O) = vcat([hvx(norm(mindisp(x, [O.L / 2, O.L / 2], O.L))) for x in O.xc], circ_of(O, x -> uvx(x, O.L)))

# ---------------------------------------------------------------- small-amplitude limit: the linear plane wave with rotation
const KPW = 2π * 3 / LDOM; const OMPW = sqrt(FJ^2 + GJ * H0J * KPW^2)
ηpw(x, t, a) = a * (FJ^2 / OMPW^2 + (GJ * H0J * KPW^2 / OMPW^2) * cos(OMPW * t)) * (sin(KPW * x[1]) + sin(KPW * x[2]))
plane_initial(O, a) = vcat([H0J + ηpw(x, 0.0, a) for x in O.xc], zeros(O.ne))

# ---------------------------------------------------------------- generic run
function run_state(mn, st, x0f; days, fcor = FJ, cfl = 0.3, exact = nothing, tag = "")
    mesh = VoronoiMesh("meshes/$mn.nc"); S = cached_stars(mn; mesh)
    O = sw_ops(mesh, S[st]; fcor = fcor)
    c = sqrt(GJ * H0J); T = days * DAY; n = ceil(Int, T / (cfl * minimum(O.le) / (c + 2U0J))); dt = T / n
    x0 = x0f(O); x = copy(x0); buf = sw_buffers(O); k1, k2, k3, k4, xt = (similar(x) for _ in 1:5)
    I0 = invariants(x, O)
    tw = @elapsed for _ in 1:n; rk4_step!(x, dt, O, buf, k1, k2, k3, k4, xt); end
    I1 = invariants(x, O)
    if exact === nothing                                                    # steady state: compare with the initial state
        ηr = x0[1:O.nc] .- H0J; eh = x[1:O.nc] .- x0[1:O.nc]
        rh = sqrt(sum(O.A .* eh .^ 2) / sum(O.A .* ηr .^ 2)); mh = maximum(abs, eh) / maximum(abs, ηr)
    else                                                                    # against an exact depth at time T
        ηT = [exact(xc, T) for xc in O.xc]; eh = (x[1:O.nc] .- H0J) .- ηT
        rh = sqrt(sum(O.A .* eh .^ 2) / sum(O.A .* ηT .^ 2)); mh = maximum(abs, eh) / maximum(abs, ηT)
    end
    rv = exact === nothing ? sqrt(sum(abs2, x[O.nc+1:end] .- x0[O.nc+1:end]) / sum(abs2, x0[O.nc+1:end])) : NaN
    r = (rh = rh, mh = mh, rv = rv, h = O.L / sqrt(O.nc), nc = O.nc, dmass = I1.mass / I0.mass - 1, denergy = I1.energy / I0.energy - 1,
         denstr = I1.enstrophy / I0.enstrophy - 1, dt = dt, n = n, wall = tw)
    sp(@sprintf("%-8s %-14s %-5s depth rms %.3e max %.3e  velocity %.3e  Δmass %+.1e Δenergy %+.2e  [%d cells, dt %.0f s, %.0f s]",
                tag, mn, st, rh, mh, rv, r.dmass, r.denergy, O.nc, dt, tw))
    return r
end

function part_zonal()
    for fam in ("", "_regular"), i in 2:7, st in STARS_A
        mn = "mesh$(i)$(fam)"; key = (:zonal, mn, st); haskey(RP, key) && continue
        RP[key] = run_state(mn, st, steady_initial; days = 1, tag = "zonal"); savep()
    end
end
function part_vortex()
    r = range(0, R0V; length = 20001); v = vθ.(r); k = argmax(v)
    sp(@sprintf("vortex: max wind %.2f m/s at r = %.3f L (paper: 31.3 m/s at 0.1 L); depression %.0f m; f = %.4e s⁻¹", v[k], r[k] / LDOM, DHV, FVX))
    for fam in ("", "_regular"), i in 2:7, st in STARS_A
        mn = "mesh$(i)$(fam)"; key = (:vortex, mn, st); haskey(RP, key) && continue
        RP[key] = run_state(mn, st, vortex_initial; days = 10, fcor = FVX, tag = "vortex"); savep()
    end
end
function part_distort()
    for s in ("00", "05", "10", "15", "20", "25", "30"), st in STARS_A
        mn = "dist_s$s"; key = (:distort, mn, st); haskey(RP, key) && continue
        S = cached_stars(mn); haskey(RP, (:distort_res, mn)) || (RP[(:distort_res, mn)] = S["info"])
        RP[key] = run_state(mn, st, steady_initial; days = 1, tag = "distort"); savep()
    end
end
function part_small()
    for a in (1.0, 10.0, 100.0), st in STARS_A
        key = (:small, a, st); haskey(RP, key) && continue
        RP[key] = run_state("mesh6", st, O -> plane_initial(O, a); days = 1.5, exact = (x, t) -> ηpw(x, t, a), tag = @sprintf("A=%gm", a)); savep()
    end
end
function part_jet()
    for mn in ("mesh5", "mesh6", "mesh7")
        isfile(joinpath(TSDIR, "sw_jet_$(mn)_H_P1_apvm.jls")) || run_jet(mn, "H_P1"; apvm = 0.5, log = sp)
        savep()
    end
    isfile(joinpath(TSDIR, "sw_jet_mesh6_H_P1.jls")) || run_jet("mesh6", "H_P1"; log = sp)
    savep()
end

# ---------------------------------------------------------------- figures
const COLS = Dict("H_DEC" => colorant"#7a7a7a", "H_P1" => colorant"#2b6cb0", "H_P2" => colorant"#c23b3a")
const MKS = Dict("H_DEC" => :circle, "H_P1" => :xcross, "H_P2" => :rect)
function fig_nonlinear()
    f = Figure(size = (1500, 900))
    function conv(pos, test, fam, title; ylabel = "")
        hk = [RP[(test, "mesh$(i)$(fam)", "H_DEC")].h / 1e3 for i in 2:7 if haskey(RP, (test, "mesh$(i)$(fam)", "H_DEC"))]
        ax = Axis(pos; xscale = log10, yscale = log10, xreversed = true, xlabel = "h (km)", ylabel = ylabel, title = title, titlesize = 13,
                  xticks = (hk, [string(round(Int, x)) for x in hk]))
        for st in STARS_A
            ks = [("mesh$(i)$(fam)") for i in 2:7 if haskey(RP, (test, "mesh$(i)$(fam)", st))]
            isempty(ks) && continue
            scatterlines!(ax, [RP[(test, m, st)].h / 1e3 for m in ks], [RP[(test, m, st)].rh for m in ks]; color = COLS[st], marker = MKS[st],
                          markersize = 10, linewidth = 2, label = st)
        end
        return ax
    end
    a1 = conv(f[1, 1], :zonal, "", "(a) zonal balance, quasi-uniform SCVT, 1 day"; ylabel = "relative depth error (rms)")
    axislegend(a1; position = :lb, framevisible = false)
    conv(f[1, 2], :zonal, "_regular", "(b) zonal balance, near-regular hexagons, 1 day")
    ax = Axis(f[1, 3]; yscale = log10, xlabel = "distortion s", title = "(c) zonal balance, distorted set (11600 cells)", titlesize = 13)
    ss = ("00", "05", "10", "15", "20", "25", "30")
    for st in STARS_A
        ks = [s for s in ss if haskey(RP, (:distort, "dist_s$s", st))]
        isempty(ks) || scatterlines!(ax, [parse(Int, s) / 100 for s in ks], [RP[(:distort, "dist_s$s", st)].rh for s in ks]; color = COLS[st],
                                     marker = MKS[st], markersize = 10, linewidth = 2)
    end
    conv(f[2, 1], :vortex, "", "(d) stationary vortex, quasi-uniform SCVT, 10 days"; ylabel = "relative depth error (rms)")
    conv(f[2, 2], :vortex, "_regular", "(e) stationary vortex, near-regular hexagons, 10 days")
    ax = Axis(f[2, 3]; xscale = log10, yscale = log10, xlabel = "amplitude (m)", title = "(f) small amplitude: nonlinear model vs linear exact, mesh 6",
              titlesize = 13, ylabel = "relative depth error at 1.5 days")
    for st in STARS_A
        as = [a for a in (1.0, 10.0, 100.0) if haskey(RP, (:small, a, st))]
        isempty(as) || scatterlines!(ax, as, [RP[(:small, a, st)].rh for a in as]; color = COLS[st], marker = MKS[st], markersize = 10, linewidth = 2)
    end
    CairoMakie.save(joinpath(@__DIR__, "..", "..", "output", "figures", "fig_sw_nonlinear.png"), f; px_per_unit = 2)
end

loadj(mn, st, sfx) = (fl = joinpath(TSDIR, "sw_jet_$(mn)_$(st)$(sfx).jls"); isfile(fl) ? deserialize(fl) : nothing)
const VMX = 8.0e-5
const VCM = cgrad([colorant"#1c5cab", colorant"#7fa9d9", colorant"#f0efec", colorant"#e39a8f", colorant"#c23b3a"])
function jet_polys2(mesh)
    tris, _, _ = triangles_ccw(mesh); LK = LDOM / 1e6
    out = Vector{Vector{Point2f}}(); own = Int[]
    for (i, T) in enumerate(tris)
        Q = [Point2f(p[1] * LK, p[2] * LK) for p in T.pts]; push!(out, Q); push!(own, i)
        any(p -> p[1] < 0 || p[1] > LK || p[2] < 0 || p[2] > LK, Q) || continue
        for sx in -1:1, sy in -1:1
            (sx, sy) == (0, 0) || (push!(out, [p .+ Point2f(sx * LK, sy * LK) for p in Q]); push!(own, i))
        end
    end
    return out, own
end
function fig_jet()
    mesh = VoronoiMesh("meshes/mesh6.nc"); polys, own = jet_polys2(mesh); LK = LDOM / 1e6
    days = (6, 8, 10, 14, 20); f = Figure(size = (1500, 960), fontsize = 14)
    for (i, st) in enumerate(STARS_A)
        R = loadj("mesh6", st, "_apvm"); R === nothing && continue
        for (j, d) in enumerate(days)
            haskey(R.snaps, d) || continue
            ax = Axis(f[i, j]; aspect = DataAspect(), title = (j == 1 ? st * ", " : "") * "day $d", titlesize = 14, titlealign = :left)
            hidedecorations!(ax); poly!(ax, polys; color = R.snaps[d].ζ[own], colormap = VCM, colorrange = (-VMX, VMX), strokewidth = 0)
            xlims!(ax, 0, LK); ylims!(ax, 0, LK)
        end
    end
    Colorbar(f[1:3, 6]; colormap = VCM, limits = (-VMX * 1e5, VMX * 1e5), label = "relative vorticity (10⁻⁵ s⁻¹)")
    CairoMakie.save(joinpath(@__DIR__, "..", "..", "output", "figures", "fig_sw_jet.png"), f; px_per_unit = 2)
    g = Figure(size = (1500, 470))
    a1 = Axis(g[1, 1]; xlabel = "days", ylabel = "E(t)/E(0) − 1", title = "(a) energy, mesh 6, APVM 0.5", titlesize = 13)
    a2 = Axis(g[1, 2]; xlabel = "days", ylabel = "Z(t)/Z(0) − 1", title = "(b) potential enstrophy, mesh 6, APVM 0.5", titlesize = 13)
    a3 = Axis(g[1, 3]; xlabel = "days", yscale = log10, ylabel = "‖ζ − ζ(H_DEC)‖ / ‖ζ(H_DEC)‖", title = "(c) difference from H_DEC, meshes 5–7", titlesize = 13)
    for st in STARS_A
        R = loadj("mesh6", st, "_apvm"); R === nothing && continue
        t = [p.t / DAY for p in R.inv]; E0 = R.inv[1].energy; Z0 = R.inv[1].enstrophy
        lines!(a1, t, [p.energy / E0 - 1 for p in R.inv]; color = COLS[st], linewidth = 2, label = st)
        lines!(a2, t, [p.enstrophy / Z0 - 1 for p in R.inv]; color = COLS[st], linewidth = 2)
    end
    axislegend(a1; position = :lb, framevisible = false)
    for (mn, ls) in (("mesh5", :dot), ("mesh6", :dash), ("mesh7", :solid)), st in ("H_P1", "H_P2")
        R0 = loadj(mn, "H_DEC", "_apvm"); R = loadj(mn, st, "_apvm"); (R0 === nothing || R === nothing) && continue
        ds = sort([d for d in keys(R.snaps) if d > 0 && haskey(R0.snaps, d)])
        isempty(ds) || lines!(a3, collect(ds), [norm(R.snaps[d].ζ .- R0.snaps[d].ζ) / norm(R0.snaps[d].ζ) for d in ds]; color = COLS[st], linestyle = ls,
                             linewidth = 2, label = "$st, $mn")
    end
    axislegend(a3; position = :rb, framevisible = false, labelsize = 11)
    CairoMakie.save(joinpath(@__DIR__, "..", "..", "output", "figures", "fig_sw_jet_diag.png"), g; px_per_unit = 2)
end

if abspath(PROGRAM_FILE) == @__FILE__
    for part in ARGS
        part == "zonal" && part_zonal(); part == "vortex" && part_vortex(); part == "distort" && part_distort()
        part == "small" && part_small(); part == "jet" && part_jet()
        part == "figures" && (fig_nonlinear(); fig_jet())
    end
    savep(); println("done")
end
