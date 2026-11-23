# ============================================================================
#  lift_offset_figures.jl  -- figures and numbers for lift_offset_symmetry.tex
#
#      julia scripts/barrier/lift_offset_figures.jl
#
#  Stars: diagonal H0 (symmetric reference), lift [P1]^2, lift [P2]^2.
#  Families: regular lattice (offset-free) and SCVT, meshes 2-5.
#  Prints every number quoted in the text and writes fig_lo_*.png.
# ============================================================================
using VoronoiMeshes, TensorsLite, TensorsLiteGeometry, SmallCollections, LinearAlgebra, NCDatasets
using StaticArrays, SparseArrays, Printf, CairoMakie
Base.similar(bc::Base.Broadcast.Broadcasted, ::Type{Union{}}) = Array{Union{}}(undef, length.(axes(bc)))
const ROOT = joinpath(@__DIR__, "..", "..")
cd(ROOT)
include(joinpath(ROOT, "src", "juliaDEC.jl")); include(joinpath(ROOT, "src", "hodge_stars.jl"))
const OUT = joinpath(ROOT, "report")

const INK = colorant"#2b2b29"; const MUTED = colorant"#8a8a85"; const GRID = colorant"#e6e5e1"
const BLUE = colorant"#2a78d6"; const ORANGE = colorant"#eb6834"; const AQUA = colorant"#1baf7a"
const DIV = cgrad([colorant"#1c5cab", colorant"#f0efec", colorant"#c23b3a"])
set_theme!(Theme(fontsize = 13, textcolor = INK,
    Axis = (xgridcolor = GRID, ygridcolor = GRID, spinewidth = 0.8, leftspinecolor = MUTED,
            bottomspinecolor = MUTED, topspinevisible = false, rightspinevisible = false,
            xtickcolor = MUTED, ytickcolor = MUTED, titlefont = :regular, titlesize = 13),
    Legend = (framevisible = false, labelsize = 11)))
savefig(name, f) = (CairoMakie.save(joinpath(OUT, name), f, px_per_unit = 2); println("  wrote ", name))

# ---------------------------------------------------------------- notation of the paper
α(m, w) = reduce_primal_edges(m, w).values
β(m, w) = reduce_dual_edges(m, star_field(w)).values
ω(m) = edge_lengths(m, false) ./ edge_lengths(m, true)
normW(m, x) = sqrt(sum(ω(m) .* x .^ 2)); normWinv(m, y) = sqrt(sum(y .^ 2 ./ ω(m)))
ε(H, m, w) = H * α(m, w) - β(m, w)
skw(H) = (H - H') / 2
s_h(m, u, v) = dot(α(m, v), β(m, u)) - dot(α(m, u), β(m, v))
meanh(m) = sum(edge_lengths(m, true)) / m.edges.n
d0m(m) = sparse(Float64.(d0_matrix_primal(m))); D1m(m) = sparse(Float64.(d1_matrix_primal(m)))
rates(h, e) = [log(e[i] / e[i+1]) / log(h[i] / h[i+1]) for i in 1:length(h)-1]
jac(w, x; d = 1e-6) = hcat((w(x .+ [d, 0]) .- w(x .- [d, 0])) ./ (2d), (w(x .+ [0, d]) .- w(x .- [0, d])) ./ (2d))

U(x) = [sin(2pi * x[2]), cos(2pi * x[1])]
V(x) = [cos(2pi * (x[1] + x[2])), sin(2pi * (x[1] - 2x[2]))]
pex(x)  = sin(2pi * x[1]) * sin(2pi * x[2]) + 0.3cos(2pi * (x[1] - 2x[2]))
gpex(x) = [2pi * cos(2pi * x[1]) * sin(2pi * x[2]) - 0.6pi * sin(2pi * (x[1] - 2x[2])),
           2pi * sin(2pi * x[1]) * cos(2pi * x[2]) + 1.2pi * sin(2pi * (x[1] - 2x[2]))]
lapp(x) = -8pi^2 * sin(2pi * x[1]) * sin(2pi * x[2]) - 6pi^2 * cos(2pi * (x[1] - 2x[2]))
uex(x)  = (g = gpex(x); [g[2], -g[1]])               # u = -J grad p : star u = grad p, curl u = -Lap p

const FAMS = ["regular" => ["mesh2_regular", "mesh3_regular", "mesh4_regular", "mesh5_regular"],
              "SCVT"    => ["mesh2", "mesh3", "mesh4", "mesh5"]]
const ALL = vcat(last.(FAMS)...)
const NAMES = ["diagonal", "lift P1", "lift P2"]
const COL = Dict("diagonal" => MUTED, "lift P1" => ORANGE, "lift P2" => BLUE)
const LS = Dict("regular" => :dash, "SCVT" => :solid)
const MK = Dict("regular" => :circle, "SCVT" => :utriangle)

MESH = Dict(mn => VoronoiMesh("meshes/$mn.nc") for mn in ALL)
println("building stars ..."); flush(stdout)
STAR = Dict(mn => Dict("diagonal" => sparse(Diagonal(ω(MESH[mn]))),
                       "lift P1"  => hodge_star_lift_matrix(MESH[mn]; only_linear = true),
                       "lift P2"  => hodge_star_lift_matrix(MESH[mn]; only_linear = false)) for mn in ALL)
H_ = Dict(mn => meanh(MESH[mn]) for mn in ALL)

# ============================================================================
# 1. offsets
# ============================================================================
function offsets(m)
    δ = zeros(m.edges.n); Ld = zeros(m.edges.n)
    for e in 1:m.edges.n
        xa, xb = edge_endpoints(m, e, true); c1, c2 = edge_endpoints(m, e, false)
        t = (xb - xa) / norm(xb - xa); n = [-t[2], t[1]]
        δ[e] = dot((c1 + c2) / 2 - (xa + xb) / 2, n); Ld[e] = norm(c2 - c1)
    end
    return δ, Ld
end
println("\n== offsets: max and rms |δ|/|ẽ|")
for mn in ALL
    δ, Ld = offsets(MESH[mn]); r = abs.(δ) ./ Ld
    @printf("  %-14s max %.3e  rms %.3e\n", mn, maximum(r), sqrt(sum(r .^ 2) / length(r)))
end
let f = Figure(size = (1000, 470))
    dpts(m, k) = [Point2f(_pt(m.cells.position[c])...) for c in 1:m.cells.n if length(m.cells.edges[c]) == k]
    for (i, (mn, ttl)) in enumerate([("mesh3_regular", "(a) regular lattice: δₑ = 0"), ("mesh3", "(b) SCVT mesh: δₑ ≠ 0 near the defects")])
        m = MESH[mn]; δ, Ld = offsets(m); r = δ ./ Ld
        ax = Axis(f[1, i], aspect = DataAspect(), title = ttl)
        hidedecorations!(ax); hidespines!(ax)
        segs = Point2f[]; cols = Float64[]
        for e in 1:m.edges.n
            a, b = edge_endpoints(m, e, true); push!(segs, Point2f(a...), Point2f(b...)); push!(cols, r[e], r[e])
        end
        linesegments!(ax, segs, color = cols, colormap = DIV, colorrange = (-0.22, 0.22), linewidth = 2.2)
        isempty(dpts(m, 5)) || scatter!(ax, dpts(m, 5), marker = :diamond, color = INK, markersize = 10, strokecolor = :white, strokewidth = 1)
        isempty(dpts(m, 7)) || scatter!(ax, dpts(m, 7), marker = :star5, color = INK, markersize = 11, strokecolor = :white, strokewidth = 1)
        xlims!(ax, 0, 1); ylims!(ax, 0, 1)
    end
    Colorbar(f[1, 3], colormap = DIV, limits = (-0.22, 0.22), label = "δₑ / |ẽ|")
    savefig("fig_lo_offset.png", f)
end

# ============================================================================
# 2. the skew pairing and the skew part of the lift
# ============================================================================
function offset_part(m, u, v)
    s = 0.0
    for e in 1:m.edges.n
        xa, xb = edge_endpoints(m, e, true); c1, c2 = edge_endpoints(m, e, false)
        L = norm(xb - xa); t = (xb - xa) / L; n = [-t[2], t[1]]; me = (xa + xb) / 2
        δ = dot((c1 + c2) / 2 - me, n)
        s += L * norm(c2 - c1) * δ * (dot(v(me), t) * dot(t, jac(u, me) * n) - dot(u(me), t) * dot(t, jac(v, me) * n))
    end
    return s
end
SK = Dict()
println("\n== s_h(U,V), its offset part, and the skew part of the lifts")
for mn in ALL
    m = MESH[mn]; d0 = d0m(m)
    s = s_h(m, U, V); off = offset_part(m, U, V)
    row = Dict{String, Any}("s" => s, "off" => off)
    for nm in ("lift P1", "lift P2")
        H = STAR[mn][nm]; K = skw(H)
        row[nm] = (KH = norm(K) / norm(H), LK = norm(d0' * K * d0) / norm(d0' * H * d0),
                   ratio = 2 * dot(α(m, V), K * α(m, U)) / s)
    end
    SK[mn] = row
    @printf("  %-14s s_h %+.2e  offset part %+.2e  rest %+.2e | P1: |K|/|H| %.2e |d0'Kd0|/|L| %.2e ratio %+.3f | P2: |K|/|H| %.2e |d0'Kd0|/|L| %.2e ratio %+.3f\n",
            mn, s, off, s - off, row["lift P1"].KH, row["lift P1"].LK, row["lift P1"].ratio, row["lift P2"].KH, row["lift P2"].LK, row["lift P2"].ratio)
end
let f = Figure(size = (1200, 400))
    hs = [H_[mn] for mn in last(FAMS[2])]; mns = last(FAMS[2]); hr = [H_[mn] for mn in last(FAMS[1])]
    ax1 = Axis(f[1, 1], xscale = log10, yscale = log10, xreversed = true, xlabel = "h", title = "(a) the skew pairing s_h(U,V)")
    scatterlines!(ax1, hs, [abs(SK[mn]["s"]) for mn in mns], color = INK, linewidth = 2.5, label = "SCVT: |s_h|")
    scatterlines!(ax1, hs, [abs(SK[mn]["off"]) for mn in mns], color = ORANGE, linewidth = 2, linestyle = :dash, label = "SCVT: offset part")
    scatterlines!(ax1, hs, [abs(SK[mn]["s"] - SK[mn]["off"]) for mn in mns], color = AQUA, linewidth = 2, linestyle = :dot, label = "SCVT: second-moment part")
    scatterlines!(ax1, hr, [abs(SK[mn]["s"]) for mn in last(FAMS[1])], color = MUTED, linewidth = 2, marker = :circle, label = "lattice: |s_h|")
    lines!(ax1, hs, 0.6 * abs(SK["mesh2"]["s"]) .* (hs ./ hs[1]) .^ 2, color = MUTED, linestyle = :dashdot, linewidth = 1, label = "slope 2")
    axislegend(ax1, position = :lc, labelsize = 10)
    ax2 = Axis(f[1, 2], xscale = log10, yscale = log10, xreversed = true, xlabel = "h", title = "(b) skew part of the lift, alone and seen by L")
    for (fam, mns_) in FAMS, nm in ("lift P1", "lift P2")
        hh = [H_[mn] for mn in mns_]
        scatterlines!(ax2, hh, [SK[mn][nm].LK for mn in mns_], color = COL[nm], linestyle = LS[fam], marker = MK[fam], linewidth = 2,
                      label = "$fam, $nm: ‖d₀ᵀKd₀‖/‖L‖")
        fam == "SCVT" && lines!(ax2, hh, [SK[mn][nm].KH for mn in mns_], color = (COL[nm], 0.45), linewidth = 4, label = "$nm: ‖K‖/‖H‖ (both)")
    end
    Legend(f[1, 3], ax2, labelsize = 10)
    ax3 = Axis(f[1, 4], xscale = log10, xreversed = true, xlabel = "h", ylabel = "2α(V)ᵀKα(U) / s_h(U,V)",
               title = "(c) SCVT: the lift's skew carries s_h")
    for nm in ("lift P1", "lift P2")
        scatterlines!(ax3, hs, [SK[mn][nm].ratio for mn in mns], color = COL[nm], linewidth = 2, label = nm)
    end
    hlines!(ax3, [1.0], color = INK, linestyle = :dot)
    axislegend(ax3, position = :rc)
    savefig("fig_lo_skew.png", f)
end

# ============================================================================
# 3. consistency: Hodge error, Laplacian truncation, Poisson, Darcy
# ============================================================================
function truncation(m, H)
    d0 = d0m(m); p = [pex(_pt(m.cells.position[c])) for c in 1:m.cells.n]
    avg = discretize_scalar_field_cells(m, lapp).values ./ m.cells.area
    τ = (d0' * (H * (d0 * p))) ./ m.cells.area .+ avg
    τε = (d0' * ε(H, m, gpex)) ./ m.cells.area
    return maximum(abs, τ) / maximum(abs, avg), maximum(abs, τ .- τε) / maximum(abs, avg)
end
function poisson_error(m, H)
    ph = solve_poisson_primal(m, H, x -> -lapp(x))
    pe = [pex(_pt(m.cells.position[c])) for c in 1:m.cells.n]; pe .-= sum(pe) / length(pe)
    A = m.cells.area
    return sqrt(sum(A .* (ph .- pe) .^ 2) / sum(A .* pe .^ 2))
end
function darcy(m, H)
    D1 = D1m(m); nE, nT = m.edges.n, m.vertices.n
    φ = discretize_scalar_field_triangles_ho(m, x -> -lapp(x)).values; φ .-= sum(φ) / nT
    Kmat = [sparse(H) -sparse(D1'); D1 spzeros(nT, nT)]
    keep = setdiff(1:nE+nT, [nE + 1]); sol = zeros(nE + nT)
    sol[keep] = Kmat[keep, keep] \ [zeros(nE); φ][keep]
    f, p = sol[1:nE], sol[nE+1:end]
    fex = α(m, uex); A = m.vertices.area
    pT = [pex(_pt(m.vertices.position[t])) for t in 1:nT]
    pp = p .- sum(A .* p) / sum(A); pe = pT .- sum(A .* pT) / sum(A)
    return normW(m, f - fex) / normW(m, fex), sqrt(sum(A .* (pp .- pe) .^ 2) / sum(A .* pe .^ 2))
end
CONS = Dict()
println("\n== consistency (relative): Hodge error on U | max|τ| | identity defect | Poisson | Darcy flux | Darcy pressure")
for mn in ALL, nm in NAMES
    m = MESH[mn]; H = STAR[mn][nm]
    τ, idd = truncation(m, H)
    fl, pr = darcy(m, H)
    CONS[(mn, nm)] = (hodge = normWinv(m, ε(H, m, U)) / normWinv(m, β(m, U)),
                      hodgeconst = normWinv(m, ε(H, m, x -> [1.0, 0.7])) / normWinv(m, β(m, x -> [1.0, 0.7])),
                      τ = τ, idd = idd, poisson = poisson_error(m, H), flux = fl, pres = pr)
    c = CONS[(mn, nm)]
    @printf("  %-14s %-9s %.2e (const %.0e) | %.2e | %.0e | %.2e | %.2e | %.2e\n", mn, nm, c.hodge, c.hodgeconst, c.τ, c.idd, c.poisson, c.flux, c.pres)
end
for (fam, mns) in FAMS, nm in NAMES
    hh = [H_[mn] for mn in mns]
    @printf("  rates %-8s %-9s hodge %s | τ %s | poisson %s | flux %s\n", fam, nm,
            join(round.(rates(hh, [CONS[(mn, nm)].hodge for mn in mns]), digits = 1), ","),
            join(round.(rates(hh, [CONS[(mn, nm)].τ for mn in mns]), digits = 1), ","),
            join(round.(rates(hh, [CONS[(mn, nm)].poisson for mn in mns]), digits = 1), ","),
            join(round.(rates(hh, [CONS[(mn, nm)].flux for mn in mns]), digits = 1), ","))
end
let f = Figure(size = (1100, 800))
    panels = [(:hodge, "(a) flux (Hodge) error ‖ε(U)‖/‖β(U)‖"), (:τ, "(b) Laplacian: max-norm truncation"),
              (:poisson, "(c) Poisson: solution error"), (:flux, "(d) Darcy: flux error")]
    for (k, (key, ttl)) in enumerate(panels)
        ax = Axis(f[(k - 1) ÷ 2 + 1, (k - 1) % 2 + 1], xscale = log10, yscale = log10, xreversed = true, xlabel = "h", title = ttl)
        for (fam, mns) in FAMS, nm in NAMES
            hh = [H_[mn] for mn in mns]
            scatterlines!(ax, hh, [getfield(CONS[(mn, nm)], key) for mn in mns], color = COL[nm], linestyle = LS[fam],
                          marker = MK[fam], linewidth = 2, label = "$nm ($fam)")
        end
        hh = [H_[mn] for mn in last(FAMS[2])]
        for (p, ls) in ((2, :dot), (4, :dashdot))
            lines!(ax, hh, 0.3 * (hh ./ hh[1]) .^ p, color = INK, linestyle = ls, linewidth = 1, label = "slope $p")
        end
        k == 1 && Legend(f[1:2, 3], ax, labelsize = 10)
    end
    savefig("fig_lo_consistency.png", f)
end

# ============================================================================
# 4. spectra
# ============================================================================
function spectrum(m, H)
    d0 = d0m(m); s = 1 ./ sqrt.(m.cells.area)
    A = Matrix(Diagonal(s) * (d0' * H * d0) * Diagonal(s))
    return eigvals(A)            # general solver on purpose: no symmetry is assumed
end
symgrad_min(m, H) = m.cells.n > 1100 ? NaN :
    (Q = Matrix(qr(Matrix(d0m(m))).Q)[:, 1:m.cells.n-1]; eigmin(Symmetric(Q' * Matrix((H + H') / 2) * Q)))
SPEC = Dict()
println("\n== spectra of L p = λ S0 p (general eigensolver)")
for mn in ALL, nm in NAMES
    m = MESH[mn]; λ = spectrum(m, STAR[mn][nm])
    pos = sort(filter(x -> real(x) > -1e-8, λ), by = real)
    low = pos[2:21]
    SPEC[(mn, nm)] = (λ = λ, nneg = count(x -> real(x) < -1e-8, λ), minre = minimum(real, λ),
                      imall = maximum(abs.(imag.(λ)) ./ max.(abs.(λ), 1e-12)),
                      imlow = maximum(abs.(imag.(low)) ./ abs.(low)), sg = symgrad_min(m, STAR[mn][nm]))
    r = SPEC[(mn, nm)]
    @printf("  %-14s %-9s N=%5d  #Re<0 %4d  min Re %+.3e  max|Im|/|λ| all %.2e  lowest 20 %.2e  min sym(H)|grad %s\n",
            mn, nm, m.cells.n, r.nneg, r.minre, r.imall, r.imlow, isnan(r.sg) ? "-" : @sprintf("%+.3f", r.sg))
end
let f = Figure(size = (1150, 820))
    ax1 = Axis(f[1, 1], xlabel = "Re λ", ylabel = "Im λ", title = "(a) lift P2, mesh4: lattice vs SCVT")
    λr = SPEC[("mesh4_regular", "lift P2")].λ; λs = SPEC[("mesh4", "lift P2")].λ
    scatter!(ax1, real.(λs), imag.(λs), color = (BLUE, 0.55), markersize = 5, label = "SCVT: complex")
    scatter!(ax1, real.(λr), imag.(λr), color = INK, marker = :vline, markersize = 9, label = "lattice: all real")
    axislegend(ax1, position = :lb)
    ax2 = Axis(f[1, 2], xlabel = "Re λ / max Re λ", ylabel = "|Im λ| / |λ|", title = "(b) lift P2, SCVT: where the imaginary parts are")
    for (mn, c) in (("mesh3", (BLUE, 0.35)), ("mesh4", (BLUE, 0.6)), ("mesh5", (INK, 0.5)))
        λ = SPEC[(mn, "lift P2")].λ; mx = maximum(real, λ)
        scatter!(ax2, real.(λ) ./ mx, abs.(imag.(λ)) ./ abs.(λ), color = c, markersize = 4, label = mn)
    end
    axislegend(ax2, position = :lt)
    ax3 = Axis(f[2, 1], xscale = log10, xreversed = true, xlabel = "h", ylabel = "(# eigenvalues with Re λ < 0) / N",
               title = "(c) lift P1: negative eigenvalues, a fixed fraction of N")
    ax4 = Axis(f[2, 2], xscale = log10, xreversed = true, xlabel = "h", ylabel = "min Re λ · h²",
               title = "(d) lift P1: the most negative eigenvalue grows like h⁻²")
    for (fam, mns) in FAMS
        hh = [H_[mn] for mn in mns]
        scatterlines!(ax3, hh, [SPEC[(mn, "lift P1")].nneg / MESH[mn].cells.n for mn in mns], color = ORANGE, linestyle = LS[fam], marker = MK[fam], linewidth = 2, label = fam)
        scatterlines!(ax4, hh, [SPEC[(mn, "lift P1")].minre * H_[mn]^2 for mn in mns], color = ORANGE, linestyle = LS[fam], marker = MK[fam], linewidth = 2, label = fam)
    end
    ylims!(ax3, 0, 0.3); axislegend(ax3, position = :rb); axislegend(ax4, position = :rt)
    savefig("fig_lo_spectrum.png", f)
end
println("done")
