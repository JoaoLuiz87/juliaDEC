# ============================================================================
#  gap_skew_figure.jl -- Figure fig_gap_skew.png of paper/hodge_gap.tex (Section 3.6).
#
#      julia scripts/barrier/gap_skew_figure.jl
#
#  (a) SCVT meshes 2-6 and (c) perturbed lattices 1-6: the skew values of ts_data/skew_split.jls (skew_split_check.jl).
#  (b) EXACT lattices.  The files meshN_regular.nc used in skew_split_check.jl store an isosceles lattice only to about
#      1e-9, and that rounding produces a small skew of order h^2 (s/h^2 ~ 2e-7, the signature of a seam).  Here the
#      lattice is built from dyadic coordinates, which are exact in floating point:
#          x_ij = (i + (j mod 2)/2)/n,  y_ij = j/n,   n = 8, 16, 32, 64, 128   (n^2 cells, h = 1/n),
#      the centred rectangular lattice of Proposition N (isosceles triangles, angles 63.4, 63.4, 53.1 degrees: no four
#      cocircular generators), triangulated by delaunay_mesh (no Lloyd step).  Proposition lattice predicts s_h = 0 for
#      the band-limited test fields; the computed values are at the level of rounding.
#  Writes ts_data/skew_lattice_exact.{jls,txt} and fig_gap_skew.png.
# ============================================================================
include(joinpath(@__DIR__, "..", "stars", "three_stars.jl"))
using CairoMakie, Printf

const P(x) = sin(TP * x[1]) * sin(TP * x[2]) + 0.5cos(2TP * x[1])
const GP(x) = [TP * cos(TP * x[1]) * sin(TP * x[2]) - TP * sin(2TP * x[1]), TP * sin(TP * x[1]) * cos(TP * x[2])]
const Q(x) = cos(TP * x[1]) * cos(2TP * x[2]) + sin(TP * (x[1] + x[2]))
const GQ(x) = [-TP * sin(TP * x[1]) * cos(2TP * x[2]) + TP * cos(TP * (x[1] + x[2])),
               -2TP * cos(TP * x[1]) * sin(2TP * x[2]) + TP * cos(TP * (x[1] + x[2]))]
rot(f) = x -> (g = f(x); [-g[2], g[1]])
function skew_of(mesh, u, v)
    α(w) = reduce_primal_edges(mesh, w).values; β(w) = reduce_dual_edges(mesh, star_field(w)).values
    return dot(α(u), β(v)) - dot(α(v), β(u))
end
"max_e |m~_e - m_e| / h: zero on an exact lattice"
function max_offset(mesh)
    h = meshsize(mesh); m = 0.0
    for e in 1:mesh.edges.n
        xe = mesh.edges.position[e]
        a, b = edge_endpoints(mesh, e, true; about = xe); v1, v2 = edge_endpoints(mesh, e, false; about = xe)
        m = max(m, norm((a .+ b) ./ 2 .- (v1 .+ v2) ./ 2))
    end
    return m / h
end

const CACHE = joinpath(TSDIR, "skew_lattice_exact.jls")
LX = isfile(CACHE) ? deserialize(CACHE) : Dict{Any,Any}()
out = String["skew on EXACT centred rectangular lattices (dyadic coordinates), x = (i + (j mod 2)/2)/n, y = j/n; s/h^2 in brackets"]
for n in (8, 16, 32, 64, 128)
    key = ("lattice", "n$n")
    if !haskey(LX, key)
        xs = [(i + (j % 2) / 2) / n for j in 0:n-1 for i in 0:n-1]; ys = [j / n for j in 0:n-1 for i in 0:n-1]
        mesh = delaunay_mesh(xs, ys); h = meshsize(mesh)
        @assert mesh.cells.n == n^2 && mesh.edges.n == 3n^2
        LX[key] = (h = h, gg = skew_of(mesh, GP, GQ), cc = skew_of(mesh, rot(GP), rot(GQ)), gc = skew_of(mesh, GP, rot(GQ)),
                   cg = skew_of(mesh, rot(GP), GQ), ud = skew_of(mesh, U2, fields.d), off = max_offset(mesh))
        serialize(CACHE, LX)
    end
    r = LX[key]
    push!(out, @sprintf("   n = %3d  h = %.5f   grad-grad %+.2e (%+.1e)   curl-curl %+.2e (%+.1e)   grad-curl %+.2e   curl-grad %+.2e   (u2,d) %+.2e   max offset/h %.1e",
                        n, r.h, r.gg, r.gg / r.h^2, r.cc, r.cc / r.h^2, r.gc, r.cg, r.ud, r.off))
end
write(joinpath(TSDIR, "skew_lattice_exact.txt"), join(out, "\n") * "\n"); foreach(println, out)

const R = deserialize(joinpath(TSDIR, "skew_split.jls"))
for (k, v) in LX; R[k] = v; end
const KINDS = ((:gg, "∇p, ∇q", colorant"#2b6cb0", :circle), (:cc, "J∇p, J∇q", colorant"#c23b3a", :rect),
               (:gc, "∇p, J∇q", colorant"#2f8f4e", :diamond), (:cg, "J∇p, ∇q", colorant"#d08a1c", :utriangle),
               (:ud, "u₂, d", colorant"#4d4d4d", :xcross))
let f = Figure(size = (1450, 500))
    for (i, (fam, sel, title)) in enumerate((("SCVT", k -> true, "(a) SCVT meshes 2–6"),
                                             ("lattice", k -> startswith(k[2], "n"), "(b) exact lattices, n² cells"),
                                             ("perturbed", k -> true, "(c) perturbed lattices 1–6")))
        ks = sort([k for k in keys(R) if k[1] == fam && sel(k)], by = k -> -R[k].h)
        hs = [R[k].h for k in ks]
        ax = Axis(f[1, i]; xscale = log10, yscale = log10, xreversed = true, xlabel = "h", ylabel = "|s_h(u, v)|", title = title,
                  titlesize = 14, xticks = (hs, [@sprintf("%.3f", h) for h in hs]), xticklabelrotation = π / 5,
                  yticks = (10.0 .^ (-18:2:0), ["10⁻¹⁸", "10⁻¹⁶", "10⁻¹⁴", "10⁻¹²", "10⁻¹⁰", "10⁻⁸", "10⁻⁶", "10⁻⁴", "10⁻²", "1"]))
        for (fld, lab, col, mk) in KINDS
            scatterlines!(ax, hs, [max(abs(getfield(R[k], fld)), 1e-19) for k in ks]; color = col, marker = mk, markersize = 10, label = lab)
        end
        if fam == "lattice"
            ylims!(ax, 1e-19, 1e-1)
            text!(ax, hs[3], 3e-6; text = "exact lattice: s_h = 0 in exact arithmetic;\ncomputed values at the rounding level", align = (:center, :center), fontsize = 13)
        else
            y0 = fam == "SCVT" ? 2e-3 : 3e-3
            lines!(ax, [hs[2], hs[end]], y0 .* ([hs[2], hs[end]] ./ hs[2]) .^ 2; color = :black, linestyle = :dot)
            text!(ax, hs[end], y0 * (hs[end] / hs[2])^2; text = "  h²", fontsize = 13, align = (:left, :center))
        end
    end
    Legend(f[2, 1:3], [[LineElement(color = c), MarkerElement(color = c, marker = m, markersize = 10)] for (_, _, c, m) in KINDS],
           ["s_h(" * l * ")" for (_, l, _, _) in KINDS]; orientation = :horizontal, framevisible = false, labelsize = 13)
    CairoMakie.save(joinpath(@__DIR__, "..", "..", "output", "figures", "fig_gap_skew.png"), f; px_per_unit = 2)
end
