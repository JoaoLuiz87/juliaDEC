# ============================================================================
#  lie_tables.jl -- LaTeX tables of lie_derivative_note.tex from the result files
#
#      julia scripts/transport/lie_tables.jl      ->  output/ts_data/lie_tables.tex
#  (needs ts_data/lie_transport.jls, lie_testcases.jls, lie_spectra.jls)
# ============================================================================
using Serialization, Printf
const TSD = joinpath(@__DIR__, "..", "..", "output", "ts_data")
T = deserialize(joinpath(TSD, "lie_transport.jls"))      # free stream, truncation (lie_transport_check.jl)
C = deserialize(joinpath(TSD, "lie_testcases.jls"))      # five tests (lie_testcases_check.jl)
S = deserialize(joinpath(TSD, "lie_spectra.jls"))        # spectra (lie_spectra_check.jl)
function e(x)
    parts = split(@sprintf("%.1e", x), "e")
    return "\$" * parts[1] * "\\cdot10^{" * string(parse(Int, parts[2])) * "}\$"
end
o(x) = abs(x) < 0.005 ? "0.00" : @sprintf("%.2f", x)
slope(v, h) = (x = log.(h); y = log.(v); xm = sum(x) / length(x); ym = sum(y) / length(y); sum((x .- xm) .* (y .- ym)) / sum((x .- xm) .^ 2))
nm(s, v) = (s == "H_DEC" && v == "TRiSK") ? "\\Hd\\ (standard DEC)" :
           (s == "exact" ? "exact \$U\$" : s == "H_DEC" ? "\\Hd" : s == "H_P1" ? "\\Hp" : "\\Hq") * (s == "H_DEC" && v == "TRiSK" ? "" : ", $v")
io = IOBuffer()

# ---- free stream
ms = ("mesh4", "mesh5", "mesh6", "mesh7"); hs = [T[(m, :h)] for m in ms]
println(io, "% free stream\n\\newcommand{\\TabFreeStream}{\\begin{tabular}{lrrrrc}\\toprule")
println(io, "star & \$h=1/32\$ & \$1/64\$ & \$1/128\$ & \$1/256\$ & order (fit) \\\\ \\midrule")
for (s, l) in (("H_DEC", "\\Hd"), ("H_P1", "\\Hp"), ("H_P2", "\\Hq"))
    v = [T[(m, s, :free)].inf for m in ms]
    println(io, l, " & ", join(e.(v), " & "), " & ", o(slope(v[2:end], hs[2:end])), " \\\\")
end
println(io, "\\bottomrule\\end{tabular}}")

# ---- truncation of the density Lie derivative
println(io, "% truncation\n\\newcommand{\\TabTrunc}{\\begin{tabular}{lrrrc|rc}\\toprule")
println(io, " & \\multicolumn{4}{c|}{\$\\max|\\tau|\$ (relative)} & \\multicolumn{2}{c}{\$\\ell_2\$} \\\\ scheme & \$1/64\$ & \$1/128\$ & \$1/256\$ & order & \$1/256\$ & order \\\\ \\midrule")
isfile(joinpath(TSD, "lie_upw_trunc.jls")) && merge!(T, deserialize(joinpath(TSD, "lie_upw_trunc.jls")))
for (s, v) in (("H_DEC", "TRiSK"), ("H_P1", "TRiSK"), ("H_P2", "TRiSK"), ("exact", "TRiSK"), ("H_P1", "DEC"), ("H_P2", "DEC"), ("H_P2", "Ham"),
               ("H_P2", "Peixoto"), ("H_P2", "CFV2"), ("exact", "CFV2"), ("H_DEC", "UPW2"), ("H_P1", "UPW2"), ("H_P2", "UPW2"), ("exact", "UPW2"))
    haskey(T, ("mesh7", s, v, :trunc)) || continue
    a = [T[(m, s, v, :trunc)].inf for m in ("mesh5", "mesh6", "mesh7")]; b = [T[(m, s, v, :trunc)].l2 for m in ("mesh6", "mesh7")]
    println(io, nm(s, v), " & ", join(e.(a), " & "), " & ", o(log(a[2] / a[3]) / log(2)), " & ", e(b[2]), " & ", o(log(b[1] / b[2]) / log(2)), " \\\\")
end
println(io, "\\bottomrule\\end{tabular}}")

# ---- spectra
println(io, "% spectra\n\\newcommand{\\TabSpectra}{\\begin{tabular}{lrrr}\\toprule")
println(io, "scheme & translation & swirl & cellular \\\\ \\midrule")
for (s, v) in (("exact", "TRiSK"), ("H_DEC", "TRiSK"), ("H_P1", "TRiSK"), ("H_P2", "TRiSK"), ("H_P1", "DEC"), ("H_P2", "DEC"), ("H_P2", "Ham"),
               ("H_P2", "Peixoto"), ("H_P2", "CFV2"), ("H_DEC", "UPW2"), ("H_P2", "UPW2"))
    cells = String[]
    for f in ("translation", "swirl", "cellular")
        r = S[("mesh5", f, s, v)]
        x = abs(r.maxre) < 1e-12 ? "\$<10^{-12}\$" : e(r.maxre)
        v == "TRiSK" && !isnan(r.bound) && (x *= " [" * (abs(r.bound) < 1e-12 ? "0" : e(r.bound)) * "]")
        push!(cells, x)
    end
    println(io, nm(s, v), " & ", join(cells, " & "), " \\\\")
end
println(io, "\\bottomrule\\end{tabular}}")

# ---- the five tests: mesh7 errors and fitted orders (meshes 5-7)
mt = ("mesh5", "mesh6", "mesh7"); ht = [C[(m, :h)] for m in mt]
tests = ("translation", "cellular", "deformational", "divergent", "cosine_bells")
schemes = (("H_DEC", "TRiSK"), ("H_P2", "TRiSK"), ("H_P2", "DEC"), ("H_P2", "Peixoto"), ("H_P2", "CFV2"),
           ("H_DEC", "UPW2"), ("H_P1", "UPW2"), ("H_P2", "UPW2"), ("exact", "UPW2"))
for (key, nrm, lab) in (("TabTestsTwo", :l2, "\\ell_2"), ("TabTestsInf", :linf, "\\ell_\\infty"))
    println(io, "% tests $nrm\n\\newcommand{\\$key}{\\begin{tabular}{l" * "rc"^length(tests) * "}\\toprule")
    println(io, "scheme & " * join(["\\multicolumn{2}{c}{$(replace(t, "_" => " "))}" for t in tests], " & ") * " \\\\")
    println(io, " & " * join(fill("\$$lab\$ & ord.", length(tests)), " & ") * " \\\\ \\midrule")
    for (s, v) in schemes
        row = String[]
        for t in tests
            vals = [getfield(C[(m, t, s, v)], nrm) for m in mt]
            push!(row, e(vals[end])); push!(row, o(slope(vals, ht)))
        end
        println(io, nm(s, v), " & ", join(row, " & "), " \\\\")
    end
    println(io, "\\bottomrule\\end{tabular}}")
end
# ---- symmetry experiment: H_P1, H_P2, sym(H_P2)
Y = deserialize(joinpath(TSD, "lie_symmetry.jls"))
println(io, "% symmetry\n\\newcommand{\\TabSymmetry}{\\begin{tabular}{lrrrrrrr}\\toprule")
println(io, " & \\multicolumn{2}{c}{free stream \$\\max|\\tilde DHv/A|\$} & \\multicolumn{3}{c}{\$\\max\\mathrm{Re}\\,\\lambda\$ (mesh 5, cellular)} & \\multicolumn{2}{c}{deformational, \$\\ell_\\infty\$ (mesh 7)} \\\\")
println(io, "star & mesh 5 & mesh 7 & TRiSK & DEC & UPW2 & UPW2 & DEC \\\\ \\midrule")
for (s, l) in (("H_P1", "\\Hp\\ (symmetric)"), ("H_P2", "\\Hq\\ (non-symmetric)"), ("H_sym", "\$\\tfrac12(\\Hq+\\Hq^T)\$"))
    g(x) = abs(x) < 1e-12 ? "\$<10^{-12}\$" : e(x)
    println(io, l, " & ", e(Y[("mesh5", s, :free)]), " & ", e(Y[("mesh7", s, :free)]), " & ", g(Y[("mesh5", "cellular", s, "TRiSK")]), " & ",
            g(Y[("mesh5", "cellular", s, "DEC")]), " & ", g(Y[("mesh5", "cellular", s, "UPW2")]), " & ", e(Y[("mesh7", :deform, s, "UPW2")].linf),
            " & ", e(Y[("mesh7", :deform, s, "DEC")].linf), " \\\\")
end
println(io, "\\bottomrule\\end{tabular}}")
# ---- extrusion-based contractions (extrusion_check.jl)
if isfile(joinpath(TSD, "extrusion.jls"))
    X = deserialize(joinpath(TSD, "extrusion.jls"))
    xm = [m for m in ("mesh5", "mesh6", "mesh7") if haskey(X, (m, "H_P2", :trunc))]
    println(io, "% extrusion truncation\n\\newcommand{\\TabExtTrunc}{\\begin{tabular}{l" * "r"^length(xm) * "c|rc}\\toprule")
    println(io, " & \\multicolumn{$(length(xm)+1)}{c|}{\$\\max|\\tau|\$ (relative)} & \\multicolumn{2}{c}{\$\\ell_2\$} \\\\ scheme & " *
                join(["mesh " * m[end:end] for m in xm], " & ") * " & order & mesh $(xm[end][end]) & order \\\\ \\midrule")
    for (key, l) in (("H_DEC", "EXT, \\Hd"), ("H_P1", "EXT, \\Hp"), ("H_P2", "EXT, \\Hq"), ("H_rec", "EXT, \$H_{\\mathrm{rec}}\$"), ("exact", "EXT, exact \$u\$"),
                     (("EXT0", "rec"), "EXT0, \$u^\\sharp\$"), (("EXT0", "exact"), "EXT0, exact \$u\$"))
        k(m) = key isa Tuple ? (m, key[1], key[2], :trunc) : (m, key, :trunc)
        haskey(X, k(xm[end])) || continue
        a = [X[k(m)].inf for m in xm]; b = [X[k(m)].l2 for m in xm]
        println(io, l, " & ", join(e.(a), " & "), " & ", o(log(a[end-1] / a[end]) / log(2)), " & ", e(b[end]), " & ", o(log(b[end-1] / b[end]) / log(2)), " \\\\")
    end
    println(io, "\\bottomrule\\end{tabular}}")
    if haskey(X, ("mesh5", "swirl", "H_P2", :spec))
        println(io, "% extrusion spectra\n\\newcommand{\\TabExtSpectra}{\\begin{tabular}{lrrrrrr}\\toprule")
        println(io, " & \\multicolumn{2}{c}{translation} & \\multicolumn{2}{c}{swirl} & \\multicolumn{2}{c}{cellular} \\\\ scheme & mesh 4 & mesh 5 & mesh 4 & mesh 5 & mesh 4 & mesh 5 \\\\ \\midrule")
        for (s, l) in (("H_DEC", "EXT, \\Hd"), ("H_P2", "EXT, \\Hq"), ("H_rec", "EXT, \$H_{\\mathrm{rec}}\$"), ("EXT0", "EXT0, \$u^\\sharp\$"))
            g(x) = abs(x) < 1e-12 ? "\$<10^{-12}\$" : e(x)
            println(io, l, " & ", join([g(X[(m, f, s, :spec)]) for f in ("translation", "swirl", "cellular") for m in ("mesh4", "mesh5")], " & "), " \\\\")
        end
        println(io, "\\bottomrule\\end{tabular}}")
    end
    mt2 = [m for m in ("mesh5", "mesh6", "mesh7") if haskey(X, (m, "deformational", "H_P2", :ext))]
    if length(mt2) >= 2
        ht2 = [X[(m, :h)] for m in mt2]
        println(io, "% extrusion tests\n\\newcommand{\\TabExtTests}{\\begin{tabular}{l" * "rc"^length(tests) * "}\\toprule")
        println(io, "scheme & " * join(["\\multicolumn{2}{c}{$(replace(t, "_" => " "))}" for t in tests], " & ") * " \\\\")
        println(io, " & " * join(fill("\$\\ell_\\infty\$ & ord.", length(tests)), " & ") * " \\\\ \\midrule")
        rowsX = (("UPW2, \\Hq\\ (straight extrusion)", m -> (C, (m, "", "H_P2", "UPW2"))), ("EXT, \\Hd", m -> (X, (m, "", "H_DEC", :ext))),
                 ("EXT, \\Hq", m -> (X, (m, "", "H_P2", :ext))), ("EXT, \$H_{\\mathrm{rec}}\$", m -> (X, (m, "", "H_rec", :ext))),
                 ("EXT, exact \$u\$", m -> (X, (m, "", "exact", :ext))), ("EXT0 (0-form), \$u^\\sharp\$", m -> (X, (m, "", "EXT0", "rec"))))
        for (l, src) in rowsX
            row = String[]
            for t in tests
                vals = map(mt2) do m
                    D, k = src(m); kk = (k[1], t, k[3:end]...)
                    D[kk].linf
                end
                push!(row, e(vals[end])); push!(row, o(slope(vals, ht2)))
            end
            println(io, l, " & ", join(row, " & "), " \\\\")
        end
        println(io, "\\bottomrule\\end{tabular}}")
    end
end
# ---- 1-form Lie derivative (extrusion_oneform_check.jl)
if isfile(joinpath(TSD, "extrusion_oneform.jls"))
    O = deserialize(joinpath(TSD, "extrusion_oneform.jls"))
    om = [m for m in ("mesh5", "mesh6", "mesh7") if haskey(O, (m, "rec", "total", :trunc))]
    println(io, "% 1-form truncation\n\\newcommand{\\TabOneTrunc}{\\begin{tabular}{l" * "r"^length(om) * "c|rc}\\toprule")
    println(io, " & \\multicolumn{$(length(om)+1)}{c|}{\$\\max|\\tau|\$ (relative)} & \\multicolumn{2}{c}{\$\\ell_2\$} \\\\ part & " *
                join(["mesh " * m[end:end] for m in om], " & ") * " & order & mesh $(om[end][end]) & order \\\\ \\midrule")
    for (vel, part, l) in (("rec", "total", "\$L_1b\$, \$u^\\sharp\$"), ("exact", "total", "\$L_1b\$, exact \$u\$"),
                           ("rec", "vort", "vorticity part \$I_2(u,D_1b)\$"), ("rec", "grad", "gradient part \$D_0I_1(u,b)\$"))
        haskey(O, (om[end], vel, part, :trunc)) || continue
        a = [O[(m, vel, part, :trunc)].inf for m in om]; b = [O[(m, vel, part, :trunc)].l2 for m in om]
        println(io, l, " & ", join(e.(a), " & "), " & ", o(log(a[end-1] / a[end]) / log(2)), " & ", e(b[end]), " & ", o(log(b[end-1] / b[end]) / log(2)), " \\\\")
    end
    println(io, "\\bottomrule\\end{tabular}}")
    if haskey(O, ("mesh5", "swirl", "L2", :spec))
        g(x) = isnan(x) ? "--" : abs(x) < 1e-12 ? "\$<10^{-12}\$" : e(x)
        println(io, "% 1-form spectra\n\\newcommand{\\TabOneSpectra}{\\begin{tabular}{lrrrrrr}\\toprule")
        println(io, " & \\multicolumn{2}{c}{translation} & \\multicolumn{2}{c}{swirl} & \\multicolumn{2}{c}{cellular} \\\\ operator & mesh 4 & mesh 5 & mesh 4 & mesh 5 & mesh 4 & mesh 5 \\\\ \\midrule")
        for (k, l) in (("L2", "\$L_2\$ (vorticity), mean-preserving fit"), ("L2plain", "\$L_2\$, plain least-squares fit"), ("L0", "\$L_0\$ (exact part) \$=\$ EXT0"))
            println(io, l, " & ", join([g(O[(m, f, k, :spec)]) for f in ("translation", "swirl", "cellular") for m in ("mesh4", "mesh5")], " & "), " \\\\")
        end
        println(io, "\\bottomrule\\end{tabular}}")
    end
    mo = [m for m in ("mesh5", "mesh6", "mesh7") if haskey(O, (m, "deformational", "rec", :oneform))]
    if length(mo) >= 2
        ho = [O[(m, :h)] for m in mo]
        println(io, "% 1-form tests\n\\newcommand{\\TabOneTests}{\\begin{tabular}{l" * "rc"^length(tests) * "}\\toprule")
        println(io, "quantity & " * join(["\\multicolumn{2}{c}{$(replace(t, "_" => " "))}" for t in tests], " & ") * " \\\\")
        println(io, " & " * join(fill("error & ord.", length(tests)), " & ") * " \\\\ \\midrule")
        X0 = isfile(joinpath(TSD, "extrusion.jls")) ? deserialize(joinpath(TSD, "extrusion.jls")) : nothing
        rowsO = Any[("1-form \$b/\\abs e\$, \$\\ell_\\infty\$", m -> (O, (m, "", "rec", :oneform)), :linf),
                    ("1-form \$b/\\abs e\$, \$\\ell_2\$", m -> (O, (m, "", "rec", :oneform)), :l2),
                    ("vorticity \$\\bar\\zeta_T\$, \$\\ell_\\infty\$", m -> (O, (m, "", "rec", :oneform)), :zlinf),
                    ("vorticity \$\\bar\\zeta_T\$, \$\\ell_2\$", m -> (O, (m, "", "rec", :oneform)), :zl2)]
        X0 === nothing || push!(rowsO, ("density EXT, \\Hq, \$\\ell_\\infty\$ (Table~\\ref{tab:ext-tests})", m -> (X0, (m, "", "H_P2", :ext)), :linf))
        for (l, src, fld) in rowsO
            row = String[]
            for t in tests
                vals = map(mo) do m
                    D, k = src(m); getfield(D[(k[1], t, k[3:end]...)], fld)
                end
                push!(row, e(vals[end])); push!(row, o(slope(vals, ho)))
            end
            println(io, l, " & ", join(row, " & "), " \\\\")
        end
        println(io, "\\bottomrule\\end{tabular}}")
    end
end

# ---- mean-preserving cubic density reconstruction (extrusion_cubic_check.jl)
if isfile(joinpath(TSD, "extrusion_cubic.jls")) && isfile(joinpath(TSD, "extrusion.jls"))
    K = deserialize(joinpath(TSD, "extrusion_cubic.jls")); X2 = deserialize(joinpath(TSD, "extrusion.jls"))
    km = [m for m in ("mesh5", "mesh6", "mesh7") if haskey(K, (m, "H_P2", "EXT3", :trunc))]
    println(io, "% cubic truncation\n\\newcommand{\\TabCubicTrunc}{\\begin{tabular}{l" * "r"^length(km) * "c|rc}\\toprule")
    println(io, " & \\multicolumn{$(length(km)+1)}{c|}{\$\\max|\\tau|\$ (relative)} & \\multicolumn{2}{c}{\$\\ell_2\$} \\\\ scheme & " *
                join(["mesh " * m[end:end] for m in km], " & ") * " & order & mesh $(km[end][end]) & order \\\\ \\midrule")
    for (D, key, l) in ((X2, m -> (m, "H_P2", :trunc), "EXT2 (quadratic, 1 ring), \\Hq"), (K, m -> (m, "H_P2", "EXT3", :trunc), "EXT3 (cubic, 2 rings, mean-pres.), \\Hq"),
                        (K, m -> (m, "exact", "EXT3", :trunc), "EXT3, exact area"))
        a = [D[key(m)].inf for m in km]; b = [D[key(m)].l2 for m in km]
        println(io, l, " & ", join(e.(a), " & "), " & ", o(log(a[end-1] / a[end]) / log(2)), " & ", e(b[end]), " & ", o(log(b[end-1] / b[end]) / log(2)), " \\\\")
    end
    println(io, "\\bottomrule\\end{tabular}}")
    k6 = [m for m in ("mesh4", "mesh5", "mesh6") if haskey(K, (m, "deformational", "H_DEC", "EXT3"))]
    if length(k6) == 3 && haskey(K, ("mesh7", "cosine_bells", "H_P2", "EXT3"))
        hk = [K[(m, :h)] for m in k6]
        println(io, "% cubic tests\n\\newcommand{\\TabCubicTests}{\\begin{tabular}{l" * "rc"^length(tests) * "}\\toprule")
        println(io, "scheme & " * join(["\\multicolumn{2}{c}{$(replace(t, "_" => " "))}" for t in tests], " & ") * " \\\\")
        println(io, " & " * join(fill("\$\\ell_\\infty\$ & ord.", length(tests)), " & ") * " \\\\ \\midrule")
        rowsK = (("EXT2, \\Hq, mesh 6", m -> X2[(m[1], m[2], "H_P2", :ext)], k6), ("EXT3, \\Hd, mesh 6", m -> K[(m[1], m[2], "H_DEC", "EXT3")], k6),
                 ("EXT3, \\Hq, mesh 6", m -> K[(m[1], m[2], "H_P2", "EXT3")], k6), ("EXT3, exact area, mesh 6", m -> K[(m[1], m[2], "exact", "EXT3")], k6),
                 ("EXT2, \\Hq, mesh 7", m -> X2[(m[1], m[2], "H_P2", :ext)], ("mesh6", "mesh7")), ("EXT3, \\Hq, mesh 7", m -> K[(m[1], m[2], "H_P2", "EXT3")], ("mesh6", "mesh7")))
        for (l, get, ms_) in rowsK
            row = String[]
            for t in tests
                vals = [get((m, t)).linf for m in ms_]
                push!(row, e(vals[end])); push!(row, o(slope(vals, [K[(m, :h)] for m in ms_])))
            end
            println(io, l, " & ", join(row, " & "), " \\\\")
        end
        println(io, "\\bottomrule\\end{tabular}}")
    end
end

write(joinpath(TSD, "lie_tables.tex"), String(take!(io)))
println("wrote ts_data/lie_tables.tex")
