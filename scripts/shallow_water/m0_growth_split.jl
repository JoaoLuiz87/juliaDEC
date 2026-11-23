# ============================================================================
#  m0_growth_split.jl -- which operator makes the consistent linear shallow-water scheme grow: P, M0ᶜ, or the star?
#  Dense eigenvalues of the linear operator  [0  H0 M0⁻¹D0ᵀH ; −gD0  −f C]  on meshes 3 and 4, for the three stars and
#      C  in { W H (TRiSK), P, Pa }       Pa = H_s⁻¹ · skew(H_s P): the part of P that does no work (energy-neutral Coriolis)
#      M0 in { A (areas), M0ᶜ, M0ˢ }      M0ˢ = (M0ᶜ + M0ᶜᵀ)/2
#
#      julia -t 4 scripts/shallow_water/m0_growth_split.jl        -> output/ts_data/m0_growth_split.txt
# ============================================================================
include(joinpath(@__DIR__, "sw_linear_three.jl"))
OUTG = String[]; sg(s) = (println(s); flush(stdout); push!(OUTG, s))
day(r) = r > 1e-12 ? @sprintf("%6.1f d", 1 / r / 86400) : "   none "
for mn in ("mesh3", "mesh4")
    mesh = VoronoiMesh("meshes/$mn.nc"); S = cached_stars(mn; mesh)
    O0 = sw_ops(mesh, S["H_DEC"]); P = Matrix(perp_p2(mesh, O0.W * O0.H)); M0c = Matrix(mass0_corrected(mesh)) .* O0.L^2
    M0s = (M0c + M0c') ./ 2
    sg(@sprintf("%s  ‖M0c − M0cᵀ‖/‖M0c‖ = %.2e   min eig(M0s)/min area = %.3f", mn, norm(M0c - M0c') / norm(M0c), minimum(eigvals(Symmetric(M0s))) / minimum(O0.A)))
    for st in STARS3
        O = sw_ops(mesh, S[st]); H = Matrix(O.H); Hs = (H + H') ./ 2; W = Matrix(O.W * O.H)
        G = Hs * P; Pa = Hs \ ((G - G') ./ 2)
        sg(@sprintf("%s %-5s  ‖sym(H_s P)‖/‖H_s P‖ = %.3f", mn, st, norm(G + G') / 2 / norm(G)))
        for (mlab, M) in (("A   ", Diagonal(O.A)), ("M0c ", M0c), ("M0s ", M0s))
            B = H0J .* (M \ Matrix(O.D0t * O.H))
            line = @sprintf("   M0 = %s", mlab)
            for (clab, C) in (("W H", W), ("P", P), ("Pa", Pa))
                λ = eigvals([zeros(O.nc, O.nc) B; -GJ .* Matrix(O.D0) -FJ .* C]); r, k = findmax(real.(λ))
                line *= @sprintf("   C = %-3s %+.2e s⁻¹ %s (|Im|/f = %.2f)", clab, r, day(r), abs(imag(λ[k])) / FJ)
            end
            sg(line)
        end
    end
end
write(joinpath(TSDIR, "m0_growth_split.txt"), join(OUTG, "\n") * "\n")
