# sw_steady_residuals.jl -- truncation errors of the three nonlinear terms of the TRiSK/DEC shallow-water model on the exact zonal
# geostrophic balance u = (U(y), 0), U = u0 sin(2πy/L), h = H + h0 cos(2πy/L), for H_DEC and H_P2.
#   continuity:  D0ᵀF/A              exact 0                         (mass-flux divergence; F = ĥ ⊙ Hv)
#   PV flux:     Q(F)_e              exact ∫_e (f + ζ) (Ju)·t ds = f∫U dy − ½[U²]   (ζ = −U')
#   KE gradient: (D0 K)_e            exact ½U(y_b)² − ½U(y_a)²
# Momentum residuals relative to max|D0(g h)|, the size of the balanced terms; continuity in m/day.
#      julia -t 4 scripts/shallow_water/sw_steady_residuals.jl   ->  ts_data/sw_steady_residuals.{txt,jls}
include(joinpath(@__DIR__, "sw_jet.jl"))
OUTR = String[]; sayr(s) = (println(s); flush(stdout); push!(OUTR, s))
RR = Dict{Any,Any}()
Uz(y) = U0J * sin(2π * y / LDOM)
sayr("Truncation errors on the exact zonal balance (max over edges/cells; momentum terms relative to max|D0 g h|)")
for mn in ("mesh4", "mesh5", "mesh6", "mesh7")
    mesh = VoronoiMesh("meshes/$mn.nc"); S = cached_stars(mn; mesh)
    for st in ("H_DEC", "H_P2")
        O = sw_ops(mesh, S[st]); x = steady_initial(O); h = x[1:O.nc]; v = x[O.nc+1:end]; L = O.L
        U = O.H * v; he = O.Phe * h; F = he .* U
        ζ = (O.D1 * v) ./ O.AT; hv = (O.Kite * h) ./ O.AT; q = (FJ .+ ζ) ./ hv; qe = O.Pqe * q
        Q = 0.5 .* (qe .* (O.W * F) .+ O.W * (qe .* F))
        K = O.Kc * (v .* U)
        Qex = zeros(O.ne); dKex = zeros(O.ne)
        for e in 1:O.ne
            p1, p2 = edge_endpoints(mesh, e, true); y1 = p1[2] * L; y2 = p2[2] * L
            Qex[e] = FJ * U0J * L / (2π) * (cos(2π * y1 / L) - cos(2π * y2 / L)) - 0.5 * (Uz(y2)^2 - Uz(y1)^2)
            dKex[e] = 0.5 * (Uz(y2)^2 - Uz(y1)^2)
        end
        scale = maximum(abs, O.D0 * (GJ .* h))
        rQ = maximum(abs, Q .- Qex) / scale; rK = maximum(abs, O.D0 * K .- dKex) / scale
        rQ2 = sqrt(sum(abs2, Q .- Qex) / O.ne) / scale; rK2 = sqrt(sum(abs2, O.D0 * K .- dKex) / O.ne) / scale
        rD = maximum(abs, (O.D0t * F) ./ O.A) * DAY; rD2 = sqrt(sum(O.A .* ((O.D0t * F) ./ O.A) .^ 2) / sum(O.A)) * DAY
        # the flux U = Hv alone, against the exact volume flux ∫_ẽ u·t_e
        function uflux(e)                     # ∫_ẽ u·t_e ds, exactly (u = (U(y), 0), ẽ straight)
            a, b = edge_endpoints(mesh, e, false); a = _p2(a) .* L; b = _p2(b) .* L
            p1, p2 = edge_endpoints(mesh, e, true); t = _p2(p2) .- _p2(p1); t = t ./ norm(t)
            dy = b[2] - a[2]; len = norm(b - a)
            return abs(dy) < 1e-9L ? Uz(a[2]) * t[1] * len : U0J * t[1] * len / dy * (L / 2π) * (cos(2π * a[2] / L) - cos(2π * b[2] / L))
        end
        Uex = [uflux(e) for e in 1:O.ne]
        rU = maximum(abs, U .- Uex) / maximum(abs, Uex)
        RR[(mn, st)] = (rQ = rQ, rK = rK, rQ2 = rQ2, rK2 = rK2, rD = rD, rD2 = rD2, rU = rU)
        sayr(@sprintf("%s %-6s  flux Hv max %.2e | mass-flux divergence max %.2e rms %.2e m/day | PV flux max %.2e rms %.2e | KE gradient max %.2e rms %.2e",
                      mn, st, rU, rD, rD2, rQ, rQ2, rK, rK2))
    end
end
serialize(joinpath(TSDIR, "sw_steady_residuals.jls"), RR)
write(joinpath(TSDIR, "sw_steady_residuals.txt"), join(OUTR, "\n") * "\n")
