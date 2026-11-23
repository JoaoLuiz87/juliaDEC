# ============================================================================
#  positivity_check.jl -- a cheap certificate that the symmetric [P1]^2 star is SPD (Section 4 of
#  commuting_symmetric_note.tex).
#
#      julia -t 4 scripts/stars/positivity_check.jl      (writes output/gs_data/positivity_check.txt)
#
#  H = Hd + C from symmetric_correction(mesh, 1, 3).  With D = Hd (the DEC diagonal) and S = D^-1/2 H D^-1/2,
#  Gershgorin's theorem gives   λmin(S) ≥ g := min_e ( S_ee - Σ_{j≠e} |S_ej| ),
#  so g > 0 certifies that H is SPD.  The exact λmin (bisection with sparse Cholesky) is printed for comparison.
# ============================================================================
ENV["PARTS"] = ""                                   # load the functions of commuting_check.jl only
include(joinpath(@__DIR__, "commuting_check.jl"))

"λmin(S) by bisection: λmin > t  <=>  S - tI has a Cholesky factorisation"
function lambda_min_bisect(S; lo = -1.0, hi = 1.0, tol = 1e-4)
    S = sparse(Symmetric(S))
    while hi - lo > tol
        t = (lo + hi) / 2
        issuccess(cholesky(S - t * I; check = false)) ? (lo = t) : (hi = t)
    end
    return (lo + hi) / 2
end

logp("Gershgorin certificate for the symmetric [P1]^2 star (3-ring), S = D^-1/2 H D^-1/2 with D = DEC")
logp("   g = min_e (S_ee - sum_{j != e} |S_ej|) <= lambda_min(S);  off = largest off-diagonal row sum;  lambda_min exact for comparison")
for mn in ("mesh2", "mesh3", "mesh4", "mesh5")
    mesh = VoronoiMesh("meshes/$mn.nc")
    H, info = symmetric_correction(mesh, 1, 3)
    Dm = Diagonal(1 ./ sqrt.(diag(hdiag(mesh)))); S = Dm * H * Dm
    rows = rowvals(S); vals = nonzeros(S); n = size(S, 1)
    g = Inf; offmax = 0.0; offmean = 0.0
    for e in 1:n
        s_ee = 0.0; off = 0.0
        for k in nzrange(S, e)
            rows[k] == e ? (s_ee = vals[k]) : (off += abs(vals[k]))
        end
        g = min(g, s_ee - off); offmax = max(offmax, off); offmean += off / n
    end
    logp(@sprintf("   %-6s  g = %.3f   lambda_min = %.3f   off: max %.3f  mean %.3f", mn, g, lambda_min_bisect(S), offmax, offmean))
end
write(joinpath(OUT, "positivity_check.txt"), join(lines, "\n") * "\n")
