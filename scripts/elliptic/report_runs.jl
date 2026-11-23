# report_runs.jl -- runs for hodge_report.tex: the velocity (flux) of the Poisson problem, F = H D0 p_h against R̃⋆∇p, next to the nodal errors, meshes 1-7,
# three stars; and the largest entries of the stars.     julia -t 4 scripts/elliptic/report_runs.jl -> ts_data/report_runs.{jls,txt}
ENV["PART"] = "none"; include(joinpath(@__DIR__, "..", "stars", "stars_cache.jl")); using Serialization
src = read(joinpath(@__DIR__, "paper_runs.jl"), String); i = findfirst("function getstars", src)[1]; j = findfirst("# ============================================================================ E:", src)[1]
const TPI = 2π; isdefined(Main, :TP) || (const TP = 2π)
include_string(Main, src[i:j-1])
gradP1(x) = [TP * cos(TP * x[1]) * sin(TP * x[2]) - TP * sin(2TP * x[1]), TP * sin(TP * x[1]) * cos(TP * x[2])]
gradP3(x) = [TP * cos(TP * x[1]) * cos(2TP * x[2]) + 0.9TP * cos(3TP * (x[1] + x[2])), -2TP * sin(TP * x[1]) * sin(2TP * x[2]) + 0.9TP * cos(3TP * (x[1] + x[2]))]
R = Dict{Any,Any}(); OUTRR = String[]; sayr(s) = (println(s); flush(stdout); push!(OUTRR, s))
for mn in ["mesh$i" for i in 1:7]
    mesh = VoronoiMesh("meshes/$mn.nc"); S = getstars(mn, mesh); A = mesh.cells.area; ld = edge_lengths(mesh, false); lp = edge_lengths(mesh, true); D0 = d0mat(mesh)
    pts = [_pt(x) for x in mesh.cells.position]; R[(mn, :h)] = meshsize(mesh)
    for (k, T, gr) in ((1, PTESTS[1], gradP1), (3, PTESTS[3], gradP3))
        pe = [T.p(x) for x in pts]; pe .-= sum(A .* pe) / sum(A); Fex = reduce_dual_edges(mesh, star_field(gr)).values; vn = maximum(norm(gr([a, b])) for a in 0:0.01:1, b in 0:0.01:1)
        for st in ("H_DEC", "H_P1", "H_P2")
            ph = solve_poisson_primal(mesh, S[st], x -> -T.lap(x)); ph .-= sum(A .* ph) / sum(A); F = S[st] * (D0 * ph); e = F .- Fex
            r = (inf = maximum(abs, ph .- pe) / maximum(abs, pe), l2 = sqrt(sum(A .* (ph .- pe) .^ 2) / sum(A .* pe .^ 2)), finf = maximum(abs.(e) ./ ld) / vn,
                 f2 = sqrt(sum(lp ./ ld .* e .^ 2) / sum(lp ./ ld .* Fex .^ 2)), ph = mn in ("mesh4",) ? ph : nothing, pe = mn in ("mesh4",) ? pe : nothing, fe = mn in ("mesh4",) ? abs.(e) ./ ld ./ vn : nothing)
            R[(mn, :P, k, st)] = r
            sayr(@sprintf("%s P%d %-5s  p max %.2e  p L2 %.2e   flux max %.2e  flux L2 %.2e", mn, k, st, r.inf, r.l2, r.finf, r.f2))
        end
    end
    hd = diag(hdiag(mesh))
    for st in ("H_P1", "H_P2")
        H = S[st]; m = 0.0; rv = rowvals(H); nz = nonzeros(H)
        for c in 1:size(H, 2), p in nzrange(H, c); rv[p] == c || (m = max(m, abs(nz[p]))); end
        dm = maximum(abs.(diag(H) .- hd)); R[(mn, :entry, st)] = (off = m / maximum(hd), diag = dm / maximum(hd), Λ = bigLambda(mesh, H))
        sayr(@sprintf("%s %-5s largest off-diagonal entry / max H_DEC %.4f   largest change of the diagonal %.4f   Λ %.3f", mn, st, m / maximum(hd), dm / maximum(hd), bigLambda(mesh, H)))
    end
    serialize(joinpath(TSDIR, "report_runs.jls"), R); write(joinpath(TSDIR, "report_runs.txt"), join(OUTRR, "\n") * "\n")
end
