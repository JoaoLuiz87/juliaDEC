# ss_appendixB_test.jl -- run the guide's Appendix B transcription (verbatim copy in ss_data/) on the porting mesh,
# first as printed, then with the share solve regularised; compare with SymStar.jl.
include(joinpath(@__DIR__, "..", "..", "SymStar.jl")); using .SymStar
include(joinpath(@__DIR__, "..", "..", "output", "ss_data", "appendixB_SymStar.jl"))
using LinearAlgebra, SparseArrays, Printf
include(joinpath(@__DIR__, "ss_mesh.jl"))
X, T, period, h = porting_mesh()
g = edge_geometry(X, T, period)
println("max|δ|/|e| = ", round(maximum(abs.(g.delta) ./ g.le), digits = 4), "  (guide Table 5: 0.044)")
try
    SymStarB.symmetric_star(g.me, g.t, g.le, g.lde, g.delta, period, h; rho = 2.0)
    println("Appendix B as printed: ran")
catch err
    println("Appendix B as printed: ", typeof(err), " in the share solve")
end
# the one-line fix: regularise the singular share matrix
@eval SymStarB function solve_shares(pats, E; tol = 1e-11)
    Is = Int[]; Js = Int[]; Gs = Float64[]; Vs = Matrix{Float64}[]
    for (P, N, R) in pats
        F = svd(skew_map(N, R)); s = F.S
        r = (isempty(s) || s[1] == 0) ? 0 : count(>(tol * s[1]), s)
        V = F.V[:, 1:r]; push!(Vs, V); Pi0 = I - V * V'
        for (b, f) in enumerate(P), (a, e) in enumerate(P)
            push!(Is, e); push!(Js, f); push!(Gs, Pi0[a, b])
        end
    end
    G = sparse(Is, Js, Gs, E, E)
    lam = cholesky(Symmetric(G + 1e-13 * maximum(diag(G)) * I)) \ ones(E)      # <- the fix
    thetas = [lam[P] - V * (V' * lam[P]) for ((P, _, _), V) in zip(pats, Vs)]
    part = zeros(E)
    for ((P, _, _), th) in zip(pats, thetas)
        part[P] .+= th
    end
    return thetas, maximum(abs.(part .- 1))
end
HB, infoB = SymStarB.symmetric_star(g.me, g.t, g.le, g.lde, g.delta, period, h; rho = 2.0)
H, info = symmetric_star(g.me, g.t, g.le, g.lde, g.delta, period, h; rho = 2.0, verbose = false)
@printf("Appendix B with the fix: partition err %.1e, affine check %.1e, Σθ² %.9f (Table 5: 49.801435), tr H %.6f (741.544716)\n",
        infoB.partition_err, SymStarB.check_affine(HB, g.me, g.t, g.le, g.lde, g.delta, period, h), infoB.theta_norm2, tr(HB))
@printf("difference to SymStar.jl: ‖H_B - H‖_max = %.1e\n", maximum(abs.(HB - H)))
