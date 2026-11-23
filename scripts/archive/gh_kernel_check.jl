# ============================================================================
#  gh_kernel_check.jl -- is the moment system solvable?  (Theorem "solvability")
#
#      julia scripts/archive/gh_kernel_check.jl
#
#  For triangle cells (1-ring stencils) and diamond cells (13-edge stencils), k = 0, 1, 2:
#   - dim ker G (G = A 𝕄⁻¹ Aᵀ, the Gram matrix of the moment conditions), from its dense spectrum (small meshes),
#     against the prediction 4·dim P_k (translation-invariant kernels κ(x-y));
#   - ‖G λ^κ‖ for the predicted kernel vectors;
#   - the obstruction: component of the right-hand side along the kernel vectors, by degree of κ
#     (predicted: 0 for degree 0 and 1 -- Lemma moments -- and non-zero for degree 2);
#   - the relative moment residual after the minimal-change solve.
# ============================================================================
include(joinpath(@__DIR__, "gh_lab.jl"))

for (mname, M) in (("lattice N=6", latmesh(6, "lattice")), ("random N=6", latmesh(6, "random")),
                   ("pattern N=6", latmesh(6, "pattern")), ("SCVT mesh2", readmesh(joinpath(@__DIR__, "..", "..", "output", "gh_data", "mesh2.txt"))))
    geos = [egeo(M, e, M.X[M.E[e][1]]) for e in eachindex(M.E)]
    @printf("\n%s: %d triangles, %d edges, max|δ|/|ẽ| = %.3f\n", mname, length(M.Tr), length(M.E), maximum(abs(g.δ) / g.Ld for g in geos))
    for (cname, cells) in (("triangle, 1-ring", tri_cells(M)), ("diamond, 13", dia_cells(M)))
        for k in 0:2
            D = celldata(M, cells, k)
            Rs, info = constrained_R(M, cells, D, geos, k)
            ev = eigvals(Symmetric(Matrix(info.G)))
            nsmall = count(<(1e-9 * ev[end]), ev)
            ob = join([@sprintf("deg%d %.1e", d, info.obstruction[d]) for d in 0:k], ", ")
            @printf("  %-17s k=%d: dim ker G = %2d (predicted %2d), ‖Gλ‖ %.0e | obstruction: %s | residual %.1e -> %.1e, max dev %.2f\n",
                    cname, k, nsmall, 4length(mons(k)), info.kerG, ob, info.resid0, info.resid, info.dev)
            flush(stdout)
        end
    end
end
