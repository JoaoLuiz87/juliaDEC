# ============================================================================
#  story_eigen.jl -- eigenvalues of the Laplacian  D0ᵀ H D0 p = λ M0 p  on the unit torus, for H_DEC, H_P1, H_P2
#
#      julia -t 4 scripts/elliptic/story_eigen.jl      ->  output/ts_data/story_eigen.{txt,jls}
#
#  Exact eigenvalues 4π²(m² + n²), (m,n) ∈ Z², with multiplicities. Discrete eigenvalues sorted by real part (zero mode dropped)
#  and compared index by index with the sorted exact list. M0 = diag(A) (exact on linear functions only, caps the order at 2)
#  and the corrected 0-form star (exact on cubics), so that the star is the only limit.
# ============================================================================
include(joinpath(@__DIR__, "..", "stars", "stars_cache.jl"))
ENV["SW_MAIN"] = "0"; include(joinpath(@__DIR__, "..", "shallow_water", "sw_linear_check.jl"))
using LinearAlgebra, SparseArrays, Printf, Serialization
OUTE = String[]; se(s) = (println(s); flush(stdout); push!(OUTE, s))
exact_eigs(n) = sort([4π^2 * (m^2 + k^2) for m in -40:40 for k in -40:40 if (m, k) != (0, 0)])[1:n]
NE = 100
λex = exact_eigs(NE)
RE = Dict{Any,Any}()
for mn in ("mesh2", "mesh3", "mesh4", "mesh5")
    mesh = VoronoiMesh("meshes/$mn.nc"); S = cached_stars(mn; mesh); A = mesh.cells.area; h = sqrt(1 / mesh.cells.n)
    D0 = sparse(Float64.(d0_matrix_primal(mesh))); M0c = Matrix(mass0_corrected(mesh))
    n = min(NE, mesh.cells.n - 1)
    for st in ("H_DEC", "H_P1", "H_P2"), (mk, M) in ((:A, Matrix(Diagonal(A))), (:C, M0c))
        L = Matrix(D0' * S[st] * D0)
        t = @elapsed (λ = eigvals(M \ L))
        λ = sort(λ; by = real)[2:end]                                   # drop the zero mode
        re = real.(λ[1:n]); im_ = abs.(imag.(λ[1:n]))
        err = abs.(re .- λex[1:n]) ./ λex[1:n]
        RE[(mn, st, mk)] = (h = h, re = re, im = im_, err = err, imax = maximum(abs.(imag.(λ))) / maximum(real.(λ)))
        se(@sprintf("%s %-5s M0=%s: rel. error of λ_1..λ_20 max %.2e, λ_1..λ_%d max %.2e; max|Im λ|/λ_max (all) %.1e; low modes max|Im|/Re %.1e (%.0f s)",
                    mn, st, mk == :A ? "A " : "Ac", maximum(err[1:min(20, n)]), n, maximum(err), RE[(mn, st, mk)].imax, maximum(im_ ./ re), t))
        serialize(joinpath(TSDIR, "story_eigen.jls"), RE)
    end
end
write(joinpath(TSDIR, "story_eigen.txt"), join(OUTE, "\n") * "\n")
println("done")
