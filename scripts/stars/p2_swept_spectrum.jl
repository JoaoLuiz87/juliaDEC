# ============================================================================
#  p2_swept_spectrum.jl -- Laplacian spectrum of the swept exact stars of p2_min_asym_check.jl on the finer meshes
#
#      SW_MESH=mesh5 SW_K=2 SW_SWEEPS=0,1,4,16 julia -t 6 scripts/stars/p2_swept_spectrum.jl
#      -> appends to output/ts_data/p2_swept_spectrum.txt ; results in p2_swept_spectrum.jls
#
#  E_s = (P_A P_S)^s P_A 0: exact on [P_k]^2 row by row, depends on the mesh within ~ s+1 stencils.
# ============================================================================
ENV["LS_MAIN"] = "0"
include(joinpath(@__DIR__, "local_symmetric_check.jl"))
BLAS.set_num_threads(max(1, Threads.nthreads()))
const mn = get(ENV, "SW_MESH", "mesh5"); const kdeg = parse(Int, get(ENV, "SW_K", "2"))
const sweeps = parse.(Int, split(get(ENV, "SW_SWEEPS", "0,1,4,16"), ","))
const SWF = joinpath(@__DIR__, "..", "..", "output", "ts_data", "p2_swept_spectrum.jls")
RS = isfile(SWF) ? deserialize(SWF) : Dict{Any,Any}()
mesh = VoronoiMesh("meshes/$mn.nc"); Hd = hdiag(mesh); hd = diag(Hd); N = mesh.cells.n; A = mesh.cells.area
d0 = sparse(Float64.(d0_matrix_primal(mesh))); D = p1_rows(mesh; k = kdeg)
rows = project_rows(D, [zeros(length(D.nbv[e])) for e in 1:D.n])
open(joinpath(@__DIR__, "..", "..", "output", "ts_data", "p2_swept_spectrum.txt"), "a") do io
    say(s) = (println(s); flush(stdout); println(io, s); flush(io))
    say(@sprintf("\n%s (%d cells), stars exact on [P%d]^2", mn, N, kdeg))
    for it in 0:maximum(sweeps)
        if it in sweeps
            H = Hd + tosparse(D, rows); t = @elapsed (λ = eigvals(Matrix(Diagonal(1 ./ A) * (d0' * H * d0))))
            λmax = maximum(abs.(λ)); isc = abs.(imag.(λ)) .> 1e-9 * λmax; nc = count(isc)
            q = (im = maximum(abs.(imag.(λ))) / λmax, frac = nc / N, cmin = nc > 0 ? minimum(abs.(λ[isc])) / λmax : NaN,
                 gamma = maximum(abs.(imag.(sqrt.(complex.(λ))))) / sqrt(λmax), asymF = norm(H - H') / norm(H), g = gersh_sym(mesh, H),
                 Lam = bigLambda(mesh, H), minre = minimum(real.(λ)) / λmax, rinf = hodge_error(mesh, H, fields.d, supn.d).inf)
            RS[(mn, kdeg, it)] = q; serialize(SWF, RS)
            say(@sprintf("   sweeps %3d: max|Im λ|/λmax %.2e   non-real %5.2f%% (lowest at |λ|/λmax %.2f)   wave growth γ %.1e   ‖H−Hᵀ‖_F/‖H‖_F %.2e   g %+.3f  Λ %.2f  Hodge r∞ %.2e  min Re λ/λmax %+.0e  (%.0f s)",
                         it, q.im, 100q.frac, q.cmin, q.gamma, q.asymF, q.g, q.Lam, q.rinf, q.minre, t))
        end
        it < maximum(sweeps) && (global rows = project_rows(D, symmetrize(D, rows)))
    end
end
