# ============================================================================
#  p2_spectrum_check.jl -- do the complex eigenvalues of the H_P2 Laplacian go away under refinement?
#
#      julia -t 6 scripts/stars/p2_spectrum_check.jl        ->  output/ts_data/p2_spectrum.{txt,jls}
#      (ENV: SPEC_MESHES=mesh2,...,mesh6 ; SPEC_KEEP=1 keeps the results already saved)
#
#  L = A^-1 D0ᵀ H D0 (A = cell areas), full spectrum by dense eigvals.  Reported per mesh:
#    λmax, max|Im λ|/λmax, the Bendixson bound β/λmax, number and fraction of non-real eigenvalues,
#    where they sit (min and max of |λ|/λmax over the non-real ones), max|Im λ|/|λ| over the resolved part
#    (|λ| < λmax/16, i.e. frequency < ω_max/4) and over the lowest 200 eigenvalues,
#    antisymmetry of H: Frobenius ‖H_a‖/‖H‖ and operator norm ‖Hd^-1/2 H_a Hd^-1/2‖₂,
#    defects: number of cells with ≠ 6 edges.
#  Localisation (meshes ≤ 5, from the eigenvectors): share of the mass of the non-real modes carried by the defect
#  cells and their neighbours, against the share of the area those cells have; participation ratio.
# ============================================================================
isdefined(Main, :cached_stars) || include(joinpath(@__DIR__, "stars_cache.jl"))
using Printf, LinearAlgebra, SparseArrays, Serialization
BLAS.set_num_threads(max(1, Threads.nthreads()))
const OUT = String[]
logp(s) = (println(s); flush(stdout); push!(OUT, s))
const SPM = haskey(ENV, "SPEC_MESHES") ? Tuple(String.(split(ENV["SPEC_MESHES"], ","))) : ("mesh2", "mesh3", "mesh4", "mesh5", "mesh6")
const SPF = joinpath(@__DIR__, "..", "..", "output", "ts_data", "p2_spectrum.jls")
R = isfile(SPF) && get(ENV, "SPEC_KEEP", "0") == "1" ? deserialize(SPF) : Dict{Any,Any}()

"‖B‖₂ of a sparse matrix by power iteration on BᵀB."
function opnorm2(B; iters = 300)
    x = randn(size(B, 2)); x ./= norm(x); s = 0.0
    for _ in 1:iters
        y = B' * (B * x); s = norm(y); s == 0 && return 0.0; x = y ./ s
    end
    return sqrt(s)
end

for mn in SPM
    mesh = VoronoiMesh("meshes/$mn.nc"); S = cached_stars(mn; mesh); N = mesh.cells.n; A = mesh.cells.area
    d0 = sparse(Float64.(d0_matrix_primal(mesh))); Hd = hdiag(mesh); hd = diag(Hd)
    nedges = [length(mesh.cells.edges[c]) for c in 1:N]; defects = findall(!=(6), nedges)
    near = Set(defects); for c in defects, c2 in mesh.cells.cells[c]; push!(near, Int(c2)); end; near = sort(collect(near))
    logp(@sprintf("\n%s: %d cells, h = 1/%d;  defect cells (≠ 6 edges): %d (%.1f%%), with neighbours %.1f%% of the area", mn, N, round(Int, sqrt(N)),
                  length(defects), 100length(defects) / N, 100sum(A[near]) / sum(A)))
    for (name, H) in (("H_P2", S["H_P2"]), ("H_DEC", Hd))
        name == "H_DEC" && mn == "mesh6" && continue
        Ha = (H - H') / 2
        asymF = norm(H - H') / norm(H); asym2 = name == "H_DEC" ? 0.0 : opnorm2(Diagonal(1 ./ sqrt.(hd)) * Ha * Diagonal(1 ./ sqrt.(hd)))
        Lm = Matrix(Diagonal(1 ./ A) * (d0' * H * d0))
        t = @elapsed begin
            if N <= 4096 && name == "H_P2"
                F = eigen(Lm); λ = F.values; V = F.vectors
            else
                λ = eigvals(Lm); V = nothing
            end
        end
        λmax = maximum(abs.(λ)); isc = abs.(imag.(λ)) .> 1e-9 * λmax; nc = count(isc)
        β = name == "H_DEC" ? 0.0 : opnorm2(Diagonal(1 ./ sqrt.(A)) * (d0' * Ha * d0) * Diagonal(1 ./ sqrt.(A))) / λmax
        res = abs.(λ) .< λmax / 16; lowest = sortperm(abs.(λ))[1:min(200, N)]
        imres = maximum(abs.(imag.(λ[res])) ./ max.(abs.(λ[res]), 1e-300); init = 0.0)
        imlow = maximum(abs.(imag.(λ[lowest])) ./ max.(abs.(λ[lowest]), 1e-300); init = 0.0)
        first4 = sort(abs.(λ))[2:5]                                  # the cluster of 4π² = 39.478
        R[(mn, name)] = (lmax = λmax, im = maximum(abs.(imag.(λ))) / λmax, beta = β, nc = nc, frac = nc / N,
                         cmin = nc > 0 ? minimum(abs.(λ[isc])) / λmax : NaN, cmax = nc > 0 ? maximum(abs.(λ[isc])) / λmax : NaN,
                         imres = imres, imlow = imlow, asymF = asymF, asym2 = asym2, ndef = length(defects), minre = minimum(real.(λ)) / λmax,
                         first = sum(first4) / 4, time = t)
        r = R[(mn, name)]
        logp(@sprintf("   %-5s λmax %.4g  max|Im λ|/λmax %.2e (Bendixson β/λmax %.2e)  non-real: %d (%.1f%% of N, %.2f per defect cell), at |λ|/λmax in [%.2f, %.2f]",
                      name, r.lmax, r.im, r.beta, r.nc, 100r.frac, r.nc / max(length(defects), 1), r.cmin, r.cmax))
        logp(@sprintf("         resolved part (|λ| < λmax/16): max|Im λ|/|λ| %.1e;  lowest 200: %.1e;  min Re λ/λmax %+.1e;  first cluster %.4f (4π² = %.4f);  ‖H−Hᵀ‖_F/‖H‖_F %.2e, ‖Hd^-1/2 H_a Hd^-1/2‖₂ %.2e  (%.0f s)",
                      r.imres, r.imlow, r.minre, r.first, 4π^2, r.asymF, r.asym2, t))
        if V !== nothing && nc > 0
            idx = findall(isc); share = Float64[]; pr = Float64[]
            for i in idx
                x = abs2.(V[:, i]) .* A; x ./= sum(x)
                push!(share, sum(x[near])); push!(pr, 1 / (N * sum(x .^ 2)))
            end
            # the same two numbers for the real modes in the same band of |λ|, for comparison
            band = findall((.!isc) .& (abs.(λ) .>= minimum(abs.(λ[isc]))) .& (abs.(λ) .<= maximum(abs.(λ[isc]))))
            shareR = [(x = abs2.(V[:, i]) .* A; x ./= sum(x); sum(x[near])) for i in band]
            prR = [(x = abs2.(V[:, i]) .* A; x ./= sum(x); 1 / (N * sum(x .^ 2))) for i in band]
            R[(mn, :loc)] = (share = sum(share) / length(share), sharemin = minimum(share), pr = sum(pr) / length(pr),
                             shareR = isempty(shareR) ? NaN : sum(shareR) / length(shareR), prR = isempty(prR) ? NaN : sum(prR) / length(prR), area = sum(A[near]) / sum(A))
            l = R[(mn, :loc)]
            logp(@sprintf("         non-real modes: %.0f%% of their mass on defect cells + neighbours (min %.0f%%; these cells are %.1f%% of the area); participation ratio %.3f.  Real modes of the same band: %.0f%%, participation %.3f",
                          100l.share, 100l.sharemin, 100l.area, l.pr, 100l.shareR, l.prR))
        end
        serialize(SPF, R)
    end
end
write(joinpath(@__DIR__, "..", "..", "output", "ts_data", "p2_spectrum.txt"), join(OUT, "\n") * "\n")
