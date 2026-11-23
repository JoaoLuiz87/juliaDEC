# ============================================================================
#  k_norm_check.jl -- does the skew part K of the lift decay under refinement?
#
#      julia scripts/barrier/k_norm_check.jl
#
#  For the [P1]² and [P2]² lifts on the regular lattice and on SCVT meshes 2-5:
#    max|K_em| / max|H_em|  (entries),  ‖K‖₂/‖H‖₂  (spectral norms, power iteration),
#    the five largest distinct |K_em| (to see whether the entries themselves repeat),
#    and, for comparison, the weak quantity 2α(V)ᵀKα(U) on two fixed smooth fields.
#  Writes output/figures/k_norm_data.txt.
# ============================================================================
using VoronoiMeshes, TensorsLite, TensorsLiteGeometry, SmallCollections, LinearAlgebra, NCDatasets
using StaticArrays, SparseArrays, Printf
Base.similar(bc::Base.Broadcast.Broadcasted, ::Type{Union{}}) = Array{Union{}}(undef, length.(axes(bc)))
const ROOT = joinpath(@__DIR__, "..", "..")
cd(ROOT)
include(joinpath(ROOT, "src", "juliaDEC.jl")); include(joinpath(ROOT, "src", "hodge_stars.jl"))

α(m, w) = reduce_primal_edges(m, w).values
meanh(m) = sum(edge_lengths(m, true)) / m.edges.n
U(x) = [sin(2pi * x[2]), cos(2pi * x[1])]
V(x) = [cos(2pi * (x[1] + x[2])), sin(2pi * (x[1] - 2x[2]))]

"largest singular value by power iteration on AᵀA"
function opnorm2(A; it = 400)
    x = normalize(randn(size(A, 2))); σ = 0.0
    for _ in 1:it
        y = A' * (A * x); σ = sqrt(norm(y)); x = y / norm(y)
    end
    σ
end

const FAMS = ["regular" => ["mesh2_regular", "mesh3_regular", "mesh4_regular", "mesh5_regular"],
              "SCVT" => ["mesh2", "mesh3", "mesh4", "mesh5"]]
open(joinpath(ROOT, "report", "k_norm_data.txt"), "w") do io
    println(io, "# family mesh star h maxK/maxH opK/opH weak_rel top5|K|")
    @printf("%-8s %-14s %-8s %8s %12s %12s %16s   %s\n", "family", "mesh", "star", "h", "max|K|/max|H|", "‖K‖₂/‖H‖₂", "αᵀKα/(|α||α|)", "five largest distinct |K_em|")
    for (fam, meshes) in FAMS, mn in meshes
        m = VoronoiMesh("meshes/$mn.nc"); h = meanh(m)
        aU = α(m, U); aV = α(m, V)
        for (lbl, ol) in (("lift P1", true), ("lift P2", false))
            H = hodge_star_lift_matrix(m; only_linear = ol); K = (H - H') / 2
            rK = maximum(abs, nonzeros(K)) / maximum(abs, nonzeros(H))
            oK = opnorm2(K) / opnorm2(H)
            weak = 2dot(aV, K * aU) / (norm(aU) * norm(aV))
            vals = first(sort(unique(round.(abs.(nonzeros(K)), digits = 6)), rev = true), 5)
            @printf("%-8s %-14s %-8s %8.4f %12.4f %12.4f %16.2e   %s\n", fam, mn, lbl, h, rK, oK, weak, join(string.(vals), " "))
            println(io, "$fam $mn $lbl $h $rK $oK $weak $(join(string.(vals), ","))")
            flush(stdout)
        end
    end
end
