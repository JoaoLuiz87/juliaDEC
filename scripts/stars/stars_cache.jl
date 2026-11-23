# ============================================================================
#  stars_cache.jl -- build the three stars of three_stars.jl once per mesh and keep them on disk
#
#      julia -t 4 scripts/stars/stars_cache.jl mesh2 mesh3 ... mesh7     ->  output/ts_data/stars_cache/<mesh>.jls
#
#  cached_stars(mn) loads the file if it exists, otherwise builds and saves it.
# ============================================================================
isdefined(Main, :three_stars) || include(joinpath(@__DIR__, "three_stars.jl"))
const STARDIR = joinpath(@__DIR__, "..", "..", "output", "ts_data", "stars_cache"); mkpath(STARDIR)

function cached_stars(mn; mesh = nothing)
    f = joinpath(STARDIR, "$mn.jls")
    isfile(f) && return deserialize(f)
    mesh === nothing && (mesh = VoronoiMesh("meshes/$mn.nc"))
    t = @elapsed ((S, info) = three_stars(mesh))
    S = Dict{String,Any}(k => sparse(v) for (k, v) in S); S["build_time"] = t; S["info"] = info
    serialize(f, S)
    return S
end

if abspath(PROGRAM_FILE) == @__FILE__
    for mn in ARGS
        S = cached_stars(mn)
        println("$mn: stars built in $(round(S["build_time"], digits = 1)) s; nnz H_P1 $(nnz(S["H_P1"])), H_P2 $(nnz(S["H_P2"]))"); flush(stdout)
    end
end
