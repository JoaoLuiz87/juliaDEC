# ============================================================================
#  story_videos.jl -- videos of the transport tests (MP4, CairoMakie + FFMPEG)
#
#      julia -t 4 scripts/transport/story_videos.jl
#        vid_uniform.mp4         uniform density in the deformational flow, mesh 6: ρ − 1 for H_DEC, H_P1, H_P2 on ONE colour scale
#        vid_deformational.mp4   two hills in the deformational flow, mesh 6: the three solutions, and H_DEC − H_P2, H_P1 − H_P2
#        vid_oneform.mp4         a 1-form in the deformational flow, mesh 5: vorticity (no star enters this operator)
#  Frames from ts_data/story_frames_mesh6.jls (story_transport.jl); the 1-form run is done here.
# ============================================================================
include(joinpath(@__DIR__, "extrusion_oneform.jl"))
include(joinpath(@__DIR__, "..", "shared", "thesis_plot_helpers.jl"))
using Serialization
const S3 = ("H_DEC", "H_P1", "H_P2")
FR = deserialize(joinpath(TSDIR, "story_frames_mesh6.jls"))
mesh6 = VoronoiMesh("meshes/mesh6.nc"); polys6, own6 = periodic_copies(cellpolys(mesh6))
function vax(pos, title)
    ax = Axis(pos; aspect = DataAspect(), title = title, titlealign = :left, titlesize = 16); hidedecorations!(ax); xlims!(ax, 0, 1); ylims!(ax, 0, 1); ax
end

# ---------------------------------------------------------------------------- uniform density
let F = Dict(s => [f .- 1 for f in FR[(:uniform, s)]] for s in S3), n = length(FR[(:uniform, "H_DEC")])
    m = maximum(maximum(abs, f) for f in F["H_DEC"])
    fig = Figure(size = (1500, 560), fontsize = 16); it = Observable(1)
    for (j, s) in enumerate(S3)
        ax = vax(fig[1, j], @lift(@sprintf("%s:  ρ − 1, max %.1e", s, maximum(abs, F[s][$it]))))
        poly!(ax, polys6; color = @lift(F[s][$it][own6]), colormap = DIVM, colorrange = (-m, m), strokewidth = 0)
    end
    Colorbar(fig[1, 4]; colormap = DIVM, limits = (-m, m), label = "ρ − 1")
    Label(fig[0, 1:3], @lift(@sprintf("uniform density in the deformational flow, 16384 cells, cubic extrusion, t = %.3f T (one colour scale)", ($it - 1) / (n - 1)));
          fontsize = 17, halign = :left)
    record(fig, thesis_out("vid_uniform.mp4"), 1:n; framerate = 12) do i; it[] = i; end
end

# ---------------------------------------------------------------------------- deformational: solutions and differences between the stars
let F = Dict(s => FR[(:deformational, s)] for s in S3), n = length(FR[(:deformational, "H_DEC")])
    r = extrema(F["H_P2"][1]); m = maximum(maximum(abs, F[s][k] .- F["H_P2"][k]) for s in ("H_DEC", "H_P1") for k in 1:n)
    fig = Figure(size = (1500, 1040), fontsize = 16); it = Observable(1)
    for (j, s) in enumerate(S3)
        ax = vax(fig[1, j], s)
        poly!(ax, polys6; color = @lift(F[s][$it][own6]), colormap = :viridis, colorrange = r, strokewidth = 0)
    end
    for (j, s) in enumerate(("H_DEC", "H_P1"))
        ax = vax(fig[2, j], @lift(@sprintf("%s − H_P2, max %.1e", s, maximum(abs, F[s][$it] .- F["H_P2"][$it]))))
        poly!(ax, polys6; color = @lift((F[s][$it] .- F["H_P2"][$it])[own6]), colormap = DIVM, colorrange = (-m, m), strokewidth = 0)
    end
    Colorbar(fig[2, 3]; colormap = DIVM, limits = (-m, m), label = "difference from H_P2", width = 20, tellwidth = false, halign = :left)
    Label(fig[0, 1:3], @lift(@sprintf("two hills in the deformational flow, 16384 cells, cubic extrusion, t = %.3f T", ($it - 1) / (n - 1)));
          fontsize = 17, halign = :left)
    record(fig, thesis_out("vid_deformational.mp4"), 1:n; framerate = 12) do i; it[] = i; end
end

# ---------------------------------------------------------------------------- a 1-form, mesh 5 (no star)
let mn = "mesh5"
    G = lie_setup(mn; cfv = (), upw = ()); extrusion_setup!(G); oneform_setup!(G)
    mesh = G["mesh"]; h = G["h"]; tris, _, _ = triangles_ccw(mesh); polys, own = periodic_copies(tripolys(tris))
    test = TESTS[:deformational]; β0 = x -> test.ρ0(x) .* [1.0, 0.5]; b = edge_integrals(mesh, β0); rhs = oneform_rhs(G, test)
    ζ(b) = G["σT"] .* (G["D1p"] * b) ./ G["AT"]
    nsteps = round(Int, test.T / (h / 8)); dt = test.T / nsteps; every = max(1, nsteps ÷ 80); frames = [ζ(b)]; t = 0.0
    for k in 1:nsteps
        k1 = rhs(b, t); k2 = rhs(b .+ (dt / 2) .* k1, t + dt / 2); k3 = rhs(b .+ (dt / 2) .* k2, t + dt / 2); k4 = rhs(b .+ dt .* k3, t + dt)
        b = b .+ (dt / 6) .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4); t += dt
        k % every == 0 && push!(frames, ζ(b))
    end
    zr = maximum(abs, frames[1]); n = length(frames)
    fig = Figure(size = (820, 800), fontsize = 16); it = Observable(1)
    ax = vax(fig[1, 1], @lift(@sprintf("1-form β = ρ₀(dx + ½dy), deformational flow, 4096 cells: vorticity, t = %.3f T", ($it - 1) / (n - 1))))
    poly!(ax, polys; color = @lift(frames[$it][own]), colormap = DIVM, colorrange = (-zr, zr), strokewidth = 0)
    Colorbar(fig[1, 2]; colormap = DIVM, limits = (-zr, zr), label = "D₁b / |T|")
    record(fig, thesis_out("vid_oneform.mp4"), 1:n; framerate = 12) do i; it[] = i; end
    println(@sprintf("1-form: vorticity at T vs t = 0, relative max difference %.2e", maximum(abs, frames[end] .- frames[1]) / zr))
end
println("videos written")
