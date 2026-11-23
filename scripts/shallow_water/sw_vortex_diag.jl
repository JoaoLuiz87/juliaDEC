# sw_vortex_diag.jl -- why H_P2 degrades on the vortex on mesh 7: error against time, and where the largest error sits
#   (relative to the obtuse Delaunay triangles of the mesh).   julia -t 4 scripts/shallow_water/sw_vortex_diag.jl
push!(ARGS, "none")
include(joinpath(@__DIR__, "sw_peixoto_all.jl"))
mn = "mesh7"; mesh = VoronoiMesh("meshes/$mn.nc"); S = cached_stars(mn; mesh)
tris, _, _ = triangles_ccw(mesh)
function maxangle(P)
    a = norm(P[2] .- P[3]); b = norm(P[1] .- P[3]); c = norm(P[1] .- P[2])
    return maximum(acosd(clamp((y^2 + z^2 - x^2) / (2y * z), -1, 1)) for (x, y, z) in ((a, b, c), (b, c, a), (c, a, b)))
end
ang = [maxangle(T.pts) for T in tris]; obt = findall(>(90.0), ang)
cobt = [sum(tris[t].pts) ./ 3 .* LDOM for t in obt]
@printf("%s: %d obtuse triangles (max angle %.2f°)\n", mn, length(obt), maximum(ang))
for st in ("H_DEC", "H_P2")
    O = sw_ops(mesh, S[st]; fcor = FVX); c = sqrt(GJ * H0J); T = 10DAY; n = ceil(Int, T / (0.3 * minimum(O.le) / (c + 2U0J))); dt = T / n
    x0 = vortex_initial(O); x = copy(x0); buf = sw_buffers(O); k1, k2, k3, k4, xt = (similar(x) for _ in 1:5); ηr = x0[1:O.nc] .- H0J
    perday = n ÷ 10
    for k in 1:n
        rk4_step!(x, dt, O, buf, k1, k2, k3, k4, xt)
        if k % perday == 0
            eh = x[1:O.nc] .- x0[1:O.nc]; i = argmax(abs.(eh))
            dmin = isempty(cobt) ? NaN : minimum(norm(mindisp(O.xc[i], q, O.L)) for q in cobt) / (O.L / sqrt(O.nc))
            rc = norm(mindisp(O.xc[i], [O.L / 2, O.L / 2], O.L)) / O.L
            @printf("%-5s day %2d: depth rms %.3e max %.3e at r = %.3f L, %.1f h from the nearest obtuse triangle\n", st, k ÷ perday,
                    sqrt(sum(O.A .* eh .^ 2) / sum(O.A .* ηr .^ 2)), maximum(abs, eh) / maximum(abs, ηr), rc, dmin)
            flush(stdout)
        end
    end
end
