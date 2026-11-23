# ============================================================================
#  sh_rate_check.jl -- at what rate does s_h go to zero? It depends on the mesh.
#
#      julia scripts/barrier/sh_rate_check.jl
#
#  Periodic triangulations of the torus [0,1) x [0, √3/2) built from the staggered
#  equilateral lattice with N x N vertices (h = 1/N), with four vertex perturbations:
#    lattice  : none                                         (δ_e = 0)
#    mapped   : a fixed smooth periodic map                  (δ_e = O(h²))
#    random   : independent random shifts of size 0.08 h    (δ_e = O(h), random signs)
#    pattern  : the same shift on every 2x2 block of vertices (δ_e = O(h), periodic, systematic)
#  s_h(u,v) = Σ_e α_e(v) β_e(u) - α_e(u) β_e(v) with 4-point Gauss on every segment, for two
#  pairs of smooth periodic fields:
#    (U, V)   : no common wavevector -> invisible to translation-invariant meshes
#    (U2, V2) : the same wavevectors, phase-shifted
#  Writes output/figures/sh_rate_data.txt.
# ============================================================================
using LinearAlgebra, Printf, Random

const J = [0.0 -1.0; 1.0 0.0]
cross2(a, b) = a[1] * b[2] - a[2] * b[1]
circumcentre(a, b, c) = (d = 2cross2(b - a, c - a); a + J * (dot(c - a, c - a) * (b - a) - dot(b - a, b - a) * (c - a)) / d)
const Ly = sqrt(3) / 2
U(x) = [sin(2pi * x[2] / Ly), cos(2pi * x[1])]
V(x) = [cos(2pi * (x[1] + x[2] / Ly)), sin(2pi * (x[1] - 2x[2] / Ly))]
U2(x) = [cos(2pi * x[1]) + sin(2pi * x[2] / Ly), cos(2pi * x[2] / Ly)]
V2(x) = [sin(2pi * x[1]) + cos(2pi * x[2] / Ly), sin(2pi * x[2] / Ly) + 0.5cos(2pi * x[1])]
const GL = [(-0.861136311594053, 0.347854845137454), (-0.339981043584856, 0.652145154862546),
            (0.339981043584856, 0.652145154862546), (0.861136311594053, 0.347854845137454)]
segint(f, a, b) = sum(w * f((a + b) / 2 + (t / 2) * (b - a)) for (t, w) in GL) * norm(b - a) / 2
"minimal image of y relative to x on the torus"
unwrap(y, x) = y - [round(y[1] - x[1]), Ly * round((y[2] - x[2]) / Ly)]

function mesh(N, kind)
    h = 1 / N; idx(i, j) = mod(j, N) * N + mod(i, N) + 1
    X = [[(i + 0.5 * mod(j, 2)) * h, j * h * Ly] for j in 0:N-1 for i in 0:N-1]
    rng = MersenneTwister(N)
    pat = Dict((0, 0) => [0.10, 0.03], (1, 0) => [-0.06, 0.08], (0, 1) => [0.04, -0.09], (1, 1) => [-0.08, -0.02])
    for j in 0:N-1, i in 0:N-1
        k = idx(i, j); x = X[k]
        if kind == "mapped"
            X[k] = x + 0.012 * [sin(2pi * x[1]) * cos(2pi * x[2] / Ly), sin(2pi * (x[1] + 0.3)) * sin(2pi * x[2] / Ly)]
        elseif kind == "random"
            X[k] = x + 0.08h * (2rand(rng, 2) .- 1)
        elseif kind == "pattern"
            X[k] = x + h * pat[(mod(i, 2), mod(j, 2))]
        end
    end
    T = NTuple{3,Int}[]
    for j in 0:N-1, i in 0:N-1
        if iseven(j)
            push!(T, (idx(i, j), idx(i + 1, j), idx(i, j + 1))); push!(T, (idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)))
        else
            push!(T, (idx(i, j), idx(i + 1, j + 1), idx(i, j + 1))); push!(T, (idx(i, j), idx(i + 1, j), idx(i + 1, j + 1)))
        end
    end
    X, T, h
end

function skews(X, T)
    edges = Dict{Tuple{Int,Int},Vector{Int}}()
    for (k, t) in enumerate(T), (a, b) in ((t[1], t[2]), (t[2], t[3]), (t[3], t[1]))
        push!(get!(edges, (min(a, b), max(a, b)), Int[]), k)
    end
    s1 = 0.0; s2 = 0.0; maxratio = 0.0; wc = true
    for ((a, b), ts) in edges
        @assert length(ts) == 2
        xa = X[a]; xb = unwrap(X[b], xa); m = (xa + xb) / 2; t = (xb - xa) / norm(xb - xa); n = J * t
        cs = map(ts) do k
            v = [unwrap(X[p], xa) for p in T[k]]; c = circumcentre(v...)
            apex = v[findfirst(p -> norm(p - xa) > 1e-12 && norm(p - xb) > 1e-12, v)]
            wc &= sign(dot(c - m, n)) == sign(dot(apex - m, n))
            c
        end
        c1, c2 = cs[1], cs[2]
        for (u, v, which) in ((U, V, 1), (U2, V2, 2))
            αu = segint(x -> dot(u(x), t), xa, xb); αv = segint(x -> dot(v(x), t), xa, xb)
            βu = segint(x -> dot(u(x), t), c1, c2); βv = segint(x -> dot(v(x), t), c1, c2)
            which == 1 ? (s1 += αv * βu - αu * βv) : (s2 += αv * βu - αu * βv)
        end
        sv = sort([dot(c1 - m, n), dot(c2 - m, n)]); maxratio = max(maxratio, abs(sum(sv) / 2) / (sv[2] - sv[1]))
    end
    s1, s2, maxratio, wc
end

open(joinpath(@__DIR__, "..", "..", "output", "figures", "sh_rate_data.txt"), "w") do io
    println(io, "# kind N h s_h(U,V) s_h(U2,V2) max(delta/|e~|)")
    @printf("%-8s %5s %9s %12s %6s %13s %6s %10s %s\n", "mesh", "N", "h", "|s_h(U,V)|", "rate", "|s_h(U2,V2)|", "rate", "max δ/|ẽ|", "w-c")
    for kind in ("lattice", "mapped", "random", "pattern")
        p1 = nothing; p2 = nothing
        for N in (8, 16, 32, 64, 128)
            X, T, h = mesh(N, kind); s1, s2, mr, wc = skews(X, T)
            r1 = p1 === nothing ? "" : @sprintf("%6.2f", log2(abs(p1) / abs(s1)))
            r2 = p2 === nothing ? "" : @sprintf("%6.2f", log2(abs(p2) / abs(s2)))
            @printf("%-8s %5d %9.5f %12.3e %6s %13.3e %6s %10.3f %s\n", kind, N, h, abs(s1), r1, abs(s2), r2, mr, wc)
            println(io, "$kind $N $h $(abs(s1)) $(abs(s2)) $mr"); p1 = s1; p2 = s2
        end
    end
end
