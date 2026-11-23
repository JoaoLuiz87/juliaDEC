# ============================================================================
#  offset_scaling_check.jl -- does δ_e/|ẽ| go to zero under refinement?
#
#      julia scripts/barrier/offset_scaling_check.jl
#
#  Three families of triangulations of (a part of) the unit square, all built
#  from the equilateral lattice with spacing h = 1/N:
#    lattice   : the lattice itself                       (δ_e = 0)
#    mapped    : x ↦ Φ(x), Φ a fixed smooth map             (expected δ_e = O(h²), δ/|ẽ| = O(h))
#    perturbed : x ↦ x + 0.10 h ξ, ξ random in [-1,1]²      (expected δ/|ẽ| = O(1))
#  For every interior edge: s_i = (c_i - m_e)·n_e, |ẽ| = s₂ - s₁, δ_e = (s₁ + s₂)/2.
#  Writes output/figures/offset_scaling.txt.
# ============================================================================
using LinearAlgebra, Printf, Random

const J = [0.0 -1.0; 1.0 0.0]
cross2(a, b) = a[1] * b[2] - a[2] * b[1]
circumcentre(a, b, c) = (d = 2cross2(b - a, c - a); a + J * (dot(c - a, c - a) * (b - a) - dot(b - a, b - a) * (c - a)) / d)

function lattice(N)
    h = 1 / N; M = round(Int, 2N / sqrt(3))
    idx(i, j) = j * (N + 1) + i + 1
    X = [[(i + 0.5 * mod(j, 2)) * h, j * h * sqrt(3) / 2] for j in 0:M for i in 0:N]
    T = NTuple{3,Int}[]
    for j in 0:M-1, i in 0:N-1
        if iseven(j)
            push!(T, (idx(i, j), idx(i + 1, j), idx(i, j + 1)))
            i + 1 <= N - 0 && i + 1 <= N && push!(T, (idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)))
        else
            push!(T, (idx(i, j), idx(i + 1, j + 1), idx(i, j + 1)))
            push!(T, (idx(i, j), idx(i + 1, j), idx(i + 1, j + 1)))
        end
    end
    X, T, h
end
Φ(x) = x + 0.012 * [sin(2pi * x[1]) * sin(2pi * x[2]), sin(2pi * (x[1] + 0.3)) * cos(2pi * x[2])]

function offsets(X, T)
    edges = Dict{Tuple{Int,Int},Vector{Int}}()
    for (k, t) in enumerate(T), (a, b) in ((t[1], t[2]), (t[2], t[3]), (t[3], t[1]))
        push!(get!(edges, (min(a, b), max(a, b)), Int[]), k)
    end
    C = [circumcentre(X[t[1]], X[t[2]], X[t[3]]) for t in T]
    r = Float64[]; wc = true
    for ((a, b), ts) in edges
        length(ts) == 2 || continue
        m = (X[a] + X[b]) / 2; tt = (X[b] - X[a]) / norm(X[b] - X[a]); n = J * tt
        s = [dot(C[k] - m, n) for k in ts]; s1, s2 = minimum(s), maximum(s)
        # well-centred check: each circumcentre on the side of its own triangle's apex
        for k in ts
            apex = setdiff(collect(T[k]), [a, b])[1]
            wc &= sign(dot(C[k] - m, n)) == sign(dot(X[apex] - m, n))
        end
        push!(r, abs((s1 + s2) / 2) / (s2 - s1))
    end
    maximum(r), sqrt(sum(r .^ 2) / length(r)), wc
end

open(joinpath(@__DIR__, "..", "..", "output", "figures", "offset_scaling.txt"), "w") do io
    println(io, "# family N h max(delta/|e~|) rms(delta/|e~|)")
    @printf("%-10s %5s %9s %14s %14s %s\n", "family", "N", "h", "max δ/|ẽ|", "rms δ/|ẽ|", "well-centred")
    for fam in ("lattice", "mapped", "perturbed")
        prev = nothing
        for N in (16, 32, 64, 128)
            X, T, h = lattice(N)
            if fam == "mapped"
                X = [Φ(x) for x in X]
            elseif fam == "perturbed"
                rng = MersenneTwister(N); X = [x + 0.10h * (2rand(rng, 2) .- 1) for x in X]
            end
            mx, rms, wc = offsets(X, T)
            rate = prev === nothing ? "" : @sprintf("  (rate %.2f)", log2(prev / mx))
            @printf("%-10s %5d %9.5f %14.3e %14.3e %s%s\n", fam, N, h, mx, rms, wc, rate)
            println(io, "$fam $N $h $mx $rms")
            prev = mx
        end
    end
end
