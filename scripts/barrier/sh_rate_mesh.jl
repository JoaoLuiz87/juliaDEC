# sh_rate_mesh.jl -- periodic test meshes (lattice / mapped / random / pattern) and segment quadrature,
# shared by sh_rate_check.jl, window_skew_check.jl and route2_check.jl. Extracted from sh_rate_check.jl.
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

