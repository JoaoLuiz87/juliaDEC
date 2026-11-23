# ============================================================================
#  gs_square_grid.jl -- on the square grid (no offsets) a symmetric star that is NOT a Galerkin form is exact on
#                       cubic fields:  H = I + (D_nn - D_tt)/24,  D = second difference over the parallel edges
#                       across (n) and along (t) the edge.  Checks symmetry, the eigenvalue range and the orders.
#
#      julia scripts/archive/gs_square_grid.jl          (writes output/gs_data/square_grid.txt)
# ============================================================================
using LinearAlgebra, SparseArrays, Printf
const GL = [(-0.861136311594053, 0.347854845137454), (-0.339981043584856, 0.652145154862546),
            (0.339981043584856, 0.652145154862546), (0.861136311594053, 0.347854845137454)]
seg(f, a, b) = sum(w * f(a + ((ξ + 1) / 2) * (b - a)) for (ξ, w) in GL) / 2       # mean of f over [a, b]
u(x) = [sin(2pi * x[1]) * cos(2pi * x[2]) + cos(4pi * x[2]), cos(2pi * x[1]) * sin(2pi * x[2]) + sin(2pi * (x[1] + x[2]))]
lines = String[]
logp(s) = (println(s); push!(lines, s))

# edges: horizontal (i,j) from (i,j)h to (i+1,j)h, index i + N j + 1; vertical (i,j) from (i,j)h to (i,j+1)h, index N² + ...
function star_and_cochains(N)
    h = 1 / N; id(i, j) = mod(i, N) + N * mod(j, N) + 1
    α = zeros(2N^2); β = zeros(2N^2)
    for j in 0:N-1, i in 0:N-1
        a = [i * h, j * h]
        α[id(i, j)] = h * seg(x -> u(x)[1], a, a + [h, 0])                                            # ∫_e u·t, t = (1,0)
        β[id(i, j)] = h * seg(x -> u(x)[1], a + [h / 2, -h / 2], a + [h / 2, h / 2])                 # ∫_ẽ u·t
        α[N^2 + id(i, j)] = h * seg(x -> u(x)[2], a, a + [0, h])                                      # t = (0,1)
        β[N^2 + id(i, j)] = h * seg(x -> u(x)[2], a + [-h / 2, h / 2], a + [h / 2, h / 2])
    end
    I_, J_, V_ = Int[], Int[], Float64[]
    add(r, c, v) = (push!(I_, r); push!(J_, c); push!(V_, v))
    for j in 0:N-1, i in 0:N-1
        e = id(i, j)                                       # horizontal: t-neighbours (i±1, j), n-neighbours (i, j±1)
        add(e, e, 1.0); add(e, id(i, j + 1), 1 / 24); add(e, id(i, j - 1), 1 / 24); add(e, id(i + 1, j), -1 / 24); add(e, id(i - 1, j), -1 / 24)
        f = N^2 + id(i, j)                                 # vertical: t-neighbours (i, j±1), n-neighbours (i±1, j)
        add(f, f, 1.0); add(f, N^2 + id(i + 1, j), 1 / 24); add(f, N^2 + id(i - 1, j), 1 / 24)
        add(f, N^2 + id(i, j + 1), -1 / 24); add(f, N^2 + id(i, j - 1), -1 / 24)
    end
    sparse(I_, J_, V_, 2N^2, 2N^2), α, β, h
end

logp(@sprintf("%5s %12s %12s %6s %6s   %s", "N", "DEC max", "corrected", "ord", "ord", "symmetry, eigenvalue range"))
prev = nothing
for N in (8, 16, 32, 64, 128)
    H, α, β, h = star_and_cochains(N)
    eD = maximum(abs.(α - β)) / h; eH = maximum(abs.(H * α - β)) / h                 # DEC = identity here (|ẽ| = |e|)
    ev = N <= 32 ? extrema(eigvals(Symmetric(Matrix(H)))) : (NaN, NaN)
    o1, o2 = prev === nothing ? (NaN, NaN) : (log2(prev[1] / eD), log2(prev[2] / eH))
    logp(@sprintf("%5d %12.3e %12.3e %6.2f %6.2f   ‖H-Hᵀ‖ = %.0e, [%.4f, %.4f]", N, eD, eH, o1, o2, norm(H - H', 1), ev...))
    global prev = (eD, eH)
end
write(joinpath(@__DIR__, "..", "..", "output", "gs_data", "square_grid.txt"), join(lines, "\n") * "\n")
