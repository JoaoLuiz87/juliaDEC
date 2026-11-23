# ss_mesh.jl -- the perturbed lattice of the guide's Section 6.5 (porting check), shared by the ss_*.jl scripts
function porting_mesh(n = 20; Lx = 1.0, Ly = sqrt(3) / 2)
    N = n^2; h = sqrt(Lx * Ly / N); idx(i, j) = mod(j, n) * n + mod(i, n) + 1
    X = [[mod((i + 0.5 * mod(j, 2) + 0.25) * Lx / n + 0.1h * sin(2pi * (3i + 2j) / n), Lx),
          mod((j + 0.5) * Ly / n + 0.1h * cos(2pi * (i + 4j) / n + 0.5), Ly)] for j in 0:n-1 for i in 0:n-1]
    T = NTuple{3,Int}[]
    for j in 0:n-1, i in 0:n-1
        if iseven(j)
            push!(T, (idx(i, j), idx(i + 1, j), idx(i, j + 1))); push!(T, (idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)))
        else
            push!(T, (idx(i, j), idx(i + 1, j + 1), idx(i, j + 1))); push!(T, (idx(i, j), idx(i + 1, j), idx(i + 1, j + 1)))
        end
    end
    X, T, (Lx, Ly), h
end
"SCVT mesh exported by gh_export_scvt.jl: points, triangles, periods, h = sqrt(Lx Ly / N)"
function scvt_mesh(name)
    L = readlines(joinpath(@__DIR__, "..", "..", "output", "gh_data", "$name.txt")); Px, Py = parse.(Float64, split(L[1])); nX = parse(Int, L[2])
    X = [parse.(Float64, split(L[2+i])) for i in 1:nX]; nT = parse(Int, L[3+nX])
    T = [Tuple(parse.(Int, split(L[3+nX+i]))) for i in 1:nT]
    X, NTuple{3,Int}.(T), (Px, Py), sqrt(Px * Py / nX)
end
