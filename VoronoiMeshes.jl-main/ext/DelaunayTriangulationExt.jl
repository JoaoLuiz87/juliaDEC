module DelaunayTriangulationExt

using Zeros, VoronoiMeshes, DelaunayTriangulation, TensorsLite, SmallCollections, TensorsLiteGeometry
using Polyester
using PrecompileTools

"""
    probable_max_dc(N::Integer, lx::Number, ly::Number, [density_extrema=(1,1)]) -> dc::Real

Given a mesh with `N` cells, (`lx`, `ly`) periods, and density extremas `density_extrema`, returns an educated guess at what the maximum cell distance `dc` will be.
"""
function probable_max_dc(N::Integer, lx::Real, ly::Real, (min_den, max_den) = (1, 1))
    0.95231281 * sqrt(4 * (lx * ly / N) / pi) * ((max_den / min_den)^(1/3))
end

"""
    probable_min_dc(N::Integer, lx::Number, ly::Number, [density_extrema=(1,1)]) -> dc::Real

Given a mesh with `N` cells, (`lx`, `ly`) periods, and density extremas `density_extrema`, returns an educated guess at what the minimum cell distance `dc` will be.
"""
function probable_min_dc(N::Integer, lx::Real, ly::Real, (min_den, max_den) = (1, 1))
    0.95231281 * sqrt(4 * (lx * ly / N) / pi) / ((max_den / min_den)^(1/3))
end


function expand_periodic_points(points::AbstractVector{<:Vec2Dxy}, xp::Number, yp::Number, density_extrema = (1, 1))
    points_indices = eachindex(points)

    N = length(points)
    #N_div = 2
    N_div = max(1, Int(floor(xp / (5 * probable_max_dc(N, xp, yp, density_extrema)))))

    pxi = filter(i -> (points[i].x < xp / N_div), eachindex(points))
    px = @view(points[pxi]) .+ (xp * 𝐢)

    pxpyi = filter(i -> ((points[i].x < xp / N_div) && (points[i].y < yp / N_div)), eachindex(points))
    pxpy = @view(points[pxpyi]) .+ (xp * 𝐢 + yp * 𝐣)

    pyi = filter(i -> (points[i].y < yp / N_div), eachindex(points))
    py = @view(points[pyi]) .+ (yp * 𝐣)

    mxpyi = filter(i -> ((points[i].x > xp * (1 - 1 / N_div)) && (points[i].y < yp / N_div)), eachindex(points))
    mxpy = @view(points[mxpyi]) .+ (-xp * 𝐢 + yp * 𝐣)

    mxi = filter(i -> (points[i].x > xp * (1 - 1 / N_div)), eachindex(points))
    mx = @view(points[mxi]) .+ (-xp * 𝐢)

    mxmyi = filter(i -> ((points[i].x > xp * (1 - 1 / N_div)) && (points[i].y > yp * (1 - 1 / N_div))), eachindex(points))
    mxmy = @view(points[mxmyi]) .+ (-xp * 𝐢 - yp * 𝐣)

    myi = filter(i -> (points[i].y > yp * (1 - 1 / N_div)), eachindex(points))
    my = @view(points[myi]) .+ (-yp * 𝐣)

    pxmyi = filter(i -> ((points[i].x < xp / N_div) && (points[i].y > yp * (1 - 1 / N_div))), eachindex(points))
    pxmy = @view(points[pxmyi]) .+ (xp * 𝐢 - yp * 𝐣)

    new_points = vcat(points, px, pxpy, py, mxpy, mx, mxmy, my, pxmy)

    inds = vcat(points_indices, pxi, pxpyi, pyi, mxpyi, mxi, mxmyi, myi, pxmyi)

    return new_points, inds
end

function generate_inital_points(N::Integer, lx::Real, ly::Real)
    Nquadrant, rest = divrem(N, 9)
    if rest == 0
        Nq1 = Nq2 = Nq3 = Nq4 = Nq5 = Nq6 = Nq7 = Nq8 = Nq9 = Nquadrant
    elseif rest == 1
        Nq2 = Nq3 = Nq4 = Nq5 = Nq6 = Nq7 = Nq8 = Nq9 = Nquadrant
        Nq1 = Nq2 + 1
    elseif rest == 2
        Nq3 = Nq4 = Nq5 = Nq6 = Nq7 = Nq8 = Nq9 = Nquadrant
        Nq1 = Nq2 = Nq3 + 1
    elseif rest == 3
        Nq4 = Nq5 = Nq6 = Nq7 = Nq8 = Nq9 = Nquadrant
        Nq1 = Nq2 = Nq3 = Nq4 + 1
    elseif rest == 4
        Nq5 = Nq6 = Nq7 = Nq8 = Nq9 = Nquadrant
        Nq1 = Nq2 = Nq3 = Nq4 = Nq5 + 1
    elseif rest == 5
        Nq6 = Nq7 = Nq8 = Nq9 = Nquadrant
        Nq1 = Nq2 = Nq3 = Nq4 = Nq5 = Nq6 + 1
    elseif rest == 6
        Nq7 = Nq8 = Nq9 = Nquadrant
        Nq1 = Nq2 = Nq3 = Nq4 = Nq5 = Nq6 = Nq7 + 1
    elseif rest == 7
        Nq8 = Nq9 = Nquadrant
        Nq1 = Nq2 = Nq3 = Nq4 = Nq5 = Nq6 = Nq7 = Nq8 + 1
    elseif rest == 8
        Nq9 = Nquadrant
        Nq1 = Nq2 = Nq3 = Nq4 = Nq5 = Nq6 = Nq7 = Nq8 = Nq9 + 1
    end
    Vq1 = VecArray(x = (rand(Nq1) .* lx / 3), y = rand(Nq1) .* ly / 3)
    Vq2 = VecArray(x = (rand(Nq2) .* lx / 3 .+ lx / 3), y = rand(Nq2) .* ly / 3)
    Vq3 = VecArray(x = (rand(Nq3) .* lx / 3 .+ 2 * lx / 3), y = rand(Nq3) .* ly / 3)
    Vq4 = VecArray(x = (rand(Nq4) .* lx / 3), y = rand(Nq4) .* ly / 3 .+ ly / 3)
    Vq5 = VecArray(x = (rand(Nq5) .* lx / 3 .+ lx / 3), y = rand(Nq5) .* ly / 3 .+ ly / 3)
    Vq6 = VecArray(x = (rand(Nq6) .* lx / 3 .+ 2 * lx / 3), y = rand(Nq6) .* ly / 3 .+ ly / 3)
    Vq7 = VecArray(x = (rand(Nq7) .* lx / 3), y = rand(Nq7) .* ly / 3 .+ 2 * ly / 3)
    Vq8 = VecArray(x = (rand(Nq8) .* lx / 3 .+ lx / 3), y = rand(Nq8) .* ly / 3 .+ 2 * ly / 3)
    Vq9 = VecArray(x = (rand(Nq9) .* lx / 3 .+ 2 * lx / 3), y = rand(Nq9) .* ly / 3 .+ 2 * ly / 3)
    return vcat(Vq1, Vq2, Vq3, Vq4, Vq5, Vq6, Vq7, Vq8, Vq9)
end

@inline const_density(𝐱) = 1

@inline TensorsLiteGeometry.mass_centroid(::typeof(const_density), points::AbstractVector{T}) where {T <: Vec} = TensorsLiteGeometry.centroid(points)

function fill_with_polygon_mass_centroids!(new_points, pol_length, N::Integer, lx::Number, ly::Number, voro, ρ::F = const_density) where {F <: Function}
    polygons = voro.polygons
    polygon_points = reinterpret(Vec2Dxy{Float64}, voro.polygon_points)

    #Compute mass centroid for central polygons
    @batch for i in Base.OneTo(N)
        @inbounds begin
            v = polygons[i]
            vertices_position = @view(polygon_points[@view(v[1:(end - 1)])])
            new_points[i] = periodic_to_base_point(mass_centroid(ρ, vertices_position), lx, ly)
            pol_length[i] = 0.95231281 * sqrt(4 * area(vertices_position) / pi)
        end
    end

    return new_points
end

function centroidal_voronoi_loyd(::TT, vor::TV, initial_generator_points, N::Integer, lx::Real, ly::Real; density::F = const_density, max_iter::Integer = typemax(Int), rtol::Real = 1.0e-8 / 100, max_time = 4.0) where {TT <: Triangulation, TV <: VoronoiTessellation, F <: Function}

    initial_new = similar(initial_generator_points)
    polygon_length = Vector{Float64}(undef, N)

    fill_with_polygon_mass_centroids!(initial_new, polygon_length, N, lx, ly, vor, density)

    den_extrema = extrema(density, initial_new)
    p_new, inds_new = expand_periodic_points(initial_new, lx, ly, den_extrema)

    new_tri::TT = triangulate(reinterpret(NTuple{2, Float64}, p_new))
    new_vor::TV = voronoi(new_tri, clip = true)

    #expected_dc = probable_min_dc(N, lx, ly, den_extrema)

    t1 = time() / 60
    println("Performing Lloyd's iteration with relative tolerance = $rtol, maximum number of iterations: $max_iter, and max time of $max_time minutes")
    iter = 0
    while !(
            mapreduce(
                (x, y, z) -> isapprox_periodic(
                    x, y, lx, ly;
                    atol = rtol * z
                ),
                &, initial_generator_points, initial_new, polygon_length
            )
        ) &&
            iter < max_iter &&
            (time() / 60) - t1 < max_time
        iter += 1
        mod(iter, 10) == 0 && print("Elapsed time: $(round((time() / 60) - t1, digits=1)) minutes. Lloyd iteration number $iter \u001b[1000D")

        initial_generator_points .= initial_new
        vor::TV = new_vor

        fill_with_polygon_mass_centroids!(initial_new, polygon_length, N, lx, ly, vor, density)

        p_new, inds_new = expand_periodic_points(initial_new, lx, ly, den_extrema)

        new_tri = triangulate(reinterpret(NTuple{2, Float64}, p_new))
        new_vor = voronoi(new_tri, clip = true)
    end

    println("Stopped Lloyd's algorithm at iteration $iter, after $(round((time() / 60) - t1, digits=1)) minutes")

    return initial_new, p_new, inds_new, new_vor, iter
end

is_inside_box(p::Vec2Dxy, lx::Number, ly::Number) = (0 <= p.x < lx) && (0 <= p.y < ly)

function check_and_fix_periodicity_points(points, lx::Number, ly::Number)
    x_max, _ = findmax(v -> v.x, points)
    x_min, _ = findmin(v -> v.x, points)
    y_max, _ = findmax(v -> v.y, points)
    y_min, _ = findmin(v -> v.y, points)

    if ((x_max - x_min) >= lx) || ((y_max - y_min) >= ly)
        throw(DomainError((lx, ly), "Points cannot be cast to periodic domain [0, $lx) × [0, $ly)"))
    end

    if (0 <= x_min < x_max < lx) && (0 <= y_min < y_max < ly)
        return points
    end

    displacement = (((x_max + x_min) / 2) * 𝐢 + ((y_max + y_min) / 2) * 𝐣) - ((lx / 2) * 𝐢 + (ly / 2) * 𝐣)

    @warn "Points given will be shifted by ($(displacement.x), $(displacement.y)) in order to be centered in the [0, $lx) × [0, $ly) domain"

    points_periodic = similar(points)
    points_periodic .= points .- displacement
    return points_periodic
end

function generate_periodic_centroidal_voronoi(points::AbstractVector, lx::Real, ly::Real; density::F = const_density, max_iter::Integer = typemax(Int), rtol::Real = 1.0e-8 / 100, max_time = 4.0) where {F <: Function}

    initial_generator_points = check_and_fix_periodicity_points(points, lx, ly)

    N = length(initial_generator_points)
    p, inds = expand_periodic_points(initial_generator_points, lx, ly, extrema(density, initial_generator_points))
    p_tuple = reinterpret(NTuple{2, Float64}, p)
    tri = triangulate(p_tuple)
    vor = voronoi(tri, clip = true)

    initial_new, p_new, inds_new, new_vor, n_iter = centroidal_voronoi_loyd(tri, vor, initial_generator_points, N, lx, ly, density = density, max_iter = max_iter, rtol = rtol, max_time = max_time)
    return new_vor
end

function find_base_point_index(point::Vec2Dxy, interior_points, x_period::Number, y_period::Number)
    findfirst(j -> isapprox_periodic(point, @inbounds(interior_points[j]), x_period, y_period), eachindex(interior_points))
end

function extract_periodic_vertices_and_cells(N::Integer, lx::Number, ly::Number, vor::VoronoiTessellation)
    #vertices position as tuples
    vertices_pos_tuple = vor.polygon_points

    #function that filters (choses) points that are inside the [0, lx) × [0, ly) domain
    fil_func = let vertices_pos_tuple = vertices_pos_tuple
        function (i)
            vp = vertices_pos_tuple[i]
            return (0 <= vp[1] < lx) && (0 <= vp[2] < ly)
        end
    end

    #indices belonging to vertices that are in the base domain [0, lx) × [0, ly)
    interior_vertex_i = filter(fil_func, eachindex(vertices_pos_tuple))

    nVertices = length(interior_vertex_i)
    interior_vertex_pos = VecArray(
        x = map(a -> a[1], @view(vertices_pos_tuple[interior_vertex_i])),
        y = map(a -> a[2], @view(vertices_pos_tuple[interior_vertex_i]))
    )

    T = SmallVector{16, Int32}
    verticesOnCell_global = vor.polygons
    verticesOnCell = Vector{T}(undef, N)
    let vertices_pos_vec = reinterpret(Vec2Dxy{Float64}, vertices_pos_tuple), 
        interior_vertex_pos = interior_vertex_pos, lx = lx, ly = ly
        func1 = @inline i -> @inbounds(getindex(vertices_pos_vec, i))
        func2 = @inline v -> find_base_point_index(v, interior_vertex_pos, lx, ly)
        @batch for c in eachindex(verticesOnCell)
            vocg = T(@view(verticesOnCell_global[c][1:(end - 1)]))
            vps = map(func1, vocg)
            verticesOnCell[c] = map(func2, vps)
        end
    end

    cell_pos = parent(vor.triangulation.points)
    interior_cell_pos = cell_pos[Base.OneTo(N)]
    cellsOnVertex_global = vor.circumcenter_to_triangle
    cellsOnVertex = Vector{FixedVector{3, Int32}}(undef, nVertices)
    let cell_pos = cell_pos, interior_cell_pos = interior_cell_pos, lx = lx, ly = ly
        func3 = @inline i -> @inbounds(getindex(cell_pos, i))
        func4 = @inline c -> find_base_point_index(c, interior_cell_pos, lx, ly)
        @batch for v in eachindex(cellsOnVertex)
            gcov = cellsOnVertex_global[interior_vertex_i[v]]
            cps = map(func3, gcov)
            cellsOnVertex[v] = map(func4, cps)
        end
    end

    return interior_cell_pos, interior_vertex_pos, verticesOnCell, cellsOnVertex
end

function VoronoiMeshes.PlanarVoronoiDiagram(initial_generator_points::AbstractVector{<:Vec2Dxy}, lx::Real, ly::Real; density::F = const_density, max_iter::Integer = typemax(Int), rtol::Real = 1.0e-8 / 100, max_time = 4.0) where {F <: Function}

    N = length(initial_generator_points)

    voro = generate_periodic_centroidal_voronoi(
        initial_generator_points, lx, ly;
        density = density, max_iter = max_iter, rtol = rtol, max_time = max_time
    )

    generators, vertices, verticesOnCell, cellsOnVertex = extract_periodic_vertices_and_cells(N, lx, ly, voro)

    meshDensity = Vector{nonzero_eltype(eltype(generators))}(undef, N)

    @batch for i in eachindex(meshDensity)
        meshDensity[i] = density(generators[i])
    end

    maxEdges = maximum(length, verticesOnCell)
    TF = nonzero_eltype(eltype(generators))
    lx_TF = convert(TF, lx)
    ly_TF = convert(TF, ly)
    if maxEdges == 6
        verticesOnCell_6 = SmVecArray{6, Int32}(N)
        verticesOnCell_6 .= verticesOnCell
        return fix_diagram!(PlanarVoronoiDiagram(generators, vertices, verticesOnCell_6, cellsOnVertex, meshDensity, lx_TF, ly_TF))
    elseif maxEdges == 7
        verticesOnCell_7 = SmVecArray{7, Int32}(N)
        verticesOnCell_7 .= verticesOnCell
        return fix_diagram!(PlanarVoronoiDiagram(generators, vertices, verticesOnCell_7, cellsOnVertex, meshDensity, lx_TF, ly_TF))
    elseif maxEdges == 8
        verticesOnCell_8 = SmVecArray{8, Int32}(N)
        verticesOnCell_8 .= verticesOnCell
        return fix_diagram!(PlanarVoronoiDiagram(generators, vertices, verticesOnCell_8, cellsOnVertex, meshDensity, lx_TF, ly_TF))
    elseif maxEdges == 9
        verticesOnCell_9 = SmVecArray{9, Int32}(N)
        verticesOnCell_9 .= verticesOnCell
        return fix_diagram!(PlanarVoronoiDiagram(generators, vertices, verticesOnCell_9, cellsOnVertex, meshDensity, lx_TF, ly_TF))
    elseif maxEdges > 10
        throw(error("Generated Voronoi diagram has polygons of more than 10 sides"))
    else
        verticesOnCell_10 = SmVecArray{10, Int32}(N)
        verticesOnCell_10 .= verticesOnCell
        return fix_diagram!(PlanarVoronoiDiagram(generators, vertices, verticesOnCell_10, cellsOnVertex, meshDensity, lx_TF, ly_TF))
    end
end

function VoronoiMeshes.VoronoiDiagram(initial_generator_points::AbstractVector{<:Vec2Dxy}, lx::Real, ly::Real; density::F = const_density, max_iter::Integer = typemax(Int), rtol::Real = 1.0e-8 / 100, max_time = 4.0) where {F <: Function}
    TF = nonzero_eltype(eltype(initial_generator_points))
    VoronoiDiagram(PlanarVoronoiDiagram(initial_generator_points, convert(TF, lx), convert(TF, ly), density = density, max_iter = max_iter, rtol = rtol, max_time = max_time))
end

function VoronoiMeshes.PlanarVoronoiDiagram(N::Integer, lx::Real, ly::Real; density::F = const_density, max_iter::Integer = typemax(Int), rtol::Real = 1.0e-8 / 100, max_time = 4.0) where {F <: Function}
    points = generate_inital_points(N, lx, ly)
    return VoronoiMeshes.PlanarVoronoiDiagram(points, lx, ly, density = density, max_iter = max_iter, rtol = rtol, max_time = max_time)
end

function VoronoiMeshes.VoronoiDiagram(N::Integer, lx::Real, ly::Real; density::F = const_density, max_iter::Integer = typemax(Int), rtol::Real = 1.0e-8 / 100, max_time = 4.0) where {F <: Function}
    VoronoiDiagram(PlanarVoronoiDiagram(N, lx, ly, density = density, max_iter = max_iter, rtol = rtol, max_time = max_time))
end

function VoronoiMeshes.VoronoiMesh(initial_generator_points::AbstractVector{<:Vec2Dxy}, lx::Real, ly::Real; density::F = const_density, max_iter::Integer = typemax(Int), rtol::Real = 1.0e-8 / 100, max_time = 4.0) where {F <: Function}
    VoronoiMesh(VoronoiDiagram(PlanarVoronoiDiagram(initial_generator_points, lx, ly, density = density, max_iter = max_iter, rtol = rtol, max_time = max_time)))
end

function VoronoiMeshes.VoronoiMesh(N::Integer, lx::Real, ly::Real; density::F = const_density, max_iter::Integer = typemax(Int), rtol::Real = 1.0e-8 / 100, max_time = 4.0) where {F}
    VoronoiMesh(VoronoiDiagram(PlanarVoronoiDiagram(N, lx, ly, density = density, max_iter = max_iter, rtol = rtol, max_time = max_time)))
end

@inline function round_to_even_Int(x::Real)
    ru = round(Int,x,RoundUp)
    rd = round(Int,x,RoundDown)
    if ru == rd
        val = ru + mod(ru,2)
    else
        val = iseven(ru) ? ru : rd
    end
    return val
end

function VoronoiMeshes.create_planar_hex_mesh(lx::Number, ly::Number, dc::Number)
    nx = round(Int,lx/dc)
    ny = round_to_even_Int((2ly)/(√3*dc))
    N = nx*ny
    generators = VecArray(x = zeros(N), y = zeros(N))
    dp = (dc/32) * (𝐢 + 𝐣)

    l = sqrt(3)*dc/3

    I = LinearIndices((Base.OneTo(nx), Base.OneTo(ny)))
    @batch for j in Base.OneTo(ny)
        y = ((3 * (j-1)) // 2 ) * l
        aux = iseven(j)
        @inbounds for i in Base.OneTo(nx)
            x = (i-1)*dc + aux*dc/2
            generators[I[i,j]] = x*𝐢 + y*𝐣 + dp
        end
    end

    xp = Float64(nx*dc)
    yp = ((3*ny) // 2) * l
    return VoronoiMesh(generators, xp, yp, max_iter=0)
end

@compile_workload begin
    m = VoronoiMesh(17, 1.0, 1.0, max_time = 0.1)
    m2 = VoronoiMesh(17, 1.0, 1.0, density=VoronoiMeshes.circular_refinement_function(1.0, 1.0), rtol=1e-4, max_time = 0.1)
    m3 = create_planar_hex_mesh(1.0, 1.0, 0.1)
end

end # Module
