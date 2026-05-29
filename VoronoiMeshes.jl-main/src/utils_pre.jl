"""
Works just like `Threads.@threads` (except you can't interpolate values with `\$`), but runs serially if `Threads.nthreads()==1` to avoid task creation overhead.
"""
macro parallel(ex)
    pex = quote
        let
            if $Threads.nthreads() == 1
                $(ex)
            else
                $Threads.@threads $(ex)
            end
        end
    end
    return esc(pex)
end

function tmap!(output, func::F, var::Vararg) where {F <: Function}
    @parallel for i in eachindex(output)
        @inbounds output[i] = @inline func(map(x -> @inbounds(x[i]), var)...)
    end
    return output
end

function copy_matrix_to_fixedvector_vector!(tuple_vector::AbstractVector{FixedVector{N, T}}, matrix::AbstractMatrix{T2}) where {N, T, T2}
    n = Val{N}()
    @parallel for k in axes(matrix, 2)
        @inbounds tuple_vector[k] = FixedVector(ntuple(i -> (convert(T, @inbounds(matrix[i, k]))), n))
    end
    return tuple_vector
end

for T in (Int32, Int64, Float32, Float64)
    for N in 2:12
        precompile(copy_matrix_to_fixedvector_vector!, (Vector{FixedVector{N, T}}, Matrix{T}))
    end
end

@inline function unsafe_drop_element(t::FixedVector{3}, el::Integer)
    if t[1] == el
        return FixedVector((t[2], t[3]))
    elseif t[2] == el
        return FixedVector((t[1], t[3]))
    else
        return FixedVector((t[1], t[2]))
    end
end

precompile(unsafe_drop_element, (FixedVector{3, Int32}, Int32))
precompile(unsafe_drop_element, (FixedVector{3, Int64}, Int64))

@inline function unsafe_fixedvector2_intersection(t1::FixedVector{2}, t2::FixedVector{2})
    t11, t12 = t1
    t21, t22 = t2
    t11 == t21 ? t11 : t11 == t22 ? t11 : t12
end

precompile(unsafe_fixedvector2_intersection, (FixedVector{2, Int32}, FixedVector{2, Int32}))
precompile(unsafe_fixedvector2_intersection, (FixedVector{2, Int64}, FixedVector{2, Int64}))

@inline function find_cellOnCell(current_cell::Integer, shared_vertex1::Integer, shared_vertex2::Integer, cellsOnVertex::AbstractVector)
    cells_v1 = cellsOnVertex[shared_vertex1]
    cells_v2 = cellsOnVertex[shared_vertex2]
    cells1 = unsafe_drop_element(cells_v1, current_cell)
    cells2 = unsafe_drop_element(cells_v2, current_cell)
    return unsafe_fixedvector2_intersection(cells1, cells2)
end

precompile(find_cellOnCell, (Int32, Int32, Int32, Vector{FixedVector{3, Int32}}))
precompile(find_cellOnCell, (Int64, Int64, Int64, Vector{FixedVector{3, Int64}}))

@inline ordered(x, y) = y < x ? FixedVector((y, x)) : FixedVector((x, y))

function compute_cellsOnCell!(cellsOnCell::SmVecArray, verticesOnCell::AbstractVector, cellsOnVertex::AbstractVector)
    # cellsOnCell[n] should be between verticesOnCell[n] and verticesOnCell[n-1]

    cellsOnCellTuple = cellsOnCell.data
    @parallel for c in eachindex(verticesOnCell)

        @inbounds begin
            cells = eltype(cellsOnCell)()
            vertices = verticesOnCell[c]
            lv = length(vertices)

            vlead = vertices[lv]
            for i in Base.OneTo(lv)
                vprev = vlead
                vlead = vertices[i]
                cells = push(cells, find_cellOnCell(c, vlead, vprev, cellsOnVertex))
            end

            cellsOnCellTuple[c] = fixedvector(cells)
        end
    end
    return cellsOnCell
end

for nEdges in 6:10
    precompile(compute_cellsOnCell!, (SmVecArray{nEdges, Int32, 1}, SmVecArray{nEdges, Int32, 1}, Vector{FixedVector{3, Int32}}))
    precompile(compute_cellsOnCell!, (SmVecArray{nEdges, Int64, 1}, SmVecArray{nEdges, Int64, 1}, Vector{FixedVector{3, Int64}}))
end

function compute_cellsOnCell(verticesOnCell::SmVecArray, cellsOnVertex::AbstractVector)
    cellsOnCell = SmallVectorArray(similar(verticesOnCell.data), verticesOnCell.length)
    return compute_cellsOnCell!(cellsOnCell, verticesOnCell, cellsOnVertex)
end

for nEdges in 6:10
    precompile(compute_cellsOnCell, (SmVecArray{nEdges, Int32, 1}, Vector{FixedVector{3, Int32}}))
    precompile(compute_cellsOnCell, (SmVecArray{nEdges, Int64, 1}, Vector{FixedVector{3, Int64}}))
end

function compute_edgesOnCell!(edgesOnCell, cellsOnCell, cells_pair_to_edge::Dict{FixedVector{2,TI}, TI}) where {TI}

    edgesOnCellTuple = edgesOnCell.data
    @parallel for c in eachindex(cellsOnCell)
        @inbounds begin
            cells = cellsOnCell[c]
            edges = eltype(cellsOnCell)()
            for e in eachindex(cells)
                pair = ordered(TI(c), cells[e])
                e_i = cells_pair_to_edge[pair]
                edges = push(edges, e_i)
            end
            edgesOnCellTuple[c] = fixedvector(edges)
        end
    end
    return edgesOnCell
end

for nEdges in 6:10
    precompile(compute_edgesOnCell!, (SmVecArray{nEdges, Int32, 1}, SmVecArray{nEdges, Int32, 1}, Dict{FixedVector{2, Int32}, Int32}))
    precompile(compute_edgesOnCell!, (SmVecArray{nEdges, Int64, 1}, SmVecArray{nEdges, Int64, 1}, Dict{FixedVector{2, Int64}, Int64}))
end

function compute_edgesOnCell(cellsOnCell::SmVecArray, cells_pair_to_edge::Dict)
    edgesOnCell = SmallVectorArray(similar(cellsOnCell.data), cellsOnCell.length)
    return compute_edgesOnCell!(edgesOnCell, cellsOnCell, cells_pair_to_edge)
end

for nEdges in 6:10
    precompile(compute_edgesOnCell, (SmVecArray{nEdges, Int32, 1}, Dict{FixedVector{2, Int32}, Int32}))
    precompile(compute_edgesOnCell, (SmVecArray{nEdges, Int64, 1}, Dict{FixedVector{2, Int64}, Int32}))
end

function compute_edgesOnVertex!(edgesOnVertex::AbstractVector, cellsOnVertex::AbstractVector, cells_pair_to_edge::Dict{FixedVector{2, TI}, TI}) where {TI}

    @parallel for v in eachindex(edgesOnVertex)
        @inbounds begin
            c1, c2, c3 = cellsOnVertex[v]
            pair1 = ordered(c3, c1)
            e1 = cells_pair_to_edge[pair1]
            pair2 = ordered(c1, c2)
            e2 = cells_pair_to_edge[pair2]
            pair3 = ordered(c2, c3)
            e3 = cells_pair_to_edge[pair3]
            edgesOnVertex[v] = (e1, e2, e3)
        end
    end
    return edgesOnVertex
end

precompile(compute_edgesOnVertex!, (Vector{FixedVector{3, Int32}}, Vector{FixedVector{3, Int32}}, Dict{FixedVector{2, Int32}, Int32}))
precompile(compute_edgesOnVertex!, (Vector{FixedVector{3, Int64}}, Vector{FixedVector{3, Int64}}, Dict{FixedVector{2, Int64}, Int64}))

function compute_edgesOnVertex(cellsOnVertex::Vector{FixedVector{3, TI}}, cells_pair_to_edge::Dict) where {TI}
    edgesOnVertex = Vector{FixedVector{3, TI}}(undef, length(cellsOnVertex))
    return compute_edgesOnVertex!(edgesOnVertex, cellsOnVertex, cells_pair_to_edge)
end

precompile(compute_edgesOnVertex, (Vector{FixedVector{3, Int32}}, Dict{FixedVector{2, Int32}, Int32}))
precompile(compute_edgesOnVertex, (Vector{FixedVector{3, Int64}}, Dict{FixedVector{2, Int64}, Int64}))

function compute_polygon_area_periodic!(output, vpos, verticesOnPolygon, xp::Number, yp::Number)
    @parallel for c in eachindex(verticesOnPolygon)
        @inbounds output[c] = area(vpos, verticesOnPolygon[c], xp, yp)
    end
    return output
end

precompile(compute_polygon_area_periodic!, (Vector{Float64}, Vec2DxyArray{Float64, 1}, Vector{FixedVector{3, Int32}}, Float64, Float64))
for nEdges in 6:10
    precompile(compute_polygon_area_periodic!, (Vector{Float64}, Vec2DxyArray{Float64, 1}, SmVecArray{nEdges, Int32, 1}, Float64, Float64))
end

function compute_polygon_area_spherical!(output, vpos, verticesOnPolygon, R::Number)
    @parallel for c in eachindex(verticesOnPolygon)
        @inbounds output[c] = spherical_polygon_area(R, vpos, verticesOnPolygon[c])
    end
    return output
end

precompile(compute_polygon_area_spherical!, (Vector{Float64}, Vec3DArray{Float64, 1}, Vector{FixedVector{3, Int32}}, Float64))
for nEdges in 6:10
    precompile(compute_polygon_area_spherical!, (Vector{Float64}, Vec3DArray{Float64, 1}, SmVecArray{nEdges, Int32, 1}, Float64))
end

function compute_polygon_centroid_periodic!(output, vpos, verticesOnPolygon, xp::Number, yp::Number)
    @parallel for c in eachindex(verticesOnPolygon)
        @inbounds output[c] = centroid(vpos, verticesOnPolygon[c], xp, yp)
    end
    return output
end

precompile(compute_polygon_centroid_periodic!, (Vec2DxyArray{Float64, 1}, Vec2DxyArray{Float64, 1}, Vector{FixedVector{3, Int32}}, Float64, Float64))
for nEdges in 6:10
    precompile(compute_polygon_centroid_periodic!, (Vec2DxyArray{Float64, 1}, Vec2DxyArray{Float64, 1}, SmVecArray{nEdges, Int32, 1}, Float64, Float64))
end

function compute_polygon_centroid_spherical!(output, vpos, verticesOnPolygon, R::Number)
    @parallel for c in eachindex(verticesOnPolygon)
        @inbounds output[c] = spherical_polygon_centroid(R, vpos, verticesOnPolygon[c])
    end
    return output
end
precompile(compute_polygon_centroid_spherical!, (Vec3DArray{Float64, 1}, Vec3DArray{Float64, 1}, Vector{FixedVector{3, Int32}}, Float64))
for nEdges in 6:10
    precompile(compute_polygon_centroid_spherical!, (Vec3DArray{Float64, 1}, Vec3DArray{Float64, 1}, SmVecArray{nEdges, Int32, 1}, Float64))
end

function compute_longitude_periodic(cpos::Vec2DxyArray{T, 1}) where {T}
    return zeros(T, length(cpos))
end
precompile(compute_longitude_periodic, (Vec2DxyArray{Float64, 1},))

function compute_longitude!(longitude::Vector, pos::TensorArray)
    px = pos.x
    py = pos.y
    @parallel for c in eachindex(pos)
        @inbounds begin
            longitude[c] = atan(py[c], px[c])
        end
    end
    return longitude
end
precompile(compute_longitude!, (Vector{Float64}, Vec3DArray{Float64, 1}))

function compute_latitude_periodic(cpos::Vec2DxyArray{T, 1}) where {T}
    return zeros(T, length(cpos))
end
precompile(compute_latitude_periodic, (Vec2DxyArray{Float64, 1},))

function compute_latitude!(latitude::Vector, pos::TensorArray)
    px = pos.x
    py = pos.y
    pz = pos.z
    @parallel for c in eachindex(pos)
        @inbounds begin
            x = px[c]
            y = py[c]
            z = pz[c]
            latitude[c] = atan(z / hypot(x, y))
        end
    end
    return latitude
end
precompile(compute_latitude!, (Vector{Float64}, Vec3DArray{Float64, 1}))

function compute_zonalVector_periodic(cpos::Vec2DxyArray{T, 1}) where {T}
    return VecArray(x = ones(T, length(cpos)))
end
precompile(compute_zonalVector_periodic, (Vec2DxyArray{Float64, 1},))

function compute_zonalVector!(zonalVector::TensorArray, pos::TensorArray)
    px = pos.x
    py = pos.y
    @parallel for c in eachindex(pos)
        @inbounds begin
            ϕ = atan(py[c], px[c])
            zonalVector[c] = cos(ϕ) * 𝐣 - sin(ϕ)𝐢
        end
    end
    return zonalVector
end
precompile(compute_zonalVector!, (Vec2DxyArray{Float64, 1}, Vec3DArray{Float64, 1}))

function compute_meridionalVector_periodic(cpos::Vec2DxyArray{T, 1}) where {T}
    return VecArray(y = ones(T, length(cpos)))
end
precompile(compute_meridionalVector_periodic, (Vec2DxyArray{Float64, 1},))

function compute_meridionalVector!(meridionalVector::TensorArray, pos::TensorArray)
    px = pos.x
    py = pos.y
    pz = pos.z
    @parallel for c in eachindex(pos)
        @inbounds begin
            x = px[c]
            y = py[c]
            z = pz[c]
            ϕ = atan(y, x)
            θ = atan(hypot(x, y), z)
            sinϕ = sin(ϕ)
            cosϕ = cos(ϕ)
            sinθ = sin(θ)
            cosθ = cos(θ)
            meridionalVector[c] = sinθ * 𝐤 - (cosϕ * cosθ) * 𝐢 - (sinϕ * cosθ) * 𝐣
        end
    end
    return meridionalVector
end
precompile(compute_meridionalVector!, (Vec3DArray{Float64, 1}, Vec3DArray{Float64, 1}))

"""
    select_kite_area(kiteAreaOnVertex::Vector{FixedVector{3}}, cellsOnVertex::Vector{FixedVector{3,<:Integer}}, v_i::Integer, c_i::Integer)
Returns the kite Area associated with cell `c_i` and vertex `v_i`.
Throws an error if vertex `v_i` doens't belong to cell `c_i`.
"""
function select_kite_area(kiteAreaOnVertex,cellsOnVertex,v,c)
    areas = kiteAreaOnVertex[v]
    cells = cellsOnVertex[v]
    if cells[1] == c
        return areas[1]
    elseif cells[2] == c
        return areas[2]
    elseif cells[3] == c
        return areas[3]
    else
        error("Vertex $v doesn't belong to cell $c")
    end
end

function sign_edge((c1,c2),c)
    if c == c1
        return 1
    elseif c == c2
        return -1
    else
        error("Edge doesn't belong to cell $c")
    end
end

function counter_clockwise_test(reference, p1, p2, normal)
    !signbit(((p1 - reference) × (p2 - reference)) ⋅ normal)
end

#spherical
function check_if_counter_clockwise(R::Number, reference_position, indicesOnReference, referee_position)
    r = Int[]
    lk = ReentrantLock()
    @parallel for e in eachindex(reference_position)
        @inbounds begin
            ref = reference_position[e]
            inds = indicesOnReference[e]
            p1 = referee_position[inds[1]]
            p2 = referee_position[inds[2]]
            if !counter_clockwise_test(ref, p1, p2, ref / R)
                lock(lk) do
                    push!(r,e)
                end
            end
        end #inbounds
    end
    n_problems = length(r)
    return n_problems == 0 ? nothing : r
end

#periodic
function check_if_counter_clockwise(reference_position, indicesOnReference, referee_position, xp::Number, yp::Number)
    r = Int[]
    lk = ReentrantLock()
    @parallel for e in eachindex(reference_position)
        @inbounds begin
            ref = reference_position[e]
            inds = indicesOnReference[e]
            p1 = closest(ref, referee_position[inds[1]], xp ,yp)
            p2 = closest(ref, referee_position[inds[2]], xp, yp)
            if !counter_clockwise_test(ref, p1, p2, 𝐤)
                lock(lk) do
                    push!(r,e)
                end
            end
        end #inbounds
    end
    n_problems = length(r)
    return n_problems == 0 ? nothing : r
end

for N in 6:9
    precompile(Tuple{typeof(check_if_counter_clockwise), Vec2DxyArray{Float64,1}, SmallVectorArray{N, Int32, 1, Array{FixedVector{N, Int32}, 1}}, Vec2DxyArray{Float64,1}, Float64, Float64})
end
precompile(Tuple{typeof(check_if_counter_clockwise), Vec2DxyArray{Float64,1}, Array{Tuple{Int32, Int32, Int32}, 1}, Vec2DxyArray{Float64,1}, Float64, Float64})

function check_indices_ordering(R::Number, ref_position, indOnRef1, rferee_pos1, indOnRef2, rferee_pos2)
    r = Int[]
    lk = ReentrantLock()
    @parallel for e in eachindex(ref_position)
        @inbounds begin
            ref = ref_position[e]
            p1 = rferee_pos1[indOnRef1[e][1]]
            p2 = rferee_pos2[indOnRef2[e][1]]
            if !counter_clockwise_test(ref, p1, p2, ref / R)
                lock(lk) do
                    push!(r,e)
                end
            end
        end #inbounds
    end
    n_problems = length(r)
    return n_problems == 0 ? nothing : r
end

function check_indices_ordering(ref_position, indOnRef1, rferee_pos1, indOnRef2, rferee_pos2, xp::Number, yp::Number)
    r = Int[]
    lk = ReentrantLock()
    @parallel for e in eachindex(ref_position)
        @inbounds begin
            ref = ref_position[e]
            p1 = closest(ref, rferee_pos1[indOnRef1[e][1]], xp, yp)
            p2 = closest(ref, rferee_pos2[indOnRef2[e][1]], xp, yp)
            if !counter_clockwise_test(ref, p1, p2, 𝐤)
                lock(lk) do
                    push!(r,e)
                end
            end
        end #inbounds
    end
    n_problems = length(r)
    return n_problems == 0 ? nothing : r
end
