compute_cell_area!(output, cells::Cells{false}) = compute_polygon_area_periodic!(output, cells.info.diagram.vertices, cells.vertices, cells.x_period, cells.y_period)
compute_cell_area!(output, cells::Cells{true}) = compute_polygon_area_spherical!(output, cells.info.diagram.vertices, cells.vertices, cells.sphere_radius)
compute_cell_area(cells::Cells) = compute_cell_area!(similar(cells.position.x), cells)

compute_cell_centroid!(output, cells::Cells{false}) = compute_polygon_centroid_periodic!(output, cells.info.diagram.vertices, cells.vertices, cells.x_period, cells.y_period)
compute_cell_centroid!(output, cells::Cells{true}) = compute_polygon_centroid_spherical!(output, cells.info.diagram.vertices, cells.vertices, cells.sphere_radius)
compute_cell_centroid(cells::Cells) = compute_cell_centroid!(similar(cells.position), cells)

compute_cell_longitude!(output, cells::Cells{false}) = fill!(output, zero(float_type(cells)))
compute_cell_longitude!(output, cells::Cells{true}) = compute_longitude!(output, cells.position)
compute_cell_longitude(cells::Cells) = compute_cell_longitude!(similar(cells.position.x), cells)

compute_cell_latitude!(output, cells::Cells{false}) = fill!(output, zero(float_type(cells)))
compute_cell_latitude!(output, cells::Cells{true}) = compute_latitude!(output, cells.position)
compute_cell_latitude(cells::Cells) = compute_cell_latitude!(similar(cells.position.x), cells)

compute_cell_normal!(output, ::Cells{false}) = fill!(output, 𝐤)
compute_cell_normal(cells::Cells{false}) = compute_cell_normal!(VecArray(z = similar(cells.position.x)), cells)
compute_cell_normal!(output, cells::Cells{true}) = tmap!(output, normalize, cells.position)
compute_cell_normal(cells::Cells{true}) = compute_cell_normal!(similar(cells.position), cells)

compute_cell_zonalVector!(output, ::Cells{false}) = fill!(output, 𝐢)
compute_cell_zonalVector!(output, cells::Cells{true}) = compute_zonalVector!(output, cells.position)
compute_cell_zonalVector(cells::Cells{false}) = compute_cell_zonalVector!(VecArray(x = similar(cells.position.x)), cells)
compute_cell_zonalVector(cells::Cells{true}) = compute_cell_zonalVector!(VecArray(x = similar(cells.position.x), y = similar(cells.position.y)), cells)

compute_cell_meridionalVector!(output, ::Cells{false}) = fill!(output, 𝐣)
compute_cell_meridionalVector(cells::Cells{false}) = compute_cell_meridionalVector!(VecArray(y = similar(cells.position.y)), cells)
compute_cell_meridionalVector!(output, cells::Cells{true}) = compute_meridionalVector!(output, cells.position)
compute_cell_meridionalVector(cells::Cells{true}) = compute_cell_meridionalVector!(similar(cells.position), cells)

function compute_cell_edgesSign!(edgesSign::SmVecArray{N, TF}, edgesOnCell, cellsOnEdge) where {N, TF<:AbstractFloat}
    data = edgesSign.data

    @parallel for c in eachindex(edgesOnCell)
        @inbounds begin
            eoc = edgesOnCell[c]

            r = SmallVector{N, TF}()
            for e in eoc
                c1e, c2e = cellsOnEdge[e]
                r = push(r, c1e == c ? TF(1) : c2e == c ? TF(-1) : TF(NaN)) # If everything is correct with the data should never be a NaN
            end

            data[c] = fixedvector(r)
        end
    end

    return edgesSign
end

function compute_cell_edgesSign(cells::Cells{S, N_MAX, TI, TF}) where {S, N_MAX, TI, TF}
    edgesSign = SmallVectorArray(Vector{FixedVector{N_MAX, TF}}(undef, cells.n), cells.nEdges)
    cellsOnEdge = getfield(getfield(cells, :info), :cellsOnEdge)
    return compute_cell_edgesSign!(edgesSign, cells.edges, cellsOnEdge)
end

function compute_cell_areaMimetic_spherical!(output::AbstractVector, cpos::VecArray, cellsOnCell, verticesOnCell, vpos, R::Number)

    @parallel for c in eachindex(output)
        @inbounds begin

            coc = cellsOnCell[c]
            voc = verticesOnCell[c]
            n = length(coc)

            cp1 = cpos[c]
            vp1 = vpos[voc[n]]
            r = zero(eltype(output))
            for i in 1:n
                vp2 = vpos[voc[i]]
                cp2 = cpos[coc[i]]
                dc = arc_length(R, cp1, cp2)
                le = arc_length(R, vp1, vp2)
                r += le * dc / 4

                vp1 = vp2
            end

            output[c] = r
        end
    end

    return output
end

compute_cell_areaMimetic!(output, cells::Cells{false}) = compute_polygon_area_periodic!(output, cells.info.diagram.vertices, cells.vertices, cells.x_period, cells.y_period)
compute_cell_areaMimetic!(output, cells::Cells{true}) = compute_cell_areaMimetic_spherical!(output, cells.position, cells.cells, cells.vertices, cells.info.diagram.vertices, cells.sphere_radius)
compute_cell_areaMimetic(cells::Cells) = compute_cell_areaMimetic!(similar(cells.position.x), cells)
