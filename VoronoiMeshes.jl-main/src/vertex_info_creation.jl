compute_vertex_area!(output, vertices::Vertices{false}) = compute_polygon_area_periodic!(output, vertices.info.diagram.generators, vertices.cells, vertices.x_period, vertices.y_period)
compute_vertex_area!(output, vertices::Vertices{true}) = compute_polygon_area_spherical!(output, vertices.info.diagram.generators, vertices.cells, vertices.sphere_radius)
compute_vertex_area(vertices::Vertices) = compute_vertex_area!(similar(vertices.position.x), vertices)

compute_vertex_centroid!(output, vertices::Vertices{false}) = compute_polygon_centroid_periodic!(output, vertices.info.diagram.generators, vertices.cells, vertices.x_period, vertices.y_period)
compute_vertex_centroid!(output, vertices::Vertices{true}) = compute_polygon_centroid_spherical!(output, vertices.info.diagram.generators, vertices.cells, vertices.sphere_radius)
compute_vertex_centroid(vertices::Vertices) = compute_vertex_centroid!(similar(vertices.position), vertices)

compute_vertex_longitude!(output, vertices::Vertices{false}) = fill!(output, zero(float_type(vertices)))
compute_vertex_longitude!(output, vertices::Vertices{true}) = compute_longitude!(output, vertices.position)
compute_vertex_longitude(vertices::Vertices) = compute_vertex_longitude!(similar(vertices.position.x), vertices)

compute_vertex_latitude!(output, vertices::Vertices{false}) = fill!(output, zero(float_type(vertices)))
compute_vertex_latitude!(output, vertices::Vertices{true}) = compute_latitude!(output, vertices.position)
compute_vertex_latitude(vertices::Vertices) = compute_vertex_latitude!(similar(vertices.position.x), vertices)

function compute_kite_areas_periodic!(output, cpos, vpos, cellsOnVertex, xp, yp)

    @parallel for v in eachindex(cellsOnVertex)
        @inbounds begin
            v_pos = vpos[v]
            c1, c2, c3 = cellsOnVertex[v]
            c1_pos = closest(v_pos, cpos[c1], xp, yp)
            c2_pos = closest(v_pos, cpos[c2], xp, yp)
            c3_pos = closest(v_pos, cpos[c3], xp, yp)

            #Those should be the edges positions
            c12_pos = 0.5 * (c1_pos + c2_pos)
            c23_pos = 0.5 * (c2_pos + c3_pos)
            c31_pos = 0.5 * (c3_pos + c1_pos)

            a1 = area(c1_pos, c12_pos, v_pos, c31_pos)
            a2 = area(c12_pos, c2_pos, c23_pos, v_pos)
            a3 = area(c23_pos, c3_pos, c31_pos, v_pos)
            output[v] = (a1, a2, a3)
        end
    end
    return output
end

function compute_kite_areas_spherical!(output, cpos, vpos, cellsOnVertex, R)

    @parallel for v in eachindex(cellsOnVertex)
        @inbounds begin
            v_pos = vpos[v]
            c1, c2, c3 = cellsOnVertex[v]
            c1_pos = cpos[c1]
            c2_pos = cpos[c2]
            c3_pos = cpos[c3]

            #Those should be the edges positions
            c12_pos = arc_midpoint(R, c1_pos, c2_pos)
            c23_pos = arc_midpoint(R, c2_pos, c3_pos)
            c31_pos = arc_midpoint(R, c3_pos, c1_pos)

            a1 = spherical_polygon_area(R, c1_pos, c12_pos, v_pos, c31_pos)
            a2 = spherical_polygon_area(R, c12_pos, c2_pos, c23_pos, v_pos)
            a3 = spherical_polygon_area(R, c23_pos, c3_pos, c31_pos, v_pos)
            output[v] = (a1, a2, a3)
        end
    end
    return output
end

compute_vertex_kiteAreas!(output, vertices::Vertices{false}) = compute_kite_areas_periodic!(
    output, vertices.info.diagram.generators, vertices.position,
    vertices.cells, vertices.x_period, vertices.y_period
)
compute_vertex_kiteAreas!(output, vertices::Vertices{true}) = compute_kite_areas_spherical!(
    output, vertices.info.diagram.generators, vertices.position,
    vertices.cells, vertices.sphere_radius
)
compute_vertex_kiteAreas(vertices::Vertices) = compute_vertex_kiteAreas!(Vector{FixedVector{3, float_type(vertices)}}(undef, vertices.n), vertices)

function compute_vertex_edgesSign!(edgesSign::AbstractVector{FixedVector{3, TF}}, edgesOnVertex, verticesOnEdge) where {TF<:AbstractFloat}

    @parallel for v in eachindex(edgesOnVertex)
        @inbounds begin
            e1, e2, e3 = edgesOnVertex[v]

            v1e1, v2e1 = verticesOnEdge[e1]
            r1 = v1e1 == v ? TF(-1) : v2e1 == v ? TF(1) : TF(NaN)

            v1e2, v2e2 = verticesOnEdge[e2]
            r2 = v1e2 == v ? TF(-1) : v2e2 == v ? TF(1) : TF(NaN)

            v1e3, v2e3 = verticesOnEdge[e3]
            r3 = v1e3 == v ? TF(-1) : v2e3 == v ? TF(1) : TF(NaN)

            # If everything is correct with the data should never have a NaN
            edgesSign[v] = (r1, r2, r3)
        end
    end

    return edgesSign
end

function compute_vertex_edgesSign(vertices::Vertices{S, N_MAX, TI, TF}) where {S, N_MAX, TI, TF}
    edgesSign = Vector{FixedVector{3, TF}}(undef, vertices.n)
    verticesOnEdge = getfield(getfield(vertices, :info), :verticesOnEdge)
    return compute_vertex_edgesSign!(edgesSign, vertices.edges, verticesOnEdge)
end

function compute_vertex_areaMimetic_spherical!(output::AbstractVector, vpos::VecArray, cellsOnVertex, cpos, R::Number)

    @parallel for v in eachindex(cellsOnVertex)
        @inbounds begin
            vp = vpos[v]
            cov = cellsOnVertex[v]

            r = zero(eltype(output))
            cp1 = cpos[cov[3]]
            for i in 1:3
                cp2 = cpos[cov[i]]
                dc = arc_length(R,cp1, cp2)
                dv = arc_length(R, vp, arc_midpoint(R, cp1, cp2))

                r += dc * dv / 2
                cp1 = cp2
            end

            output[v] = r
        end
    end

    return output
end

compute_vertex_areaMimetic!(output, vertices::Vertices{false}) = compute_polygon_area_periodic!(output, vertices.info.diagram.generators, vertices.cells, vertices.x_period, vertices.y_period)
compute_vertex_areaMimetic!(output, vertices::Vertices{true}) = compute_vertex_areaMimetic_spherical!(output, vertices.position, vertices.cells, vertices.info.diagram.generators, vertices.sphere_radius)
compute_vertex_areaMimetic(vertices::Vertices) = compute_vertex_areaMimetic!(similar(vertices.position.x), vertices)

function compute_kite_areas_mimetic_spherical!(output::AbstractVector, cpos::VecArray, vpos::VecArray, cellsOnVertex, R::Number)

    @parallel for v in eachindex(cellsOnVertex)
        @inbounds begin
            v_pos = vpos[v]
            c1, c2, c3 = cellsOnVertex[v]
            c1_pos = cpos[c1]
            c2_pos = cpos[c2]
            c3_pos = cpos[c3]

            #Those should be the edges positions
            c12_pos = arc_midpoint(R, c1_pos, c2_pos)
            c23_pos = arc_midpoint(R, c2_pos, c3_pos)
            c31_pos = arc_midpoint(R, c3_pos, c1_pos)

            l1 = arc_length(R, v_pos, c12_pos)
            l2 = arc_length(R, v_pos, c23_pos)
            l3 = arc_length(R, v_pos, c31_pos)

            t1 = l1 * arc_length(R, c1_pos, c2_pos) / 2
            t2 = l2 * arc_length(R, c2_pos, c3_pos) / 2
            t3 = l3 * arc_length(R, c3_pos, c1_pos) / 2
            

            a1 = (t1 + t3) / 2
            a2 = (t1 + t2) / 2 
            a3 = (t2 + t3) / 2 
            output[v] = (a1, a2, a3)
        end
    end
    return output
end

compute_vertex_kiteAreasMimetic!(output, vertices::Vertices{false}) = compute_kite_areas_periodic!(
    output, vertices.info.diagram.generators, vertices.position,
    vertices.cells, vertices.x_period, vertices.y_period
)
compute_vertex_kiteAreasMimetic!(output, vertices::Vertices{true}) = compute_kite_areas_mimetic_spherical!(
    output, vertices.info.diagram.generators, vertices.position,
    vertices.cells, vertices.sphere_radius
)
compute_vertex_kiteAreasMimetic(vertices::Vertices) = compute_vertex_kiteAreasMimetic!(Vector{FixedVector{3, float_type(vertices)}}(undef, vertices.n), vertices)
