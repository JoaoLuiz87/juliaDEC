function compute_edge_midpoint_periodic!(output, vpos, verticesOnEdge, xp, yp)
    @parallel for e in eachindex(verticesOnEdge)
        @inbounds begin
            v1, v2 = verticesOnEdge[e]
            vp1 = vpos[v1]
            vp2 = closest(vp1, vpos[v2], xp, yp)
            output[e] = (vp1 + vp2) / 2
        end
    end
    return output
end

compute_edge_midpoint!(output, edges::Edges{false}) = compute_edge_midpoint_periodic!(output, edges.info.diagram.vertices, edges.vertices, edges.x_period, edges.y_period)

function compute_edge_midpoint_spherical!(output, vpos, verticesOnEdge, R)
    @parallel for e in eachindex(verticesOnEdge)
        @inbounds begin
            v1, v2 = verticesOnEdge[e]
            output[e] = arc_midpoint(R, vpos[v1], vpos[v2])
        end
    end
    return output
end

compute_edge_midpoint!(output, edges::Edges{true}) = compute_edge_midpoint_spherical!(output, edges.info.diagram.vertices, edges.vertices, edges.sphere_radius)
compute_edge_midpoint(edges::Edges) = compute_edge_midpoint!(similar(edges.position), edges)

function compute_edge_length_periodic!(output, vpos, verticesOnEdge, xp, yp)
    @parallel for e in eachindex(verticesOnEdge)
        @inbounds begin
            v1, v2 = verticesOnEdge[e]
            vp1 = vpos[v1]
            vp2 = closest(vp1, vpos[v2], xp, yp)
            output[e] = norm(vp2 - vp1)
        end
    end
    return output
end

compute_edge_length!(output, edges::Edges{false}) = compute_edge_length_periodic!(output, edges.info.diagram.vertices, edges.vertices, edges.x_period, edges.y_period)

function compute_edge_length_spherical!(output, vpos, verticesOnEdge, R)
    @parallel for e in eachindex(verticesOnEdge)
        @inbounds begin
            v1, v2 = verticesOnEdge[e]
            output[e] = arc_length(R, vpos[v1], vpos[v2])
        end
    end
    return output
end

compute_edge_length!(output, edges::Edges{true}) = compute_edge_length_spherical!(output, edges.info.diagram.vertices, edges.vertices, edges.sphere_radius)
compute_edge_length(edges::Edges) = compute_edge_length!(similar(edges.position.x), edges)

compute_edge_lengthDual!(output, edges::Edges{false}) = compute_edge_length_periodic!(output, edges.info.diagram.generators, edges.cells, edges.x_period, edges.y_period)
compute_edge_lengthDual!(output, edges::Edges{true}) = compute_edge_length_spherical!(output, edges.info.diagram.generators, edges.cells, edges.sphere_radius)
compute_edge_lengthDual(edges::Edges) = compute_edge_lengthDual!(similar(edges.position.x), edges)

function compute_edge_angle_periodic!(output, cpos, cellsOnEdge, xp::Number, yp::Number)
    @parallel for i in eachindex(cellsOnEdge)
        @inbounds begin
            ic1, ic2 = cellsOnEdge[i]
            cpos2 = cpos[ic2]
            cpos1 = closest(cpos2, cpos[ic1], xp, yp)
            𝐧 = normalize(cpos2 - cpos1)
            # The `angle` function below always return a positive value
            output[i] = copysign(TensorsLiteGeometry.angle(𝐧, 𝐢), 𝐧.y)
        end
    end
    return output
end

compute_edge_angle!(output, edges::Edges{false}) = compute_edge_angle_periodic!(output, edges.info.diagram.generators, edges.cells, edges.x_period, edges.y_period)

function compute_edge_angle_spherical!(output, epos, cpos, cellsOnEdge)
    @parallel for i in eachindex(cellsOnEdge)
        @inbounds begin
            ic1, ic2 = cellsOnEdge[i]
            cpos2 = cpos[ic2]
            cpos1 = cpos[ic1]
            normal = normalize(cpos2 - cpos1)
            ep = epos[i]
            zonalVec = eastward_vector(ep)
            # The `angle` function below always return a positive value
            output[i] = copysign(TensorsLiteGeometry.angle(normal, zonalVec), (zonalVec × normal) ⋅ ep)
        end
    end
    return output
end

compute_edge_angle!(output, edges::Edges{true}) = compute_edge_angle_spherical!(output, edges.position, edges.info.diagram.generators, edges.cells)
compute_edge_angle(edges::Edges) = compute_edge_angle!(similar(edges.position.x), edges)

compute_edge_longitude!(output, edges::Edges{false}) = fill!(output, zero(float_type(edges)))
compute_edge_longitude!(output, edges::Edges{true}) = compute_longitude!(output, edges.position)
compute_edge_longitude(edges::Edges) = compute_edge_longitude!(similar(edges.position.x), edges)

compute_edge_latitude!(output, edges::Edges{false}) = fill!(output, zero(float_type(edges)))
compute_edge_latitude!(output, edges::Edges{true}) = compute_latitude!(output, edges.position)
compute_edge_latitude(edges::Edges) = compute_edge_latitude!(similar(edges.position.x), edges)

function compute_edge_normal_periodic!(output, cpos, cellsOnEdge, xp::Number, yp::Number)
    @parallel for i in eachindex(cellsOnEdge)
        @inbounds begin
            ic1, ic2 = cellsOnEdge[i]
            cpos2 = cpos[ic2]
            cpos1 = closest(cpos2, cpos[ic1], xp, yp)
            output[i] = normalize(cpos2 - cpos1)
        end
    end
    return output
end

compute_edge_normal!(output, edges::Edges{false}) = compute_edge_normal_periodic!(output, edges.info.diagram.generators, edges.cells, edges.x_period, edges.y_period)

function compute_edge_normal_spherical!(output, cpos, cellsOnEdge)
    @parallel for i in eachindex(cellsOnEdge)
        @inbounds begin
            ic1, ic2 = cellsOnEdge[i]
            cpos2 = cpos[ic2]
            cpos1 = cpos[ic1]
            #since the edge position is at the arcmidpoint between cpos2 and cpos1, the arc
            #tangent is parallel to the vector connecting the points.
            normal = normalize(cpos2 - cpos1)
            output[i] = normal
        end
    end
    return output
end

compute_edge_normal!(output, edges::Edges{true}) = compute_edge_normal_spherical!(output, edges.info.diagram.generators, edges.cells)
compute_edge_normal(edges::Edges) = compute_edge_normal!(similar(edges.position), edges)
                                                     #yes this is correct
compute_edge_tangent!(output, edges::Edges{false}) = compute_edge_normal_periodic!(output, edges.info.diagram.vertices, edges.vertices, edges.x_period, edges.y_period)

function compute_edge_tangent_spherical!(t, R::Number,  epos, vpos, verticesOnEdge)
    @parallel for i in eachindex(verticesOnEdge)
        @inbounds begin
            _, iv2 = verticesOnEdge[i]
            vp = vpos[iv2] / R
            ep = epos[i] / R
            t[i] = normalize(ep × normalize(vp × ep))
        end
    end

    return t
end

compute_edge_tangent!(output, edges::Edges{true}) = compute_edge_tangent_spherical!(output, edges.sphere_radius, edges.position, edges.info.diagram.vertices, edges.vertices)
compute_edge_tangent(edges::Edges) = compute_edge_tangent!(similar(edges.position), edges)
