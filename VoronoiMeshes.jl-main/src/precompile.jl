@setup_workload begin

    include("mesh_plane_inc.jl")

    for verticesOnCell in (verticesOnCell6, verticesOnCell7, verticesOnCell8)
        @compile_workload begin
            d = VoronoiDiagram(PlanarVoronoiDiagram(generators, vertices, verticesOnCell, cellsOnVertex, meshDensity, x_period, y_period))
            m = VoronoiMesh(fix_diagram!(d))

            m.cells.area
            m.cells.centroid
            m.cells.edgesSign
            m.cells.normal
            m.cells.zonalVector
            m.cells.meridionalVector

            m.vertices.area
            m.vertices.kiteAreas
            m.vertices.centroid
            m.vertices.edgesSign

            m.edges.length
            m.edges.lengthDual
            m.edges.midpoint
            m.edges.normal
            m.edges.tangent
            m.edges.angle

            graph_partition(m)

            warn_mesh_issues(m)

            find_obtuse_triangles(m)
        end
    end

    include("mesh_sph_inc.jl")

    for verticesOnCell in (verticesOnCell6, verticesOnCell7, verticesOnCell8)
        @compile_workload begin
            d = VoronoiDiagram(SphericalVoronoiDiagram(generators, vertices, verticesOnCell, cellsOnVertex, meshDensity, sphere_radius))
            m = VoronoiMesh(fix_diagram!(d))

            m.cells.area
            m.cells.centroid
            m.cells.edgesSign
            m.cells.longitude
            m.cells.latitude
            m.cells.normal
            m.cells.zonalVector
            m.cells.meridionalVector
            m.cells.areaMimetic

            m.vertices.area
            m.vertices.kiteAreas
            m.vertices.centroid
            m.vertices.edgesSign
            m.vertices.longitude
            m.vertices.latitude
            m.vertices.areaMimetic
            m.vertices.kiteAreasMimetic

            m.edges.length
            m.edges.lengthDual
            m.edges.midpoint
            m.edges.normal
            m.edges.tangent
            m.edges.angle
            m.edges.longitude
            m.edges.latitude

            graph_partition(m)

            warn_mesh_issues(m)
        end    
    end

end
