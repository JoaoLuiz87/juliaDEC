using VoronoiMeshes, TensorsLite, SmallCollections, TensorsLiteGeometry, NCDatasets, LinearAlgebra, DelaunayTriangulation
using Test
using WriteVTK, ReadVTK     # For saving meshes in VTU format

function my_approx(a, b; atol = 0.0, rtol = sqrt(eps(Float64)))
    all(x -> isapprox(x[1], x[2], atol = atol, rtol = rtol), zip(a, b))
end

function my_approx_periodic(a, b, xp::Number, yp::Number; atol = 0.0, rtol = sqrt(eps(Float64)))
    all(x -> isapprox_periodic(x[1], x[2], xp, yp, atol = atol, rtol = rtol), zip(a, b))
end

function my_approx(a::AbstractVector{<:NTuple{N}}, b::AbstractVector{<:NTuple{N}}; atol = 0.0, rtol = sqrt(eps(Float64))) where {N}
    all(x -> mapreduce((y, z) -> isapprox(y, z, atol = atol, rtol = rtol), &, x[1], x[2]), zip(a, b))
end

fix_longitude(lon::Number) = lon > π ? lon - 2π : lon
fix_longitude(lonVec::Vector) = map(fix_longitude, lonVec)

@testset "CellInfo creation" begin
    mesh_iso = VoronoiMesh("mesh.nc")
    mesh_dist = VoronoiMesh("mesh_distorted.nc")
    mesh_spherical = VoronoiMesh("spherical_grid_500km.nc")

    for mesh in (mesh_iso, mesh_dist, mesh_spherical)
        cells = mesh.cells
        if mesh === mesh_iso
            xp = mesh.x_period
            yp = mesh.y_period
            @test my_approx_periodic(cells.position, cells.centroid, xp, yp, atol = 1e-8)
        end
        if mesh === mesh_spherical
            @test my_approx(cells.position, cells.centroid, atol = 5.0e-2)
        end
        @test my_approx(cells.area, VoronoiMeshes.compute_cell_area(cells), atol = 1e-8)
        @test my_approx(fix_longitude(cells.longitude), fix_longitude(VoronoiMeshes.compute_cell_longitude(cells)))
        @test my_approx(cells.latitude, VoronoiMeshes.compute_cell_latitude(cells))

        zonal = mesh.cells.zonalVector
        meridional = mesh.cells.meridionalVector
        normal = mesh.cells.normal
        @test my_approx(normal, zonal .× meridional)
        @test my_approx(zonal, meridional .× normal)
        @test my_approx(meridional, normal .× zonal)
    end
end

@testset "VertexInfo creation" begin
    mesh_iso = VoronoiMesh("mesh.nc")
    mesh_dist = VoronoiMesh("mesh_distorted.nc")
    mesh_spherical = VoronoiMesh("spherical_grid_500km.nc")

    for mesh in (mesh_iso, mesh_dist, mesh_spherical)
        vertices = mesh.vertices
        if mesh === mesh_iso
            xp = mesh.x_period
            yp = mesh.y_period
            @test my_approx_periodic(vertices.position, vertices.centroid, xp, yp)
        end
        @test my_approx(vertices.area, VoronoiMeshes.compute_vertex_area(vertices), atol = 1e-8)
        @test my_approx(fix_longitude(vertices.longitude), fix_longitude(VoronoiMeshes.compute_vertex_longitude(vertices)))
        @test my_approx(vertices.latitude, VoronoiMeshes.compute_vertex_latitude(vertices))
        @test my_approx(vertices.kiteAreas, VoronoiMeshes.compute_vertex_kiteAreas(vertices), atol = 1e-8)

    end
end

@testset "EdgeInfo creation" begin
    mesh_iso = VoronoiMesh("mesh.nc")
    mesh_dist = VoronoiMesh("mesh_distorted.nc")
    mesh_spherical = VoronoiMesh("spherical_grid_500km.nc")

    for mesh in (mesh_iso, mesh_dist, mesh_spherical)
        edges = mesh.edges
        if mesh === mesh_iso
            xp = mesh.x_period
            yp = mesh.y_period
            @test my_approx_periodic(edges.position, edges.midpoint, xp, yp)
        end
        @test my_approx(edges.length, VoronoiMeshes.compute_edge_length(edges), atol = 1e-8)
        @test my_approx(edges.lengthDual, VoronoiMeshes.compute_edge_lengthDual(edges), atol = 1e-8)
        @test my_approx(fix_longitude(edges.longitude), fix_longitude(VoronoiMeshes.compute_edge_longitude(edges)))
        @test my_approx(edges.latitude, VoronoiMeshes.compute_edge_latitude(edges))
        @test my_approx(edges.normal, VoronoiMeshes.compute_edge_normal(edges))

        if mesh === mesh_spherical
            inds = abs.(edges.latitude) .<= 80*pi/180
            @test norm(cos.(edges.angle) .- cos.(VoronoiMeshes.compute_edge_angle(edges))) / edges.n <= 1e-4
        else
            @test my_approx(cos.(edges.angle), cos.(VoronoiMeshes.compute_edge_angle(edges)), atol = 1e-2, rtol = 1e-4)
        end

        if mesh === mesh_spherical
            @test my_approx(normalize.(edges.position), edges.normal .× edges.tangent)
        else
            @test all(𝐤 .≈ edges.normal .× edges.tangent)
        end
    end
end

@testset "VoronoiMesh creation" begin
    mesh_dist = VoronoiMesh("mesh.nc")
    m_dist = VoronoiMesh(mesh_dist.cells.info.diagram)
    @test isnothing(check_mesh(m_dist))
    @test isnothing(check_edge_normal_and_tangent(m_dist))

    @test iszero(mapreduce(sum, +, mesh_dist.cells.edgesSign))
    @test iszero(mapreduce(sum, +, mesh_dist.vertices.edgesSign))

    xp = m_dist.x_period
    yp = m_dist.y_period
    edges = m_dist.edges
    @test my_approx_periodic(
        edges.position,
        VoronoiMeshes.compute_edge_midpoint_periodic!(
            similar(edges.position),
            m_dist.cells.position, edges.cells, xp, yp
        ),
        xp, yp
    )

    mesh_spherical = VoronoiMesh("spherical_grid_500km.nc")
    m_spherical = VoronoiMesh(mesh_spherical.cells.info.diagram)
    @test isnothing(check_mesh(m_spherical))
    @test isnothing(check_edge_normal_and_tangent(m_spherical))

    @test iszero(mapreduce(sum, +, mesh_spherical.cells.edgesSign))
    @test iszero(mapreduce(sum, +, mesh_spherical.vertices.edgesSign))

    R = m_spherical.sphere_radius
    cs = m_spherical.cells
    es = m_spherical.edges

    @test my_approx(
        es.position,
        VoronoiMeshes.compute_edge_midpoint_spherical!(similar(es.position), cs.position, es.cells, R)
    )
end

function test_cell_spherical_mimetic_area(areaMimetic::AbstractVector, mesh)
    le = VoronoiMeshes.compute_edge_length(mesh.edges)
    dc = VoronoiMeshes.compute_edge_lengthDual(mesh.edges)
    edgesOnCell = mesh.cells.edges
    areatest = map(edges -> sum(le[edges] .* dc[edges]) / 4, edgesOnCell)
    return my_approx(areaMimetic, areatest)
end


@testset "Mimetic Areas" begin
    mesh_spherical = VoronoiMesh("spherical_grid_500km.nc")

    @test test_cell_spherical_mimetic_area(mesh_spherical.cells.areaMimetic, mesh_spherical)

    @test sum(mesh_spherical.cells.areaMimetic) ≈ sum(mesh_spherical.vertices.areaMimetic)

    @test my_approx(mesh_spherical.vertices.areaMimetic, map(sum, mesh_spherical.vertices.kiteAreasMimetic))

    mesh_dist = VoronoiMesh("mesh_distorted.nc")

    @test my_approx(mesh_dist.cells.area, mesh_dist.cells.areaMimetic)
    @test my_approx(mesh_dist.vertices.area, mesh_dist.vertices.areaMimetic)
    @test my_approx(mesh_dist.vertices.kiteAreas, mesh_dist.vertices.kiteAreasMimetic)

end

@testset "Mesh checks" begin
    @test !isnothing(check_mesh(VoronoiMesh("x1.4002.grid.nc")))
    @test !isnothing(check_mesh(VoronoiMesh("mesh_issues.nc")))
    @test isnothing(check_mesh(VoronoiMesh(fix_diagram!(VoronoiDiagram("x1.4002.grid.nc")))))
    @test isnothing(check_mesh(VoronoiMesh(fix_diagram!(VoronoiDiagram("mesh_issues.nc")))))
end

@testset "DelaunayTriangulation.jl VoronoiMesh creation" begin
    mesh_dist = VoronoiMesh("mesh_distorted.nc")
    mesh_iso = VoronoiMesh("mesh.nc")
    mesh = VoronoiMesh(mesh_dist.cells.position, mesh_dist.x_period, mesh_dist.y_period, rtol=1e-6)
    @test my_approx(mesh.cells.area, mesh_iso.cells.area, rtol=1e-6)
    @test my_approx(mesh.vertices.area, mesh_iso.vertices.area, rtol=1e-4)
    @test my_approx(mesh.edges.length, mesh_iso.edges.length, rtol=1e-4)
    @test my_approx(mesh.edges.lengthDual, mesh_iso.edges.lengthDual, rtol=1e-4)
end

@testset "Save to NetCDF" begin
    mesh_dist = VoronoiMesh("mesh_distorted.nc")
    mesh_spherical = VoronoiMesh("spherical_grid_500km.nc")
    for mesh in (mesh_dist, mesh_spherical)
        for p in Base.propertynames(mesh.cells)
            getproperty(mesh.cells, p);
        end
        for p in Base.propertynames(mesh.vertices)
            getproperty(mesh.vertices, p);
        end
        for p in Base.propertynames(mesh.edges)
            getproperty(mesh.edges, p);
        end

        save("test_save.nc", mesh; force3D=true, write_computed=true)
        test_mesh = VoronoiMesh("test_save.nc");

        for p in Base.propertynames(mesh.cells)
            @test getproperty(mesh.cells, p) == getproperty(test_mesh.cells, p)
        end
        for p in Base.propertynames(mesh.vertices)
            @test getproperty(mesh.vertices, p) == getproperty(test_mesh.vertices, p)
        end
        for p in Base.propertynames(mesh.edges)
            @test getproperty(mesh.edges, p) == getproperty(test_mesh.edges, p)
        end
 
        Base.Filesystem.rm("test_save.nc")
        Base.Filesystem.rm("test_save.graph.info")
    end

    normalEdgesOnCelldata = map(e->fixedvector(mesh_spherical.edges.normal[e]), mesh_spherical.cells.edges)
    normalEdgesOnCell = SmallVectorArray(normalEdgesOnCelldata, mesh_spherical.cells.nEdges)

    write_field_to_netcdf("test_field.nc", normalEdgesOnCell, "normalEdgesOnCell", ("R3","maxEdges","nCells"), ["unit" => "-", "long_name" => "This is a long description"])

    normalEdgesOnCellArray = NCDataset("test_field.nc") do ds
        ds["normalEdgesOnCell"][:,:,:]::Array{Float64,3}
    end

    Base.Filesystem.rm("test_field.nc")

    @test all(normalEdgesOnCellArray[:, j, k] == normalEdgesOnCell.data[k][j] for k in axes(normalEdgesOnCellArray, 3), j in axes(normalEdgesOnCellArray, 2))

end

@testset "Save and read from VTU" begin
    mesh = VoronoiMesh(VoronoiDiagram("mesh_distorted.nc"))

    save("test_save.vtu", mesh)

    test_mesh = VoronoiMesh("test_save.vtu");

    for p in Base.propertynames(mesh.cells)
        @test getproperty(mesh.cells, p) == getproperty(test_mesh.cells, p)
    end
    for p in Base.propertynames(mesh.vertices)
        @test getproperty(mesh.vertices, p) == getproperty(test_mesh.vertices, p)
    end
    for p in Base.propertynames(mesh.edges)
        @test getproperty(mesh.edges, p) == getproperty(test_mesh.edges, p)
    end

    Base.Filesystem.rm("test_save_tri.vtu")
    Base.Filesystem.rm("test_save_vor.vtu")
end
