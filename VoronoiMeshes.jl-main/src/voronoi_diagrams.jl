abstract type AbstractVoronoiDiagram{S, maxEdges, TI <: Integer, TF <: Real, Tz} end

struct PlanarVoronoiDiagram{maxEdges, TI, TF} <: AbstractVoronoiDiagram{false, maxEdges, TI, TF, Zeros.Zero}
    generators::Vec2DxyArray{TF, 1}
    vertices::Vec2DxyArray{TF, 1}
    verticesOnCell::SmVecArray{maxEdges, TI, 1}
    cellsOnVertex::Vector{FixedVector{3, TI}}
    meshDensity::Vector{TF}
    x_period::TF
    y_period::TF
end

struct SphericalVoronoiDiagram{maxEdges, TI, TF} <: AbstractVoronoiDiagram{true, maxEdges, TI, TF, TF}
    generators::Vec3DArray{TF, 1}
    vertices::Vec3DArray{TF, 1}
    verticesOnCell::SmVecArray{maxEdges, TI, 1}
    cellsOnVertex::Vector{FixedVector{3, TI}}
    meshDensity::Vector{TF}
    sphere_radius::TF
end

struct VoronoiDiagram{S, maxEdges, TI, TF, Tz} <: AbstractVoronoiDiagram{S, maxEdges, TI, TF, Tz}
    data::Union{PlanarVoronoiDiagram{maxEdges, TI, TF}, SphericalVoronoiDiagram{maxEdges, TI, TF}}

    function VoronoiDiagram(d::PlanarVoronoiDiagram{maxEdges, TI, TF}) where {maxEdges, TI, TF}
        return new{false, maxEdges, TI, TF, Zero}(d)
    end

    function VoronoiDiagram(d::SphericalVoronoiDiagram{maxEdges, TI, TF}) where {maxEdges, TI, TF}
        return new{true, maxEdges, TI, TF, TF}(d)
    end
end

get_diagram(v::VoronoiDiagram{true, maxEdges, TI, TF, TF}) where {maxEdges, TI, TF} = Base.getfield(v, :data)::SphericalVoronoiDiagram{maxEdges, TI, TF}
get_diagram(v::VoronoiDiagram{false, maxEdges, TI, TF, Zero}) where {maxEdges, TI, TF} = Base.getfield(v, :data)::PlanarVoronoiDiagram{maxEdges, TI, TF}

Base.getproperty(v::VoronoiDiagram, s::Symbol) = _getproperty(v, Val{s}())
_getproperty(v::VoronoiDiagram, ::Val{:data}) = get_diagram(v)

for maxEdges in 6:10
    let TF = Float64, TI = Int32
        precompile(PlanarVoronoiDiagram, (Vec2DxyArray{TF, 1}, Vec2DxyArray{TF, 1}, SmVecArray{maxEdges, TI, 1}, Vector{FixedVector{3, TI}}, Vector{TF}, TF, TF))
        precompile(SphericalVoronoiDiagram, (Vec3DArray{TF, 1}, Vec3DArray{TF, 1}, SmVecArray{maxEdges, TI, 1}, Vector{FixedVector{3, TI}}, Vector{TF}, TF))
        precompile(VoronoiDiagram, (PlanarVoronoiDiagram{maxEdges, TI, TF},))
        precompile(VoronoiDiagram, (SphericalVoronoiDiagram{maxEdges, TI, TF},))
        precompile(get_diagram, (VoronoiDiagram{true, maxEdges, TI, TF, TF},))
        precompile(get_diagram, (VoronoiDiagram{false, maxEdges, TI, TF, Zero},))
        precompile(Base.getproperty, (VoronoiDiagram{true, maxEdges, TI, TF, TF}, Symbol))
        precompile(Base.getproperty, (VoronoiDiagram{false, maxEdges, TI, TF, Zero}, Symbol))
        precompile(_getproperty, (VoronoiDiagram{true, maxEdges, TI, TF, TF}, Val{:data}))
        precompile(_getproperty, (VoronoiDiagram{false, maxEdges, TI, TF, Zero}, Val{:data}))
    end
end

for s in fieldnames(PlanarVoronoiDiagram)
    @eval _getproperty(v::VoronoiDiagram{false}, ::Val{$(QuoteNode(s))}) = getfield(get_diagram(v), $(QuoteNode(s)))

    for maxEdges in 6:10
        let TF = Float64, TI = Int32
            @eval precompile(_getproperty, (VoronoiDiagram{false, $maxEdges, $TI, $TF, Zero}, Val{$(QuoteNode(s))}))
        end
    end
end

const pvd_property_names = (fieldnames(VoronoiDiagram)..., fieldnames(PlanarVoronoiDiagram)...)
Base.propertynames(::VoronoiDiagram{false}) = pvd_property_names

for s in fieldnames(SphericalVoronoiDiagram)
    @eval _getproperty(v::VoronoiDiagram{true}, ::Val{$(QuoteNode(s))}) = getfield(get_diagram(v), $(QuoteNode(s)))

    for maxEdges in 6:10
        let TF = Float64, TI = Int32
            @eval precompile(_getproperty, (VoronoiDiagram{true, $maxEdges, $TI, $TF, $TF}, Val{$(QuoteNode(s))}))
        end
    end
end

const svd_property_names = (fieldnames(VoronoiDiagram)..., fieldnames(SphericalVoronoiDiagram)...)
Base.propertynames(::VoronoiDiagram{true}) = svd_property_names

