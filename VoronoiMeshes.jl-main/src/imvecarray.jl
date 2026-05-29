"""
    SmallVectorArray{NT, T, N} <: AbstractArray{SmallVector{NT, T}, N}

An array of `SmallVector`s that follows a SoA layout ("Struct of Arrays") for the `SmallVector`'s `data` and `length` fields
"""
struct SmallVectorArray{N_MAX, T, N, TD <: AbstractArray{FixedVector{N_MAX,T},N}} <: AbstractArray{SmallVector{N_MAX, T}, N}
    data::TD
    length::Array{Int16, N}

    function SmallVectorArray(data::AbstractArray{FixedVector{N_MAX,T},N}, l::Array{Int16, N}) where {N_MAX, T, N}
        size(l) == size(data) || throw(DimensionMismatch())
        return new{N_MAX, T, N, typeof(data)}(data, l)
    end
end

const SmVecArray{N_MAX, T, N} = SmallVectorArray{N_MAX, T, N, Array{FixedVector{N_MAX, T}, N}}

SmallVectorArray(data::AbstractArray{FixedVector{N_MAX,T}, N}, l::AbstractArray{<:Integer,N}) where {N_MAX, T, N} = SmallVectorArray(data, Int16.(l))

SmallVectorArray{N_MAX, T}(s::Vararg{Integer}) where {N_MAX, T} = SmallVectorArray(Array{FixedVector{N_MAX, T}}(undef, s...), zeros(Int16, s...))

SmVecArray{N_MAX, T}(s::Vararg{Integer}) where {N_MAX, T} = SmallVectorArray{N_MAX, T}(s...)

Base.size(IVA::SmallVectorArray) = size(IVA.data)
Base.length(IVA::SmallVectorArray) = length(IVA.data)
Base.IndexStyle(::Type{SmallVectorArray{NN, T, N, TD}}) where {NN, T, N, TD} = IndexStyle(TD)
Base.similar(a::SmallVectorArray, dims=size(a)) = SmallVectorArray(similar(a.data, dims), similar(a.length, dims))

@inline function Base.getindex(a::SmallVectorArray, i::Integer)
    @boundscheck checkbounds(a,i)
    data = a.data
    l = a.length
    @inbounds SmallVector(data[i], l[i])
end

@inline function Base.setindex!(a::SmallVectorArray, v, i::Integer)
    @boundscheck checkbounds(a,i)
    cv = convert(eltype(a), v)
    data = a.data
    l = a.length
    @inbounds data[i] = fixedvector(cv)
    @inbounds l[i] = cv.n
    return a
end

@inline function Base.getindex(a::SmallVectorArray{NN, T, N}, I::Vararg{Integer}) where {NN, T, N}
    @boundscheck checkbounds(a,I...)
    data = a.data
    l = a.length
    @inbounds SmallVector(data[I...], l[I...])
end

@inline function Base.setindex!(a::SmallVectorArray{NN, T, N}, v, I::Vararg{Integer}) where {NN, T, N}
    @boundscheck checkbounds(a,I...)
    cv = convert(eltype(a), v)
    data = a.data
    l = a.length
    @inbounds data[I...] = fixedvector(cv)
    @inbounds l[I...] = cv.n
    return a
end

