
using Distributions

"""
    GaussianMixtureModel(components::Vector{MvNormal}, weights=nothing)

generate `GaussianMixtureModel` without specifying `weights` will result in equal weights.

# Fields
```
components :: Vector{MvNormal}
weights    :: Vector{Float64}
```
"""
mutable struct GaussianMixtureModel
    components::Vector{MvNormal}
    weights::Vector{Float64}

    function GaussianMixtureModel(components::Vector{<:MvNormal}, w::Union{Nothing, Vector{Float64}}=nothing)
        w = isnothing(w) ? ones(length(components))/length(components) : w/sum(w)
        new(components, w)
    end
end


"""
    GaussianMixtureModel(components::Vector{Tuple{Vector{T},Matrix{T}}}, w::Union{Nothing, Vector{T}}=nothing) where T <:Real
    GaussianMixtureModel(mus::Vector{Vector{T}}, sigmas::Vector{Matrix{T}}, w::Union{Nothing, Vector{T}}=nothing) where T <:Real
    GaussianMixtureModel(mus::Matrix{T}, sigmas::Array{T,3}, w::Union{Nothing, Vector{T}}=nothing) where T <:Real

generate `GaussianMixtureModel` with components' `μ`s and `Σ`s.
"""
function GaussianMixtureModel(params::Vector{Tuple{AbstractVector,AbstractMatrix}}, w::Union{Nothing, AbstractVector}=nothing)
    components = [MvNormal(p...) for p in params]
    w = isnothing(w) ? ones(length(components))/length(components) : w/sum(w)
    return GaussianMixtureModel(components, w)
end

function GaussianMixtureModel(mus::Vector{AbstractVector}, sigmas::Vector{AbstractMatrix}, w::Union{Nothing, AbstractVector}=nothing)
    components = [MvNormal(μ,Σ) for (μ,Σ) in zip(mus,sigmas)]
    w = isnothing(w) ? ones(length(components))/length(components) : w/sum(w)
    return GaussianMixtureModel(components, w)
end

function GaussianMixtureModel(mus::AbstractMatrix, sigmas::AbstractArray, w::Union{Nothing, AbstractVector}=nothing)
    components = [MvNormal(μ,Σ) for (μ,Σ) in zip(eachcol(mus),eachslice(sigmas,dims=3))]
    w = isnothing(w) ? ones(length(components))/length(components) : w/sum(w)
    return GaussianMixtureModel(components, w)
end


Base.copy(gmm::GaussianMixtureModel)   = GaussianMixtureModel(copy(gmm.components),copy(gmm.weights))

# Number of mixture components
Base.length(gmm::GaussianMixtureModel) = length(gmm.components)

# (Number of mixture components, MvNormal dimension)
Base.size(gmm::GaussianMixtureModel)   = (length(gmm.components), size(gmm.components[1]))

# MvNormal Dimension
Base.ndims(gmm::GaussianMixtureModel)  = length(gmm.components[1])

function Base.:+(gmm::GaussianMixtureModel, v::Vector)
    components = [gm+v for gm in gmm.components]
    return GaussianMixtureModel(components,copy(gmm.weights))
end

function Base.:-(gmm::GaussianMixtureModel, v::Vector)
    components = [gm-v for gm in gmm.components]
    return GaussianMixtureModel(components,copy(gmm.weights))
end

function Base.:(==)(gmm1::GaussianMixtureModel, gmm2::GaussianMixtureModel)
    return (gmm1.components == gmm2.components) && (gmm1.weights == gmm2.weights)
end
