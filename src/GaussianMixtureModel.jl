
using Distributions

mutable struct GaussianMixtureModel
    components::Vector{MvNormal}
    weights::Vector{Float64}
    # how to check the given init values
    #function GaussianMixtureModel(components::Vector{MvNormal})
    #    w = ones(length(components))/length(components)
    #    new(components, w)
    #end
end

Base.copy(gmm::GaussianMixtureModel)   = GaussianMixtureModel(copy(gmm.components),copy(gmm.weights))

# Number of mixture components
Base.length(gmm::GaussianMixtureModel) = length(gmm.components)

Base.size(gmm::GaussianMixtureModel)   = size(gmm.components)

Base.ndims(gmm::GaussianMixtureModel)  = length(gmm.components[1])

function Base.:+(gmm::GaussianMixtureModel, v::Vector)
    components = [gm+v for gm in gmm.components]
    return GaussianMixtureModel(components,copy(gmm.weights))
end

function Base.:-(gmm::GaussianMixtureModel, v::Vector)
    components = [gm-v for gm in gmm.components]
    return GaussianMixtureModel(components,copy(gmm.weights))
end

