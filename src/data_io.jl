#using Distributions, HDF5

### load data ########
"""
    loaddata(filename)

load traces and intermediate_values from HDF5 file.
"""
function loaddata(filename)
    f = h5open(filename, "r")
    Traces = read(f["Data"]["Trace"])
    try
        Ans = read(f["Data"]["Ans"])
    catch KeyError
        Ans = nothing
    end
    close(f)
    return Traces, Ans
end

"""
    loadmodels(gaussians, prior=nothing)

generate a GaussianMixtureModel from the given vector of (μ, Σ).
If `prior` is not specified, all components are given equal prior.

# Examples
```julia-repl
julia> loadmodels([([0,1.0],[1 0; 0 1.0]), ([1.0,0],[1 0.5; 0.5 1])])
GaussianMixtureModel(Distributions.MvNormal[FullNormal(
dim: 2
μ: [0.0, 1.0]
Σ: [1.0 0.0; 0.0 1.0]
)
, FullNormal(
dim: 2
μ: [1.0, 0.0]
Σ: [1.0 0.5; 0.5 1.0]
)
], [0.5, 0.5])
```
"""
function loadmodels(gaussians::Vector{Tuple{Vector{T}, Matrix{T}}}, 
                    prior=nothing::Union{Nothing,Vector{T}}) where T<:AbstractFloat
    components = [MvNormal(mu,sig) for (mu,sig) in gaussians]
    prior      = isnothing(prior) ? ones(length(components))./length(components) : prior
    gmm        = GaussianMixtureModel(components, prior)
    return gmm
end

"""
    loadmodels(mus, sigmas, prior=nothing)

generate a GaussianMixtureModel from the given `mus` and `sigmas`.
"""
function loadmodels(mus::Matrix{T}, sigmas::Array{T, 3},
                    prior=nothing::Union{Nothing,Vector{T}}) where T<:AbstractFloat
    components = [MvNormal(mu,sig) for (mu,sig) in zip(eachcol(mus),eachslice(sigmas;dims=3))]
    prior      = isnothing(prior) ? ones(length(components))./length(components) : prior
    gmm        = GaussianMixtureModel(components, prior)
    return gmm
end

### generate data ########
"""
    generatemodels(N; dims, equalprior::Bool)

randomly generate a GaussianMixtureModel.

# Arguments
- `N::Integer`: number of components.
- `dims::Integer`: the dimension of the GaussianMixtureModel.
- `equalprior`: if false, assign random prior to each component.
"""
function generatemodels(N::Integer; dims::Integer=2, equalprior::Bool=false)
    mus,prior  = rand(-10:0.5:10,dims,N),rand(1:10,N)
    sigs       = rand(0.7:0.1:2,dims,2*dims,N).*rand(-1:2:1,dims,2*dims,N)
    sigs       = [sigs[:,:,i]*transpose(sigs[:,:,i]) for i in 1:N]
    prior      = equalprior ? ones(N)./N : prior./sum(prior)
    components = [MvNormal(m,s) for (m,s) in zip(eachcol(mus),sigs)]
    return GaussianMixtureModel(components, prior)
end

"""
    generatedata(gmm::GaussianMixtureModel, datasize::Integer=5000)

generate data from the given GaussianMixtureModel.
"""
function generatedata(gmm::GaussianMixtureModel, datasize::Integer=5000)
    data = Matrix{Float64}(undef,ndims(gmm),0)
    datasize = datasize<1000 ? 1000*length(gmm) : datasize
    for (p,gm) in zip(gmm.weights, gmm.components)
        data = hcat(data, rand(gm, Int(round(p*datasize))))
    end
    return data
end
