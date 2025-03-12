
## EM Algorithm internal steps ##

function update!(gmm::GaussianMixtureModel, index, component, prob)
    gmm.components[index] = component
    gmm.weights[index]    = prob
end

function update!(gmm::GaussianMixtureModel, index, component)
    gmm.components[index] = component
end

function estep!(gammas, data, gmm)
    for i in 1:length(gmm)
        gammas[:,i] .= gmm.weights[i]*pdf(gmm.components[i], data)
    end
    likelihood = sum(log2.(sum(gammas,dims=2)))
    gammas ./= sum(gammas; dims=2)
    return likelihood
end

function mstep!(gmm, data, gammas)
    for (i,gamma) in enumerate(eachcol(gammas))
        # calculate new parameters
        gamma_sum = sum(gamma)
        μ = data*gamma ./ gamma_sum
        x = (data.-μ).*transpose(sqrt.(gamma)) # try LinearAlgebra.BLAS.syr!
        Σ = (x*transpose(x)) ./ gamma_sum
        p = gamma_sum / length(gamma)

        update!(gmm, i, MvNormal(μ,Σ), p)
    end
end

function mstepfixweight!(gmm, data, gammas)
    for (i,gamma) in enumerate(eachcol(gammas))
        # calculate new parameters
        gamma_sum = sum(gamma)
        μ = data*gamma ./ gamma_sum
        x = (data.-μ).*transpose(sqrt.(gamma))
        Σ = (x*transpose(x)) ./ gamma_sum
        p = gmm.weights[i]    # do not update weights

        update!(gmm, i, MvNormal(μ,Σ))
    end
end

## helper functions

"""
Cauculate the responsibility of each gmm component from the given data.
This is the same as the p update in mstep!.
This is used to evaluate the weight of each component for mstepfixweight!.
"""
function datadistribution(gmm::GaussianMixtureModel, data)
    gammas = Matrix{Float64}(undef, size(data,2), length(gmm))
    for i in 1:length(gmm)
        gammas[:,i] .= gmm.weights[i]*pdf(gmm.components[i], data)
    end
    gammas ./= sum(gammas; dims=2)
    return vec(sum(gammas;dims=1) / size(gammas,1))
end

## Different Types of EM Algorithm ##

"""
    emalgorithm!(gmm, data, num_epoch=1000; δ::AbstractFloat)

given an initial GaussianMixtureModel `gmm` and a set of `data`, use EM algorithm to
modify the `gmm` and find the best fitted GaussianMixtureModel.
"""
function emalgorithm!(gmm, data, num_epoch::Integer=1000; δ::AbstractFloat=10e-7)
    gammas = Matrix{Float64}(undef, size(data,2), length(gmm))
    likelihoods = Vector{Float64}(undef,0)
    for epoch in 1:num_epoch
        llh = estep!(gammas, data, gmm)
        mstep!(gmm, data, gammas)
        push!(likelihoods, llh)

        # converge & early abort
        if (length(likelihoods)>5) &&
           (llh-likelihoods[end-1])/(llh-likelihoods[2]) < δ
            break
        end
    end
    return likelihoods[end]
end

"""
    emalgorithm_anime!(gmm, data, num_epoch::Integer=1000; δ::AbstractFloat=10e-7, axes=[1,2])

run the emalgorithm! while plotting the intermediate results.
"""
function emalgorithm_anime!(gmm, data, num_epoch::Integer=1000; δ::AbstractFloat=10e-7, axes=[1,2], kwargs...)
    gammas = Matrix{Float64}(undef, size(data,2), length(gmm))
    gmm_init, weight = copy(gmm), copy(gmm.weights)
    likelihoods = Vector{Float64}(undef,0)
    for epoch in 1:num_epoch
        # estep
        llh = estep!(gammas, data, gmm)
        weight .= vec(sum(gammas,dims=1)./size(gammas)[1])

        # mstep
        mstep!(gmm, data, gammas)
        push!(likelihoods, llh)

        # plot
        print("epoch: $epoch          \r")
        if ((epoch < 40) || (epoch % 40 == 39))
            gmm_show = GaussianMixtureModel(gmm.components, weight)
            plotEM(data, gmm_show; axes, kwargs...)
        end
        if (length(likelihoods)>5) &&
           (llh-likelihoods[end-1])/(llh-likelihoods[2]) < δ
            break
        end
    end
    gmm_final = GaussianMixtureModel(gmm.components, weight)
    plotEM(data, gmm_final; axes, kwargs...)
    plotGMM!(gmm_init; axes, label="", linestyle=:dash, kwargs...)
    return likelihoods
end

"""
    emalgorithm_fixedweight!(gmm, data, num_epoch=1000; δ::AbstractFloat=10e-7)

return the best fitted GaussianMixtureModel with a fixed mixture weights,
only modify the (μ, Σ) of the `gmm` components.
"""
function emalgorithm_fixedweight!(gmm, data, num_epoch::Integer=1000; δ::AbstractFloat=10e-7)
    gammas = Matrix{Float64}(undef, size(data)[end], length(gmm))
    likelihoods = Vector{Float64}(undef,0)
    for epoch in 1:num_epoch
        print("epoch: $epoch          \r")

        llh = estep!(gammas, data, gmm)
        mstepfixweight!(gmm, data, gammas)
        push!(likelihoods, llh)

        # converge & early abort
        if (length(likelihoods)>5) &&
           (llh-likelihoods[end-1])/(llh-likelihoods[2]) < δ
            break
        end
    end
    return likelihoods[end]
end

function emalgorithm_fixedweight_anime!(gmm, data, num_epoch::Integer=1000; δ::AbstractFloat=10e-7, axes=[1,2], kwargs...)
    gammas = Matrix{Float64}(undef, size(data)[end], length(gmm))
    gmm_init, weight = copy(gmm), copy(gmm.weights)
    likelihoods = Vector{Float64}(undef,0)
    for epoch in 1:num_epoch
        print("epoch: $epoch          \r")

        llh = estep!(gammas, data, gmm)
        weight .= vec(sum(gammas,dims=1)./size(gammas)[1])
        mstepfixweight!(gmm, data, gammas)
        push!(likelihoods, llh)

        # plot
        print("epoch: $epoch          \r")
        if ((epoch < 40) || (epoch % 40 == 39))
            gmm_show = GaussianMixtureModel(gmm.components, weight)
            plotEM(data, gmm_show; axes, kwargs...)
        end
        # converge & early abort
        if (length(likelihoods)>5) &&
           (llh-likelihoods[end-1])/(llh-likelihoods[2]) < δ
            break
        end
    end
    gmm_final = GaussianMixtureModel(gmm.components, weight)
    plotEM(data, gmm_final; axes, kwargs...)
    plotGMM!(gmm_init; axes, label="", linestyle=:dash, kwargs...)
    return likelihoods
end



### multi-process ####
#using Distributed
#using SharedArrays, Distributions
#@everywhere using SharedArrays, Distributions
#check out: https://stackoverflow.com/questions/47685536/how-do-you-load-a-module-everywhere-inside-a-function-in-julia

function estep_mprocess!(gammas::SharedMatrix, data::AbstractMatrix, gmm::GaussianMixtureModel)
    @sync @distributed for i in 1:length(gmm)
        gammas[:,i] .= gmm.weights[i]*pdf(gmm.components[i], data)
    end
    likelihood = sum(log2.(sum(gammas,dims=2)))
    gammas   ./= sum(gammas; dims=2)
    return likelihood
end

function mstepfixweight_mprocess!(gmm, data, gammas; μs::SharedMatrix, Σs::SharedArray)
    @sync @distributed for i in 1:length(gmm)
        # calculate new parameters
        gamma_sum = sum(gammas[:,i])
        μs[:,i]   = data*gammas[:,i] ./ gamma_sum
        x         = (data.-μs[:,i]).*transpose(sqrt.(gammas[:,i]))
        Σs[:,:,i] = (x*transpose(x)) ./ gamma_sum
    end
    for i in 1:length(gmm)
        gmm.components[i] = MvNormal(μs[:,i],Σs[:,:,i])
    end
end

"""
    emalg_addprocs(n::Integer)

add `n` worker process, and run emalg_mprocs_init.jl on all worker process.
"""
function emalg_addprocs(n::Integer=1)
    newworkers = addprocs(n)
    @everywhere include(joinpath(@__DIR__,"emalg_mprocs_init.jl"))
    return newworkers
end

function emalgorithm_fixedweight_mprocess!(gmm, data, num_epoch::Integer=1000; δ::AbstractFloat=10e-7)
    noworker = (nprocs()<2)
    if noworker
        newworkers = addprocs(Sys.CPU_THREADS÷2)
    end
    @everywhere include(joinpath(@__DIR__,"emalg_mprocs_init.jl"))
    gammas = SharedMatrix{Float64}(size(data,2), length(gmm))
    μs     = SharedMatrix{Float64}(ndims(gmm), length(gmm))
    Σs     = SharedArray{Float64}(ndims(gmm), ndims(gmm), length(gmm))
    likelihoods = Vector{Float64}(undef,0)
    for epoch in 1:num_epoch
        print("epoch: $epoch  \r")
        llh = estep_mprocess!(gammas, data, gmm)
        mstepfixweight_mprocess!(gmm, data, gammas; μs, Σs)
        push!(likelihoods, llh)

        # converge & early abort
        if (length(likelihoods)>5) &&
            #(llh-likelihoods[end-1])/(likelihoods[end-1]-likelihoods[end-2]) < (1-δ)
            (llh-likelihoods[end-1])/(llh-likelihoods[2]) < δ
            break
        end
    end
    noworker && rmprocs(newworkers)
    return likelihoods
end

