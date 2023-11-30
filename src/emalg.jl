
function update!(gmm::GaussianMixtureModel, index, component, prob)
    gmm.components[index] = component
    gmm.weights[index]    = prob
end

function estep!(gammas, data, gmm)
    for (gamma,p,gm) in zip(eachcol(gammas),gmm.weights,gmm.components)
        gamma .= p*pdf(gm, data)
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

function mstepfixprior!(gmm, data, gammas)
    for (i,gamma) in enumerate(eachcol(gammas))
        # calculate new parameters
        gamma_sum = sum(gamma)
        μ = data*gamma ./ gamma_sum
        x = (data.-μ).*transpose(sqrt.(gamma))
        Σ = (x*transpose(x)) ./ gamma_sum
        p = gmm.weights[i]    # do not update prior
        
        update!(gmm, i, MvNormal(μ,Σ), p)
    end
end

"""
    emalgorithm!(gmm, data, num_epoch=1000; δ::AbstractFloat)

given an initial GaussianMixtureModel `gmm` and a set of `data`, use EM algorithm to
modify the `gmm` and find the best fitted GaussianMixtureModel.
"""
function emalgorithm!(gmm, data, num_epoch::Integer=1000; δ::AbstractFloat=10e-7)
    gammas = Matrix{Float64}(undef, size(data)[end], length(gmm))
    likelihoods = Vector{Float64}(undef,0)
    for epoch in 1:num_epoch
        llh = estep!(gammas, data, gmm)
        mstep!(gmm, data, gammas)
        push!(likelihoods, llh)
        
        # converge & early abort
        if (length(likelihoods)>10) && 
           (llh-likelihoods[end-1])/(llh-likelihoods[2]) < δ
            break
        end
    end
end

"""
    emalgorithm_fixedweight!(gmm, data, num_epoch=1000; δ::AbstractFloat=10e-7)

return the best fitted GaussianMixtureModel with a fixed a priori, 
only modify the (μ, Σ) of the `gmm` components.
"""
function emalgorithm_fixedweight!(gmm, data, num_epoch::Integer=1000; δ::AbstractFloat=10e-7)
    gammas = Matrix{Float64}(undef, size(data)[end], length(gmm))
    likelihoods = Vector{Float64}(undef,0)
    for epoch in 1:num_epoch
        print("epoch: $epoch          \r")
        
        llh = estep!(gammas, data, gmm)
        mstepfixprior!(gmm, data, gammas)
        push!(likelihoods, llh)
        
        # converge & early abort
        if (length(likelihoods)>10) && 
           (llh-likelihoods[end-1])/(llh-likelihoods[2]) < δ
            break
        end
    end
end

"""
    emalgorithm_anime!(gmm, data, num_epoch::Integer=1000; δ::AbstractFloat=10e-7, axes=[1,2])

run the emalgorithm! while plotting the intermediate results.
"""
function emalgorithm_anime!(gmm, data, num_epoch::Integer=1000; δ::AbstractFloat=10e-7, axes=[1,2])
    gammas = Matrix{Float64}(undef, size(data)[end], length(gmm))
    gmm_init, prior = copy(gmm), copy(gmm.weights)
    likelihoods = Vector{Float64}(undef,0)
    for epoch in 1:num_epoch
        # estep
        llh = estep!(gammas, data, gmm)
        prior .= vec(sum(gammas,dims=1)./size(gammas)[1])
        
        # mstep
        mstep!(gmm, data, gammas)
        push!(likelihoods, llh)

        # plot
        print("epoch: $epoch          \r")
        if ((epoch < 40) || (epoch % 40 == 39))
            gmm_show = GaussianMixtureModel(gmm.components, prior)
            plot(;size=(800,600)); plotEM!(data, gmm_show; axes)
        end
        if (length(likelihoods)>10) && 
           (llh-likelihoods[end-1])/(llh-likelihoods[2]) < δ
            break
        end
    end
    gmm_final = GaussianMixtureModel(gmm.components, prior)
    plot(;size=(800,600)); plotEM!(data, gmm_final; axes)
    plotGMM!(gmm_init; axes, label="", linestyle=:dash)
    return likelihoods
end


