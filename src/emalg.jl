
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
function emalgorithm!(gmm, data, num_epoch=1000)
    gammas = Matrix{Float64}(undef, size(data)[end], length(gmm))
    likelihoods = Vector{Float64}(undef,0)
    for epoch in 1:num_epoch
        push!(likelihoods, estep!(gammas, data, gmm))
        mstep!(gmm, data, gammas)
    end
end
"""

function emalgorithm_fixedweight!(gmm, data, num_epoch=1000; δ=10e-3)
    gammas = Matrix{Float64}(undef, size(data)[end], length(gmm))
    likelihoods = Vector{Float64}(undef,0)
    for epoch in 1:num_epoch
        print("epoch: $epoch          \r")
        
        push!(likelihoods, estep!(gammas, data, gmm))
        mstepfixprior!(gmm, data, gammas)
        
        # converge & early abort
        if (length(likelihoods)>10) && sum(likelihoods[end].-likelihoods[end-10:end-1]) < δ
            break
        end
    end
    #gmm.weights .= vec(sum(gammas,dims=1)./size(gammas)[1])
end

function emalgorithm_anime!(gmm, data, num_epoch=1000; δ=10e-6, axes=[1,2])
    gammas = Matrix{Float64}(undef, size(data)[end], length(gmm))
    gmm_init, prior = copy(gmm), copy(gmm.weights)
    likelihoods = Vector{Float64}(undef,0)
    for epoch in 1:num_epoch
        # estep
        push!(likelihoods, estep!(gammas, data, gmm))
        prior .= vec(sum(gammas,dims=1)./size(gammas)[1])
        
        # mstep
        mstep!(gmm, data, gammas)

        # plot
        print("epoch: $epoch          \r")
        if ((epoch < 40) || (epoch % 40 == 39))
            plot(;size=(800,600))
            plotdatascatter!(data; axes, show=false)
            gmm_show = GaussianMixtureModel(gmm.components, prior)
            plotGMM!(gmm_show; axes)
        end
        if (length(likelihoods)>10) && sum(likelihoods[end].-likelihoods[end-10:end-1]) < δ
            break
        end
    end
    plot(;size=(800,600))
    plotdatascatter!(data; axes, show=false)
    gmm.weights .= prior
    plotGMM!(gmm; axes)
    plotGMM!(gmm_init; axes, label="", linestyle=:dash)
    return likelihoods
end


