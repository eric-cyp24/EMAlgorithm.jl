#!/usr/bin/env -S julia

using StatsPlots, Distributions

function generatemodels(N;murange=(-4,8),sigmarange=(0.5,1.3))
    (ml,mh), (sl,sh) = murange, sigmarange
    ml,mh = ml*2, mh*2
    sr = (sh-sl)
    return [Normal(rand(ml:mh)/2, sl+sr*rand(Float64)) for i in 1:N]
end

function generatedata(gaussianmodels; plot=false)
    data = []
    groupsize = []
    for gm in gaussianmodels
        num_data = 1000*rand(1:10)
        data = vcat(data, rand(gm, num_data))
        append!(groupsize,num_data)
    end
    if plot
        plotdatahistogram(data; block=false)
        ratio = groupsize ./ sum(groupsize)
        for (i,(p,gm)) in enumerate(zip(ratio,gaussianmodels))
            println("N$i = ($p, $gm)")
            plot!(gm)
        end
        gui()
        print("\rpress enter to close plot            ")
        readline()
    end
    return data
end

function plotdatahistogram(X; block=true)
    plot(;size=(800,600))
    histogram!(X, label="data", bins=50, normalize=true, color=:grey)
    density!(X, label="Kernel Density Estimation", color=1, linewidth=1.5)
    gui()
    if block
        print("\rpress enter to close plot           ")
        readline()
    end
end

function plotgmmodels(gmms, X; show=true, block=true)
    p = plot(;size=(800,600))
    density!(X, label="Kernel Density Estimation", color=1, linewidth=1.5)
    for (i,(p,gm)) in enumerate(gmms)
        x = range(gm.μ-gm.σ*4, gm.μ+gm.σ*4, 2000)
        y = p*pdf.(gm, x)
        #plot!(gm)
        plot!(x, y, label="gmm $i")
    end
    if show
        gui()
        if block
            print("\rpress enter to close plot           ")
            readline()
        end
    end
    return p
end

function estep(data, gmms)
    gammas = Matrix(Float64, size(X)[end], size(gmms)[end])
    estep!(gammas, data, gmms)
    return gammas
end

function estep!(gammas, data, gmms)
    for (gamma,(p,gm)) in zip(eachcol(gammas),gmms)
        gamma .= p*pdf.(gm, data)
    end
    gammas ./= sum(gammas; dims=2)
end

function mstep!(gmms, data, gammas)
    for (i,gamma) in zip(eachindex(gmms),eachcol(gammas))
        p, gm = gmms[i]
        gamma_sum = sum(gamma)
        μ = sum(gamma.*data) / gamma_sum
        σ = sqrt(sum(gamma.*(data.-gm.μ).^2) / gamma_sum)
        p = gamma_sum / length(gamma)
        gmms[i] = (p,Normal(μ,σ))
    end
end

function loglikelihood(gammas, data, gmms)
    likelihood = zeros(size(gammas)[1])
    for (gamma,(p,gm)) in zip(eachcol(gammas),gmms)
        likelihood += p*pdf.(gm, data)
    end
    loglikelihood = sum(log2.(likelihood))
    return loglikelihood
end

function emalgorithm!(gmms, data, num_epoch=1000; plot=true, tracelikelihood=true)
    loglikelihoods = Vector{Float64}(undef, num_epoch)
    gammas = Matrix{Float64}(undef, size(data)[end], size(gmms)[end])
    for epoch in 1:num_epoch
        print("epoch: $epoch          \r")
        estep!(gammas, data, gmms) # Compute the responsibility: gammas = Pr(Gmms|data)
        mstep!(gmms, data, gammas)

        # trace em-alg
        if tracelikelihood
            loglikelihoods[epoch] = loglikelihood(gammas, data, gmms)
        end
        if plot && (epoch < 20 || epoch % 20 == 19)
            plotgmmodels(gmms, data; block=false)
        end
    end
    if tracelikelihood
        return loglikelihoods
    end
end


function main()

    N = 2
    num_epoch = 1000
    # Generate Data
    models = generatemodels(N)
    X = generatedata(models; plot=true)
    # plot data histogram

    # Initialize Models
    gmms = [(1/N,gm) for gm in generatemodels(N)]
    # plot gm models
    plotgmmodels(gmms, X)


    # EM Algorithm
    likelihoods = emalgorithm!(gmms, X, 4000)

    
    # Show Result
    println("EM Alg result;")
    for (i,(p,gm)) in enumerate(gmms)
        println("N$i ~($p, $gm)")
    end
    p1 = plotgmmodels(gmms, X; show=false)
    p2 = plot(likelihoods)
    plot(p1,p2,size=(1600,600))


end

if !isinteractive()
    main()
end


