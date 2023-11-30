
# this function should be move to some place else
function loadtemplate(filename)
    f = h5open(filename, "r")
    labels     = read(f["Template"]["label"])
    mus        = read(f["Template"]["mu"])
    sigmas     = read(f["Template"]["sigma"])
    gmm        = loadmodels(mus, sigmas)
    close(f)
    return gmm, labels
end


function testadjtemplate(filename; axes=[1,2])
    Traces, Ans = loaddata(filename)
    gmm,_       = loadtemplate(filename)
    if length(gmm) == 5
        gmm.weights[:] = [1,4,6,4,1]/16
    end

    plot(;size=(1200,800))
    plotdatascatter!(Traces; axes, show=false)
    plotGMM!(gmm; axes, linestyle=:dash)


    tr_avg = mean(Traces,dims=2)
    h5open(filename,"r") do f
        mu_avg = read(f["Template"]["mean"])
        gmm += vec(tr_avg - mu_avg)
    end
    plotGMM!(gmm; axes)
    print("\rpress enter to close plot          ");readline()


    #llh = emalgorithm!(gmm,Traces,1000; anime=true, axes)
    llh = emalgorithm_fixedweight!(gmm,Traces,1000)

    # plot result
    plot(;size=(1200,900))
    glist = [[] for _ in 1:length(Set(Ans))]
    n_min = min(Set(Ans)...)
    for (i,n) in enumerate(Ans)
        push!(glist[n+1-n_min], i)
    end
    for (n,gl) in enumerate(glist)
        plotdatascatter!(Traces[:,gl]; axes, color=n, show=false)
    end
    plotGMM!(gmm; axes)

end
