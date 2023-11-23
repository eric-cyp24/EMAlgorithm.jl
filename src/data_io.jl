
using HDF5

### load data ########

function loaddata(filename)
    f = h5open(filename, "r")
    Traces = read(f["Data"]["Trace"])
    Ans    = read(f["Data"]["Ans"])
    close(f)
    return Traces, Ans
end

function loadtemplate(filename)
    f = h5open(filename, "r")
    labels     = read(f["Template"]["label"])
    mus        = read(f["Template"]["mu"])
    sigmas     = read(f["Template"]["sigma"])
    components = [MvNormal(m,s) for (m,s) in zip(eachcol(mus),eachslice(sigmas;dims=3))]
    prior      = ones(length(components))./length(components)
    gmm        = GaussianMixtureModel(components, prior)
    close(f)
    return gmm, labels
end
##########################


### generate data ########

function generatemodels(N; dims=2, equalprior=false)
    mus,prior  = rand(-10:0.5:10,dims,N),rand(1:10,N)
    sigs       = rand(0.7:0.1:2,dims,2*dims,N).*rand(-1:2:1,dims,2*dims,N)
    sigs       = [sigs[:,:,i]*transpose(sigs[:,:,i]) for i in 1:N]
    prior      = equalprior ? ones(N)./N : prior./sum(prior)
    components = [MvNormal(m,s) for (m,s) in zip(eachcol(mus),sigs)]
    return GaussianMixtureModel(components, prior)
end

function generatedata(gmm, datasize=5000)
    data = Matrix{Float64}(undef,ndims(gmm),0)
    datasize = datasize<1000 ? 1000*length(gmm) : datasize
    for (p,gm) in zip(gmm.weights, gmm.components)
        data = hcat(data, rand(gm, Int(round(p*datasize))))
    end
    return data
end
##########################
