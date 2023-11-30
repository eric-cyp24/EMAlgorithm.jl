using StatsPlots, HDF5, Statistics #, Distribution, LinearAlgebra
using Distributions: MvNormal
using EMAlgorithm

mutable struct Template
    # Raw Trace space
    TraceMean::Vector{Float64}
    TraceVar::Vector{Float64}
    ProjMatrix::Matrix{Float64} # Raw-to-LDA_Subspace
    # LDA Subspace
    mean::Vector{Float64}
    covMatrix::Matrix{Float64}
    # MVG models
    labels::Vector{Integer}
    mus::Vector{Vector{Float64}}
    sigmas::Vector{Matrix{Float64}}
    probs::Vector{Float64}
    pooled_cov_inv::Matrix{Float64} # for Pooled Covariance Matrix

    function Template(TraceMean, TraceVar, ProjMatrix, mean, covMatrix, labels,
                      mus::Matrix{T}, sigmas::Array{T,3}, probs, pooled_cov_inv) where T <: AbstractFloat
        mus    = eachcol(mus)
        sigmas = eachslice(sigmas, dims=3)
        new(TraceMean, TraceVar, ProjMatrix, mean, covMatrix,
            labels, mus, sigmas, probs, pooled_cov_inv)
    end
end

#templatefields = String.(fieldnames(Template))
#               = ("TraceMean", "TraceVar", "ProjMatrix", "mean", "covMatrix",
#                  "labels", "mus", "sigmas", "probs", "pooled_cov_inv")

load_dataset(h5group, h5dataset) = h5dataset in keys(h5group) ? read_dataset(h5group, h5dataset) : nothing
function loadtemplate(filename, byte=0)
    t = h5open(filename, "r") do h5
        g = open_group(h5, "Templates/byte $byte")
        return [load_dataset(g, String(n)) for n in fieldnames(Template)]
    end
    return Template(t...)
end

function writetemplate(filename, template; byte=0)
    group_path="Templates/byte $byte"
    h5open(filename, "cw") do h5
        g = open_group(h5, group_path)
        for n in fieldnames(Template)
            if n == :mus || n == :sigmas
                write_dataset(g,String(n),stack(getproperty(template,n)))
            else
                write_dataset(g,String(n),getproperty(template,n))
            end
        end
    end
end

function Template2GMM(t::Template)
    components = [MvNormal(μ,Σ) for (μ,Σ) in zip(t.mus, t.sigmas)]
    weights    = t.probs
    return GaussianMixtureModel(components, weights)
end

function adjtemplate!(template, tr_lda, num_epoch=500; δ=10e-9)
    # gmm re-center
    gmm = Template2GMM(template) + vec(mean(tr_lda,dims=2) - template.mean)
    llh = emalgorithm_fixedweight!(gmm, tr_lda, num_epoch; δ)
    for i in 1:length(gmm)
        template.mus[i]    = gmm.components[i].μ
        template.sigmas[i] = gmm.components[i].Σ
    end
    return gmm
end


function testadjtemplate(filename; axes=[1,2])
    Traces, Ans = loaddata(filename)
    template    = loadtemplate(filename)
    gmm         = Template2GMM(template)

    plot(;size=(1200,800))
    plotEM!(Traces, gmm; axes, show=false, linestyle=:dash)
    sleep(3)


    adjtemplate!(template, Traces)

    # plot result
    plot(;size=(1200,800))
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
