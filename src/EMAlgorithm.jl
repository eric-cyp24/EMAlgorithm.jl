module EMAlgorithm

using Distributed, StatsPlots, HDF5, LinearAlgebra, Distributions, SharedArrays

import Statistics: mean, cov

include("GaussianMixtureModel.jl")  # mutable struct GaussianMixtureModel
include("data_io.jl")               # loadtemplate, generatemodels, generatedata,
include("utils.jl")                 # plotGMM!, plotdatascatter!, plotMvNormal!
include("emalg.jl")                 # emalgorithm!, emalgorithm_fixedweight!

export GaussianMixtureModel, mean, cov,
       generatemodels, generatedata, loadmodels,
       plotMvNormal, plotGMM, plotEM, plotMvNormal!, plotGMM!, plotEM!,
       emalgorithm!, emalgorithm_anime!, emalgorithm_fixedweight!, emalgorithm_fixedweight_mprocess!

end # module EMAlgorithm
