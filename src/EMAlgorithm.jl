module EMAlgorithm

using Distributed, StatsPlots, HDF5, LinearAlgebra, Distributions, SharedArrays

include("GaussianMixtureModel.jl")  # mutable struct GaussianMixtureModel
include("data_io.jl")               # loaddata, loadtemplate, generatemodels, generatedata,
include("utils.jl")                 # plotGMM!, plotdatascatter!, plotMvNormal!
include("emalg.jl")                 # emalgorithm!, emalgorithm_fixedweight!

export GaussianMixtureModel, 
       generatemodels, generatedata, loaddata, loadmodels, 
       plotMvNormal!, plotGMM!, plotdatascatter!, plotEM!,
       emalgorithm!, emalgorithm_anime!, emalgorithm_fixedweight!, emalgorithm_fixedweight_mprocess!

end # module EMAlgorithm
