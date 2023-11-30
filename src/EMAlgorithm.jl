module EMAlgorithm

using Distributions, StatsPlots, HDF5, LinearAlgebra

include("GaussianMixtureModel.jl")  # mutable struct GaussianMixtureModel
include("data_io.jl")               # loaddata, loadtemplate, generatemodels, generatedata,
include("utils.jl")                 # plotGMM!, plotdatascatter!, plotMvNormal!
include("emalg.jl")                 # emalgorithm!, emalgorithm_fixedweight!

export GaussianMixtureModel, 
       generatemodels, generatedata, loaddata, loadmodels, 
       plotMvNormal!, plotGMM!, plotdatascatter!, plotEM!,
       emalgorithm!, emalgorithm_anime!, emalgorithm_fixedweight!

end # module EMAlgorithm
