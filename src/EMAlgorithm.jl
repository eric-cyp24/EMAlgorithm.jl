module EMAlgorithm

using Distributions, StatsPlots, HDF5, LinearAlgebra

include("GaussianMixtureModel.jl")  # mutable struct GaussianMixtureModel
include("data_io.jl")               # loaddata, loadtemplate, generatemodels, generatedata,
include("utils.jl")                 # plotGMM!, plotdatascatter!, plotMvNormal!
include("emalg.jl")                 # emalgorithm!, emalgorithm_fixedweight!

export GaussianMixtureModel, 
       loaddata, loadtemplate, generatemodels, generatedata,
       plotMvNormal!, plotGMM!, plotdatascatter!,
       estep!, mstep!, emalgorithm!, emalgorithm_fixedweight!


end # module EMAlgorithm
