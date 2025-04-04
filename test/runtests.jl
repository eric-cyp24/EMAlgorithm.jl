using Test, Random
using EMAlgorithm



@testset verbose=true "EMAlgorithm tests" begin

    @testset verbose=true "GMM.jl" begin
        include("GMM_tests.jl")
    end
    @testset verbose=true "emalg.jl" begin
        include("emalg_tests.jl")
    end
end






