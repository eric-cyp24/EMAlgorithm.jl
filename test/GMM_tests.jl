
using StatsBase
## set ups ##
num_comp, dims = 4, 3;
mu1, w = [0 1 2 0; 0 1 2 1; 0 1 2 2]*2, [0.1, 0.2, 0.3, 0.4]
sigs   = stack([[1 0 0; 0 1 0; 0 0 1] for _ in 1:num_comp])*0.1

@testset verbose=true "GMM" begin
    gmm1 = GaussianMixtureModel(mu1, sigs, w)
    # GMM attibute access
    @test stack(EMAlgorithm.means(gmm1))   == mu1
    @test stack(EMAlgorithm.covs( gmm1))   == sigs
    @test stack(EMAlgorithm.weights(gmm1)) == w
    # basic operations
    gmm2 = copy(gmm1)
    @test gmm1 == gmm2 && !(gmm1 === gmm2)
    @test length(gmm1) == 4
    @test ndims(gmm1)  == 3
    @test size(gmm1)   == (4,(3,))
    # statistics
    @test mean(gmm1) ≈ mean( mu1, FrequencyWeights(w); dims=2) rtol=0.001
    @test cov( gmm1) ≈ wsum(sigs, w, 3) + cov(mu1, FrequencyWeights(w), 2) rtol=0.001
end


