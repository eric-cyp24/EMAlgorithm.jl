using Test, Random
using EMAlgorithm


## set ups ##
num_comp, dims = 4, 3;

mu1      = [0 1 2 0; 0 1 2 1; 0 1 2 2]*2
mu2      = (mu1.-0.5)*1.5;
sigs, w  = stack([[1 0 0; 0 1 0; 0 0 1] for _ in 1:num_comp])*0.1, [0.1, 0.2, 0.3, 0.4];
gmm_true = GaussianMixtureModel(mu1, sigs, w);
X        = generatedata(gmm_true, num_comp*4000);
perm     = collect(1:num_comp)
#gmm_init = GaussianMixtureModel(mu2, sigs);

@testset verbose=true "emalg tests" begin
    local gmm1_mu, idx1, gmm2, gmm2_mu, gmm3_mu
    @testset "emalgorithm!" begin
        gmm1 = GaussianMixtureModel(mu2, sigs*5);
        llh1 = emalgorithm!(gmm1, X, 8000);
        idx1 = sortperm(gmm1.weights);
        gmm1_mu = stack([g.μ for g in gmm1.components])
        @test gmm1.weights[idx1] ≈ gmm_true.weights rtol=0.01;
        @test gmm1_mu[:,idx1] ≈ mu1 rtol=0.1;
    end

    @testset "emalg fixedweight!" begin
        gmm2,llh2 = nothing,-Inf
        for i in 1:24
            gmm = GaussianMixtureModel(view(mu2,:,perm), sigs*10, w);
            llh = emalgorithm_fixedweight!(gmm, X, 4000);
            gmm2,llh2 = llh > llh2 ? (gmm,llh) : (gmm2,llh2)
            if sortperm(EMAlgorithm.datadistribution(gmm2,X)) == [1,2,3,4] break end
            global perm = shuffle(perm)
        end
        gmm2_mu = stack([g.μ for g in gmm2.components])
        @test gmm2.weights == gmm_true.weights == w;
        @test gmm2_mu ≈ mu1 rtol=0.1;
        @test gmm1_mu[:,idx1] ≈ gmm2_mu rtol=0.01;
    end

    @testset "emalg fixedweight mprocess" begin
        gmm3 = GaussianMixtureModel(view(mu2,:,perm), sigs*10, w)
        llh3 = emalgorithm_fixedweight_mprocess!(gmm3, X, 4000)
        gmm3_mu = stack([g.μ for g in gmm3.components])
        @test gmm3.weights == gmm_true.weights == w
        @test gmm3_mu ≈ gmm2_mu rtol=0.01
        @test gmm3 == gmm2
    end

end







