### A Pluto.jl notebook ###
# v0.19.32

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 9b7bef9c-854b-11ee-2d30-b1723ec29902
# ╠═╡ show_logs = false
begin
	import Pkg
	#Pkg.add("PlutoUI")
	#Pkg.add("ProgressLogging")
    # activate the shared project environment
    Pkg.activate(Base.current_project())
    # instantiate, i.e. make sure that all packages are downloaded
    Pkg.instantiate()

    using Distributions, StatsPlots, HDF5, LinearAlgebra
	using PlutoUI, ProgressLogging
	using EMAlgorithm
end

# ╔═╡ 9617f29c-a8e9-4c2f-aedc-c0747c6a9771
md"""
# EM Algorithm

"""

# ╔═╡ be9db8c8-714a-4270-9b15-31e010b226cc
md"""
## Models and Data Generation
A generative model generates a set of data D = (X, Z) = (Observable data, Latened Data).

e.g. A 2D Multivariate Gaissian Mixture Model (GMM), where the Observable data and Latened data are power traces (in LDA subspace) and corresponding Intermediate Values, respectively.
"""

# ╔═╡ a9f124e4-b722-4b44-9e56-098695d7ded1
# randomly generates a Gausssian Mixture Model (GMM) and the complete data D
begin
	gmm_true = generatemodels(3);
	X = generatedata(gmm_true, 3000);
	X_size = [Int(round(w*3000)) for w in gmm_true.weights];
	Z = cat([ones(Int,n)*z for (z,n) in enumerate(X_size)]...; dims=1);
	D = [(x,z) for (x,z) in zip(X,Z)];
	idx = cat(0,accumulate(+,X_size);dims=1);

	# plot the true model and the complete data D
	plot(;size=(800,600),lims=(-22.5,22.5))
	Z_color = [RGB(1,0,0), RGB(0,1,0), RGB(0,0,1)];
	for i in 1:3
		plotdatascatter!(X[:,idx[i]+1:idx[i+1]], color=Z_color[i], markersize=2.5)
	end
	plotGMM!(gmm_true; colors=Z_color)
	plot!(title="Generative Models & Complete Data D -- X color-coded with Z")
end

# ╔═╡ db67c73b-15d6-4911-b7ae-3c967e8078a6
md"""

"""

# ╔═╡ d82a485b-6abf-467e-8120-518caccab632
gmm_init = generatemodels(3);

# ╔═╡ 6970b753-7aa6-4c64-8e62-8332e5fa46bf
gmm = copy(gmm_init); nothing

# ╔═╡ e89c6308-8b2e-4466-a882-1385d5de8114
@bind CheckBox1 MultiCheckBox(["gmm_true", "gmm_init"])

# ╔═╡ a44f5089-1206-4563-8357-516eb5aa9a41
begin
	plot_gmm_true = "gmm_true" in CheckBox1;
	plot_gmm_init = "gmm_init" in CheckBox1;
	nothing
end

# ╔═╡ 204d98b5-288a-4269-b1e7-f232be0bba78
begin
	# observed/incomplete data X 
	plot(;size=(800,600),lims=(-22.5,22.5))
	plotdatascatter!(X, color=RGB(0.33,0.33,0.33), markersize=2.5)
	if plot_gmm_init
		plotGMM!(gmm_init; colors=Z_color, linestyle=:dash)
	end
	if plot_gmm_true
		plotGMM!(gmm_true; colors=Z_color)
	end
	plot!(title="Observed/Incomplete Data X & Random Initial GMM")
end

# ╔═╡ f267b7f5-774c-42c2-a2c0-8e54d7c3cf3c
md"""
EM Algorithm
"""

# ╔═╡ 3eb8896d-2215-4710-b566-d467d906c8cb
begin
	num_epoch = 5000
	gammas = Matrix{Float64}(undef, size(X)[end], length(gmm))
	gmm_history, weight = [copy(gmm)], copy(gmm.weights)
    likelihoods = Vector{Float64}(undef,0)
    @progress for epoch in 1:num_epoch
        # estep
		llh = EMAlgorithm.estep!(gammas, X, gmm)
        push!(likelihoods, llh)
        
        # mstep
        EMAlgorithm.mstep!(gmm, X, gammas)
		push!(gmm_history, copy(gmm))

        if (length(likelihoods)>10) && 
		   (likelihoods[end]-likelihoods[end-1])/(likelihoods[end]-likelihoods[2]) < 10e-7
            break
        end
    end
end

# ╔═╡ c3f83355-5b57-47e4-b2f0-5907f065a9c5
md"""
show true model: $(@bind show_true_gmm CheckBox())
"""

# ╔═╡ 44d71f7f-0094-4148-a263-c568d68c8c60
md"""
show Z:  $( 
	@bind Z_info Select(["unknown", "estimate", "true"]) 
)
"""

# ╔═╡ be35bc27-01ba-4ef9-a571-764eb2b4f266
md"""
iteration: $(
	@bind t Slider(1:length(gmm_history), show_value=true)
)
"""

# ╔═╡ 2a87fcdd-905e-40f5-a5c0-14597a442796
begin
	# observed/incomplete data X 
	plot(;size=(800,600),lims=(-22.5,22.5),
	      title="EM Algorithm & Latent Variables Estimation")
	if Z_info == "unknown"
		plotdatascatter!(X, color=RGB(0.33,0.33,0.33), markersize=2.5)
	elseif Z_info == "estimate"
		EMAlgorithm.estep!(gammas, X, gmm_history[t])
		Z_estimate = [RGB(g...) for g in eachrow(gammas)]
		scatter!(X[1,:],X[2,:]; color=Z_estimate, markersize=2.5,
				 label="", markeralpha=0.7, markerstrokewidth=-1)
	elseif Z_info == "true"
		for i in 1:3
			plotdatascatter!(X[:,idx[i]+1:idx[i+1]], color=Z_color[i], markersize=2.5)
		end
	else
		plotdatascatter!(X, color=RGB(0.33,0.33,0.33), markersize=2.5)
	end

	
	plotGMM!(gmm_history[t]; colors=Z_color, linestyle=:dash)
	if show_true_gmm
		plotGMM!(gmm_true; colors=Z_color)
	end
	plot!()
end

# ╔═╡ Cell order:
# ╠═9b7bef9c-854b-11ee-2d30-b1723ec29902
# ╠═9617f29c-a8e9-4c2f-aedc-c0747c6a9771
# ╟─be9db8c8-714a-4270-9b15-31e010b226cc
# ╠═a9f124e4-b722-4b44-9e56-098695d7ded1
# ╠═db67c73b-15d6-4911-b7ae-3c967e8078a6
# ╠═d82a485b-6abf-467e-8120-518caccab632
# ╠═6970b753-7aa6-4c64-8e62-8332e5fa46bf
# ╟─e89c6308-8b2e-4466-a882-1385d5de8114
# ╟─a44f5089-1206-4563-8357-516eb5aa9a41
# ╟─204d98b5-288a-4269-b1e7-f232be0bba78
# ╟─f267b7f5-774c-42c2-a2c0-8e54d7c3cf3c
# ╠═3eb8896d-2215-4710-b566-d467d906c8cb
# ╟─c3f83355-5b57-47e4-b2f0-5907f065a9c5
# ╟─44d71f7f-0094-4148-a263-c568d68c8c60
# ╟─be35bc27-01ba-4ef9-a571-764eb2b4f266
# ╠═2a87fcdd-905e-40f5-a5c0-14597a442796
