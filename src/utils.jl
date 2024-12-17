
#using Distributions, StatsPlots

### plot utils #####

"""
    plotMvNormal!(mvg::MvNormal; axes=[1,2], linecolor=1, kwargs...)

plot the 90%, 95%, and 99% confidence ellipse for a given Gaussian distribution.
"""
function plotMvNormal!(mvg::MvNormal; axes=[1,2], linecolor=1, kwargs...)
    loc, cov = mvg.μ[axes], mvg.Σ[axes,axes]
    # n_std_prob: 90%:2.146, 95%:2.448, 99%3.035, 99.5%: 3.255 (df=2)
    covellipse!(loc,cov; n_std=2.146, linecolor, linealpha=1.0, fillalpha=0, kwargs...)
    covellipse!(loc,cov; n_std=2.448, linecolor, linealpha=0.5, fillalpha=0, kwargs...)
    covellipse!(loc,cov; n_std=3.035, linecolor, linealpha=0.2, fillalpha=0, kwargs...)
    gui()
    return plot!()
end

"""
    plotGMM!(gmm::GaussianMixtureModel; axes=[1,2], show=true, colors=nothing, kwargs...)

plot the 95% and 99% confidence ellipses of the given gmm components.
"""
function plotGMM(gmm::GaussianMixtureModel; axes=[1,2], show=true, colors=nothing, kwargs...)
    plot(); plotGMM!(gmm;axes,show,colors,kwargs...)
end
function plotGMM!(gmm::GaussianMixtureModel; axes=[1,2], show=true, colors=nothing, kwargs...)
    colors = isnothing(colors) ? (1:length(gmm)) : colors
    for (i,(c,gm)) in enumerate(zip(colors, gmm.components))
        loc, cov = gm.μ[axes], gm.Σ[axes,axes]
        covellipse!(loc,cov; n_std=2.448, fillalpha=0, linealpha=1, 
                             linewidth=1, linecolor=c, label="$(gmm.weights[i]*100) %",kwargs...)
        covellipse!(loc,cov; n_std=3.035, fillalpha=0, linealpha=0.5, 
                             linewidth=0.7, linecolor=c, kwargs...,label="")
    end
    show && gui()
    return plot!()
end

"""
    plotdatascatter!(data::AbstractMatrix; axes=[1,2], show=true, kwargs...)

scatterplot of the data along the given two axes.
"""
function plotdatascatter(data::AbstractMatrix; axes=[1,2], show=true, kwargs...)
    plot(); plotdatascatter!(data; axes, show, kwargs...)
end
function plotdatascatter!(data::AbstractMatrix; axes=[1,2], show=true, kwargs...)
    x,y = eachrow(data[axes,:])
    scatter!(x, y; label="", markersize=1, markeralpha=0.7, markerstrokewidth=-1, color=:grey, kwargs...)
    show && gui()
    return plot!()
end

"""
    plotEM!(data::AbstractMatrix, gmm::GaussianMixtureModel; 
            axes=[1,2], colors=nothing, show=true, kwargs...)

plot the scatterplot of the given data and the 95% and 99% 
confidence ellipses of the gmm components.
"""
function plotEM(data::AbstractMatrix, gmm::GaussianMixtureModel; 
                 axes=[1,2], colors=nothing, show=true, kwargs...)
    plot(); plotEM!(data, gmm; axes, colors, show, kwargs...)
end
function plotEM!(data::AbstractMatrix, gmm::GaussianMixtureModel; 
                 axes=[1,2], colors=nothing, show=true, kwargs...)
    plotdatascatter!(data; axes, show=false, kwargs...)
    plotGMM!(gmm; axes, show, colors, kwargs...)
end

###################
