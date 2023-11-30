
using StatsPlots
using EMAlgorithm

dim      = 2
num_comp = 3
δ        = 10e-5
axes     = [1,2]
Z_color = [RGB(1,0,0), RGB(0,1,0), RGB(0,0,1)]

gmm_true = generatemodels(num_comp; dims=dim)
X        = generatedata(gmm_true, num_comp*2000)
gmm_init = generatemodels(num_comp; dims=dim)

plot(;size=(1200,800))
plotdatascatter!(X; show=false)
plotGMM!(gmm_init)
sleep(3)
#print("\rpress enter to close plot          ");readline()


gmm = copy(gmm_init)
llh = emalgorithm_anime!(gmm, X, 3000; δ, axes)


idx = cat(0,accumulate(+,[Int(round(w*6000)) for w in gmm_true.weights]);dims=1)
plot(;size=(1200,900))
for i in 1:3
    plotdatascatter!(X[:,idx[i]+1:idx[i+1]], color=Z_color[i], markersize=2.5)
end
plotGMM!(gmm)
print("\rpress enter to close plot          ");readline()




