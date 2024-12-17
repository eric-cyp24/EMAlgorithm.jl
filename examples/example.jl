
using StatsPlots
using EMAlgorithm
using EMAlgorithm:plotdatascatter!

dim      = 4
num_comp = 3
δ        = 10e-5
size     = (1200,900)
axes     = [3,4]
Z_color  = [RGB(0,0,1), RGB(1,0,0), RGB(0,1,0)]

gmm_true = generatemodels(num_comp; dims=dim)
X        = generatedata(gmm_true, num_comp*2000)
gmm_init = generatemodels(num_comp; dims=dim)

plot(;size)
plotdatascatter!(X; axes, show=false)
plotGMM!(gmm_init; axes)
sleep(3)
#print("\rpress enter to close plot          ");readline()


gmm = copy(gmm_init)
llh = emalgorithm_anime!(gmm, X, 3000; δ, axes, size)

plot(;size)
idx = cat(0,accumulate(+,[Int(round(w*6000)) for w in gmm_true.weights]);dims=1)
for i in 1:3
    plotdatascatter!(X[:,idx[i]+1:idx[i+1]]; axes, color=Z_color[i], markersize=2.5)
end
plotGMM!(gmm_true; axes)
plotGMM!(gmm; axes, linestyle=:dash)
print("\rpress enter to close plot          ");readline()




