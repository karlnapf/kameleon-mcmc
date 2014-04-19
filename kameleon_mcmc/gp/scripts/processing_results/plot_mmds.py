'''
Created on 30 Jan 2014

@author: dino
'''
from matplotlib.pyplot import figure, plot, show, legend, xlabel, ylabel,\
    savefig, fill, fill_between, errorbar, xlim
from numpy.ma.core import mean, std, shape, sqrt
from pickle import load
from kameleon_mcmc.paper_figures import latex_plot_init
import matplotlib as mpl
from numpy.ma.extras import median
mpl.rcParams['text.usetex']=True
mpl.rcParams['text.latex.unicode']=True
sampler_names = ["KameleonWindowLearnScale", "AdaptiveMetropolisLearnScale","AdaptiveMetropolis", "StandardMetropolis"]
sampler_names_short = ["KAMH-LS","AM-LS","AM-FS","SM"]
which_plot = "mmd"
#hich_plot = "mean"

for sampler_name in sampler_names:
    filename = "/nfs/home2/dino/kamh-results/mmds/"+sampler_name+"_mmds.bin"
    f = open(filename,"r")
    upto, mmds, mean_dist = load(f)
    trials=shape(mean_dist)[1]
    figure(1)
    if which_plot == "mean":
        stds = std(mean_dist,1)/sqrt(trials)
        means = mean(mean_dist,1)
    if which_plot == "mmd":
        stds = std(mmds,1)/sqrt(trials)
        means = mean(mmds,1)
    zscore=1.28
    yerr = zscore*stds
    #plot(upto,means)
    errorbar(upto,means,yerr=yerr,fmt='o-')
    #fill_between(upto,means-zscore*stds,means+zscore*stds,color="grey")
legend(sampler_names_short)
xlabel("number of samples",fontsize=12)
xlim(460,5040)
if which_plot == "mean":
    ylabel("mean distance " + r"$\left\Vert \hat{\mu}_{\theta}-\mu_{\theta}^{b}\right\Vert _{2}$",fontsize=12)
if which_plot == "mmd":
    ylabel("MMD from the benchmark sample",fontsize=12)
#
show()
#savefig("/nfs/home2/dino/Dropbox/icml2014/plots/"+which_plot+"_comparison.eps", bbox_inches='tight')