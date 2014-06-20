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
import sys
import os


if len(sys.argv) <= 1:
    print "usage:", str(sys.argv[0]).split(os.sep)[-1], "<which_plot[mmd,mean]> <highlight[SM,AM,KAMH;optional]> <directory[optional]>"
    exit()
    
which_plot = str(sys.argv[1])

if len(sys.argv) >= 3:
    highlight = str(sys.argv[2])
else:
    highlight=None
    
if len(sys.argv) >= 4:
    directory = str(sys.argv[3])
else:
    directory = '/nfs/home2/dino/kamh-results/mmds/'

mpl.rcParams['text.usetex']=True
mpl.rcParams['text.latex.unicode']=True

sampler_names_short = ["SM","AM-FS","AM-LS","KAMH-LS"]
sampler_names = ["StandardMetropolis","AdaptiveMetropolis","AdaptiveMetropolisLearnScale","KameleonWindowLearnScale"]

colours = ['blue', 'red', 'magenta', 'green']


ii=0
for sampler_name in sampler_names:
    filename = directory+sampler_name+"_mmds.bin"
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
    if highlight == "SM":
        condition = sampler_name == "StandardMetropolis"
    elif highlight == "AM":
        condition = sampler_name == "AdaptiveMetropolis" or sampler_name == "AdaptiveMetropolisLearnScale"
    elif highlight == "KAMH":
        condition = sampler_name == "KameleonWindowLearnScale"
    else:
        condition = True
    
    if condition:
        errorbar(upto,means,yerr=yerr,fmt='o-',color=colours[ii])
    else:
        errorbar(upto,means,yerr=yerr,fmt='o-',alpha=0.1,color=colours[ii],ecolor='0.9')
    ii+=1
    #fill_between(upto,means-yerr,means+yerr,color="grey",alpha=0.2)
legend(sampler_names_short)
xlabel("number of samples",fontsize=12)
xlim(460,5040)
if which_plot == "mean":
    ylabel("mean distance " + r"$\left\Vert \hat{\mu}_{\theta}-\mu_{\theta}^{b}\right\Vert _{2}$",fontsize=12)
if which_plot == "mmd":
    ylabel("MMD from the benchmark sample",fontsize=12)
#
show()
#savefig("/nfs/home2/dino/Dropbox/talks/"+which_plot+"_comparisonC.pdf", bbox_inches='tight')