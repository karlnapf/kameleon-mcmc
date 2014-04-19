from pickle import load, dump
#from kameleon_mcmc.mcmc import *  # @UnusedWildImport
from numpy.oldnumeric.random_array import permutation
from matplotlib.pyplot import title, plot, figure, show, draw, clf, contour,\
    xlim, ylim, imshow
from numpy.ma.core import arange, mean, reshape, shape, sqrt, floor, zeros
from kameleon_mcmc.kernel.PolynomialKernel import PolynomialKernel

from numpy.linalg.linalg import norm
from numpy.lib.npyio import loadtxt

#from kameleon_mcmc.mcmc.samplers.KameleonWindowLearnScale import KameleonWindowLearnScale

plotting=False
pkernel = PolynomialKernel(degree=3)

samples_long = loadtxt("/nfs/home2/dino/kamh-results/StandardMetropolis_PseudoMarginalHyperparameterDistribution_merged_samples.txt")
samples_long = samples_long[:10000]
# f_long=open("/nfs/home2/dino/kamh-results/long_experiment_output.bin")
# experiment_long=load(f_long)
# f_long.close()
# thin_long=100
# mcmc_chain_long=experiment_long.mcmc_chain
# burnin=mcmc_chain_long.mcmc_params.burnin
# indices_long = range(burnin, mcmc_chain_long.iteration,thin_long)
# samples_long=mcmc_chain_long.samples[indices_long]
mu_long = mean(samples_long,0)

print 'using this many samples for the long chain: ', shape(samples_long)[0]



how_many_chains = 20
stats_granularity = 10

path_above = "/nfs/home2/dino/git/kameleon-mcmc/main/gp/scripts/glass_gaussian_ard/"
#path_above = "/nfs/data3/ucabhst/kameleon_experiments/glass_ard/"
path_below = "output/experiment_output.bin"

#sampler_names = ["KameleonWindowLearnScale", "AdaptiveMetropolisLearnScale","AdaptiveMetropolis"]
sampler_names = ["StandardMetropolis"]
path_temp = "_PseudoMarginalHyperparameterDistribution_#/"

for sampler_name in sampler_names:
    mean_dist=zeros((stats_granularity,how_many_chains))
    mmds = zeros((stats_granularity,how_many_chains))
    for num_chain in range(0,how_many_chains):
        path = path_temp.replace('#',str(num_chain))
        print sampler_name+path
        f=open( path_above+sampler_name+path+path_below )
        experiment=load(f)
        f.close()
        mcmc_chain=experiment.mcmc_chain
        burnin=mcmc_chain.mcmc_params.burnin
        print 'burnin: ', burnin
        print 'total chain length: ', mcmc_chain.iteration
        thin=20
        print 'thinning by: ', thin
        #indices = range(0, mcmc_chain.iteration,thin)
        indices = range(0, 100000, thin)
        print 'after thinning: ', shape(indices)[0]
        upto_increment = int(shape(indices)[0]/stats_granularity)
        upto = range(upto_increment, shape(indices)[0]+1, upto_increment)
        print 'now computing mmds...'
        for jj in range(0, stats_granularity):
            print 'using this many samples: ', upto[jj]
            samples=mcmc_chain.samples[indices[:upto[jj]]]
            mu = mean(samples,0)
            mmds[jj,num_chain]=pkernel.estimateMMD(samples,samples_long,unbiased=True)
            print 'MMD (poly-3): ', mmds[jj,num_chain]
            mean_dist[jj,num_chain]=norm(mu-mu_long)
            print 'distance to the long-run mean: ', mean_dist[jj,num_chain]
    save_filename = "/nfs/home2/dino/kamh-results/mmds/"+sampler_name+"_mmds.bin"
    save_f = open(save_filename,"w")
    dump([upto, mmds, mean_dist], save_f)