from main.distribution.Banana import Banana
from main.kernel.GaussianKernel import GaussianKernel
from main.mcmc.MCMCChain import MCMCChain
from main.mcmc.MCMCParams import MCMCParams
from main.mcmc.output.StatisticsOutput import StatisticsOutput
from main.mcmc.samplers.MCMCHammer import MCMCHammer
from numpy.ma.core import zeros
import cProfile
import pstats

def main():
    distribution = Banana(dimension=8)
    
    sigma = GaussianKernel.get_sigma_median_heuristic(distribution.sample(1000).samples)
    print "using sigma", sigma
    kernel = GaussianKernel(sigma=sigma)
    
    mcmc_sampler = MCMCHammer(distribution, kernel, distribution.sample(1000).samples)
    
    start = zeros(distribution.dimension)
    mcmc_params = MCMCParams(start=start, num_iterations=10000)
    chain = MCMCChain(mcmc_sampler, mcmc_params)
    
    chain.append_mcmc_output(StatisticsOutput())
    chain.run()
    
cProfile.run("main()", "profile.tmp")
p = pstats.Stats("profile.tmp")
p.sort_stats("cumulative").print_stats(10)