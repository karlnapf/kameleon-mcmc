from main.distribution.Banana import Banana
from main.kernel.GaussianKernel import GaussianKernel
from main.mcmc.MCMCChain import MCMCChain
from main.mcmc.MCMCParams import MCMCParams
from main.mcmc.output.StatisticsOutput import StatisticsOutput
from main.mcmc.samplers.MCMCHammerWindowLearnScale import \
    MCMCHammerWindowLearnScale
from numpy.core.numeric import zeros
from numpy.ma.core import array
import cProfile
import pstats


def main():
    distribution = Banana(dimension=8, bananicity=0.1, V=100.0)
    
    sigma = GaussianKernel.get_sigma_median_heuristic(distribution.sample(1000).samples)
    sigma=5
    print "using sigma", sigma
    kernel = GaussianKernel(sigma=sigma)
    
    mcmc_sampler = MCMCHammerWindowLearnScale(distribution, kernel)
    
    start = array([0.0, -3.0])
    start=zeros(distribution.dimension)
    mcmc_params = MCMCParams(start=start, num_iterations=20000)
    chain = MCMCChain(mcmc_sampler, mcmc_params)
    
#    chain.append_mcmc_output(PlottingOutput(distribution, plot_from=3000))
    chain.append_mcmc_output(StatisticsOutput())
    chain.run()
    
    print distribution.emp_quantiles(chain.samples)
    
#    Visualise.visualise_distribution(distribution, chain.samples)

cProfile.run("main()", "profile.tmp")
p = pstats.Stats("profile.tmp")
p.sort_stats("cumulative").print_stats(10)