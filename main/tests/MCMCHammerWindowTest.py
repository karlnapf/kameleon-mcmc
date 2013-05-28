from IPython.core.magics.execution import pstats
from main.distribution.Banana import Banana
from main.kernel.GaussianKernel import GaussianKernel
from main.mcmc.MCMCChain import MCMCChain
from main.mcmc.MCMCParams import MCMCParams
from main.mcmc.output.StatisticsOutput import StatisticsOutput
from main.mcmc.samplers.MCMCHammerWindow import MCMCHammerWindow
from numpy.ma.core import array
import cProfile

def main():
    distribution = Banana(dimension=8, bananicity=0.1, V=100.0)
    
    sigma = 5
    print "using sigma", sigma
    kernel = GaussianKernel(sigma=sigma)
    
    mcmc_sampler = MCMCHammerWindow(distribution, kernel, num_samples_Z=1000)
    
    start = array([-2, -2])
    mcmc_params = MCMCParams(start=start, num_iterations=80000)
    chain = MCMCChain(mcmc_sampler, mcmc_params)
    
#    chain.append_mcmc_output(PlottingOutput(distribution, plot_from=2000))
    chain.append_mcmc_output(StatisticsOutput())
    chain.run()
    
#    Visualise.visualise_distribution(distribution, chain.samples)

cProfile.run("main()", "profile.tmp")
p = pstats.Stats("profile.tmp")
p.sort_stats("cumulative").print_stats(10)
