from main.distribution.Banana import Banana
from main.mcmc.MCMCChain import MCMCChain
from main.mcmc.MCMCParams import MCMCParams
from main.mcmc.output.StatisticsOutput import StatisticsOutput
from main.mcmc.samplers.StandardMetropolis import StandardMetropolis
from numpy.core.numeric import zeros
from numpy.ma.core import array
import cProfile
import pstats


def main():
    distribution = Banana(dimension=8, bananicity=0.1, V=100.0)
    
    mcmc_sampler = StandardMetropolis(distribution)
    
    start = array([0.0, -3.0])
    start=zeros(distribution.dimension)
    mcmc_params = MCMCParams(start=start, num_iterations=20000)
    chain = MCMCChain(mcmc_sampler, mcmc_params)
    
    chain.append_mcmc_output(StatisticsOutput())
    chain.run()
    
    print distribution.emp_quantiles(chain.samples)
    
cProfile.run("main()", "profile.tmp")
p = pstats.Stats("profile.tmp")
p.sort_stats("cumulative").print_stats(10)