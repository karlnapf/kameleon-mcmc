from main.distribution.Banana import Banana
from main.kernel.GaussianKernel import GaussianKernel
from main.mcmc.MCMCChain import MCMCChain
from main.mcmc.MCMCParams import MCMCParams
from main.mcmc.output.PlottingOutput import PlottingOutput
from main.mcmc.output.ProgressOutput import ProgressOutput
from main.mcmc.output.StatisticsOutput import StatisticsOutput
from main.mcmc.samplers.MCMCHammer import MCMCHammer
from main.tools.Visualise import Visualise
from numpy.ma.core import array

if __name__ == '__main__':
    distribution = Banana()
    
    sigma = GaussianKernel.get_sigma_median_heuristic(distribution.sample(1000).samples)
    print "using sigma", sigma
    kernel = GaussianKernel(sigma=sigma)
    
    mcmc_sampler = MCMCHammer(distribution, kernel, distribution.sample(1000).samples, nu2=.1 ** 2)
    
    start = array([0.0, -3.0])
    mcmc_params = MCMCParams(start=start, num_iterations=10000)
    chain = MCMCChain(mcmc_sampler, mcmc_params)
    
    chain.append_mcmc_output(ProgressOutput())
    chain.append_mcmc_output(PlottingOutput(distribution, plot_from=1))
    chain.append_mcmc_output(StatisticsOutput())
    chain.run()
    
    Visualise.visualise_distribution(distribution, chain.samples)
