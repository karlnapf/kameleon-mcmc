from main.distribution.Ring import Ring
from main.experiments.SingleChainExperiment import SingleChainExperiment
from main.kernel.GaussianKernel import GaussianKernel
from main.mcmc.MCMCChain import MCMCChain
from main.mcmc.MCMCParams import MCMCParams
from main.mcmc.output.ProgressOutput import ProgressOutput
from main.mcmc.samplers.MCMCHammerWindow import MCMCHammerWindow
from numpy.ma.core import array
from posixpath import expanduser
import os

if __name__ == '__main__':
    distribution = Ring()
    kernel = GaussianKernel(sigma=1)
    mcmc_sampler = MCMCHammerWindow(distribution, kernel)
    
    start = array([-2, -2])
    mcmc_params = MCMCParams(start=start, num_iterations=20000, burnin=5000)
    mcmc_chain = MCMCChain(mcmc_sampler, mcmc_params)
    mcmc_chain.append_mcmc_output(ProgressOutput())
    
    home_dir = expanduser("~") + os.sep + "mcmc_hammer_experiments" + os.sep
    experiment = SingleChainExperiment(mcmc_chain, folder_prefix=home_dir)
    experiment.run()
