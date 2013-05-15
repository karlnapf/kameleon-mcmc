from main.distribution.Ring import Ring
from main.kernel.GaussianKernel import GassianKernel
from main.mcmc.MCMCChain import MCMCChain
from main.mcmc.MCMCParams import MCMCParams
from main.mcmc.output.PlottingOutput import PlottingOutput
from main.mcmc.output.ProgressOutput import ProgressOutput
from main.mcmc.samplers.MCMCHammer import MCMCHammer
from main.mcmc.samplers.MCMCSampler import MCMCSampler
from main.tools.Visualise import Visualise
from numpy.core.function_base import linspace
from numpy.core.numeric import array

class MCMCHammerWindow(MCMCHammer):
    def __init__(self, distribution, kernel, eta=0.1, gamma=0.1, window_size=5000, thinning_factor=10):
        MCMCHammer.__init__(self, distribution, kernel, Z=None, eta=0.1, gamma=0.1)
        
        self.kernel = kernel
        self.eta = eta
        self.gamma = gamma
        self.window_size = window_size
        self.thinning_factor = thinning_factor
    
    def init(self, start):
        self.Z = start
        MCMCSampler.init(self, start)
    
    def update(self, samples, ratios):
        """
        Updates the sliding window of samples to use
        """
        # use samples from history window, thinned out
        if len(samples) > 0:
            sample_idxs = range(max(0, len(samples) - self.window_size + 1), \
                              len(samples), self.thinning_factor)
            self.Z = samples[sample_idxs]
        
if __name__ == '__main__':
    distribution = Ring()
    kernel = GassianKernel(sigma=1)
    mcmc_sampler = MCMCHammerWindow(distribution, kernel)
    
    start = array([[-2, -2]])
    mcmc_params = MCMCParams(start=start, num_iterations=10000)
    chain = MCMCChain(distribution, mcmc_sampler, mcmc_params)
    
    chain.append_mcmc_output(ProgressOutput())
    Xs = linspace(-5, 5, 50)
    Ys = linspace(-5, 5, 50)
    chain.append_mcmc_output(PlottingOutput(Xs, Ys, plot_from=2000))
    chain.run()
    
    Visualise.visualise_distribution(distribution, chain.samples)
