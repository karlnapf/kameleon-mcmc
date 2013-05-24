from main.experiments.ExperimentAggregator import ExperimentAggregator
from numpy.linalg.linalg import norm
from numpy.ma.core import arange, zeros, mean, std

class SingleChainExperimentAggregator(ExperimentAggregator):
    def __init__(self, folders, ref_quantiles=arange(0.1,1,0.1)):
        ExperimentAggregator.__init__(self, folders)
        self.ref_quantiles=ref_quantiles
    
    def __process_results__(self):
        if len(self.experiments)==0:
            print "no experiments to process"
            return
        
        # burnin is the same for all chains
        burnin = self.experiments[0].mcmc_chain.mcmc_params.burnin
        
        quantiles = zeros((len(self.experiments), len(self.ref_quantiles)))
        norm_of_means = zeros(len(self.experiments))
        acceptance_rates = zeros(len(self.experiments))
        
        for i in range(len(self.experiments)):
            burned_in = self.experiments[i].mcmc_chain.samples[burnin:, :]
            quantiles[i, :] = self.experiments[i].mcmc_chain.mcmc_sampler.distribution.emp_quantiles( \
                              burned_in, self.ref_quantiles)
            norm_of_means[i] = norm(mean(burned_in, 0))
            acceptance_rates[i] = mean(self.experiments[i].mcmc_chain.accepteds[burnin:])

        mean_quantiles = mean(quantiles, 0)
        std_quantiles = std(quantiles, 0)
        
        print "quantiles:"
        for i in range(len(self.ref_quantiles)):
            print mean_quantiles[i], "+-", std_quantiles[i]
        
        print "norm of means:"
        print mean(norm_of_means), "+-", std(norm_of_means)
        
        print "acceptance rate:"
        print mean(acceptance_rates)
