from main.experiments.Experiment import Experiment

class SingleChainExperiment(Experiment):
    def __init__(self, mcmc_chain, experiment_dir=""):
        name=mcmc_chain.mcmc_sampler.__class__.__name__ + "_" + \
             mcmc_chain.mcmc_sampler.distribution.__class__.__name__
        Experiment.__init__(self, experiment_dir, name)
        
        self.mcmc_chain = mcmc_chain
    
    def __run_experiment__(self):
        self.mcmc_chain.run()
