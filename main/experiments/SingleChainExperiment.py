from main.experiments.Experiment import Experiment

class SingleChainExperiment(Experiment):
    def __init__(self, mcmc_chain, experiment_dir="", name=None):

        if name is None:
            name=mcmc_chain.mcmc_sampler.__class__.__name__ + "_" + \
                 mcmc_chain.mcmc_sampler.distribution.__class__.__name__
                 
        self.mcmc_chain = mcmc_chain
        Experiment.__init__(self, experiment_dir, name)
        
    
    def __run_experiment__(self):
        self.mcmc_chain.run()

    def __str__(self):
        s=self.__class__.__name__+ "=["
        s += "mcmc_chain="+ str(self.mcmc_chain)
        s += ", " + Experiment.__str__(self)
        s += "]"
        return s