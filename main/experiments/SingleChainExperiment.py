from main.experiments.Experiment import Experiment
from pickle import load
import sys

class SingleChainExperiment(Experiment):
    def __init__(self, mcmc_chain, experiment_dir="", name=None):

        if name is None:
            name=mcmc_chain.mcmc_sampler.__class__.__name__ + "_" + \
                 mcmc_chain.mcmc_sampler.distribution.__class__.__name__
                 
        Experiment.__init__(self, experiment_dir, name)
        
        self.mcmc_chain = mcmc_chain
    
    def __run_experiment__(self):
        self.mcmc_chain.run()

if __name__ == '__main__':
    if len(sys.argv)!=2:
        print "usage:", str(sys.argv[0]), "<Experiment instance pickle filename>"
        print "example:"
        print "python run_single_chain_experiment.py ~/mcmc_hammer_experiments/MCMCHammerWindow_Ring_0/experiment_instance.bin"
        exit()
        
    filename=str(sys.argv[1])
    print "running experiment file", filename
    
    f=open(filename, 'r')
    experiment=load(f)
    f.close()
    experiment.run()