from main.experiments.SingleChainExperiment import SingleChainExperiment
from main.tools.GitTools import GitTools
from numpy.linalg.linalg import norm
from numpy.ma.core import mean, std, arange, zeros
from pickle import load
import os

class SingleChainAggregator(object):
    def __init__(self, folders):
        self.folders = folders
        
        f = open(folders[0] + SingleChainExperiment.filenames["gitversion"])
        ref_githash = f.readline().strip()
        ref_gitbranch = f.readline().strip()
        f.close()
        
        for folder in folders:
            # assert that all experiments ran under the same git version
            assert(os.path.exists(folder))
            
            f = open(folder + SingleChainExperiment.filenames["gitversion"])
            githash = f.readline().strip()
            gitbranch = f.readline().strip()
            f.close()
            
            # assert that all git hashs are equal
            assert(githash == ref_githash)
            assert(gitbranch == ref_gitbranch)
            
            # this might be false if script was altered after experiment, comment out then
            if githash != GitTools.get_hash() or gitbranch != GitTools.get_branch():
                print "WARNING: git version of experiment not equal to this"
    
    def load_parameter(self, parameter):
        # load number of iterations
        f = open(self.folders[0] + SingleChainExperiment.filenames["parameters"])
        lines = f.readlines()
        f.close()
        parameters = "\n".join(lines)
        idx_a = parameters.find(parameter)
        idx_b = parameters.find(",", idx_a)
        return parameters[idx_a:idx_b]
    
    def load_raw_results(self):
        self.experiments = []
        for i in range(len(self.folders)):
            filename = self.folders[i] + SingleChainExperiment.filenames["output_folder"] + \
                     os.sep + SingleChainExperiment.filenames["output"]
            
            try:
                f = open(filename , "r")
                self.experiments.append(load(f))
                f.close()
            except IOError:
                print "skipping", filename, "due to IOError"
                print "cluster_error output"
                ef=open(self.folders[i] + "cluster_error.txt")
                print ef.readlines()
                pass
            
    def post_process(self, ref_quantiles=arange(0.1, 1, 0.1)):
        n = len(self.experiments)
        print "post processing", n, "experiments"
        
        # burnin is the same for all chains
        burnin = self.experiments[0].mcmc_chain.mcmc_params.burnin
        
        quantiles = zeros((len(self.experiments), len(ref_quantiles)))
        norm_of_means = zeros(len(self.experiments))
        acceptance_rates = zeros(len(self.experiments))
        
        for i in range(n):
            # burned in view
            burned_in = self.experiments[i].mcmc_chain.samples[burnin:, :]
        
            # quantiles
            quantiles[i, :] = self.experiments[i].mcmc_chain.mcmc_sampler.distribution.emp_quantiles(burned_in, ref_quantiles)
            
            # norm of means
            norm_of_means[i] = norm(mean(burned_in, 0))
            
            acceptance_rates[i] = mean(self.experiments[i].mcmc_chain.accepteds[burnin:])

        mean_quantiles = mean(quantiles, 0)
        std_quantiles = std(quantiles, 0)
        
        print "quantiles:"
        for i in range(len(ref_quantiles)):
            print mean_quantiles[i], "+-", std_quantiles[i]
        
        print "norm of means:"
        print mean(norm_of_means), "+-", std(norm_of_means)
        
        print "acceptance rate:"
        print mean(acceptance_rates)
