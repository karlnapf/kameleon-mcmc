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
            assert(githash==ref_githash)
            assert(gitbranch==ref_gitbranch)
            
            # this might be false if script was altered after experiment, comment out then
            assert(githash == GitTools.get_hash())
            assert(gitbranch == GitTools.get_branch())
    
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
        self.mcmc_chains = [None for _ in range(len(self.folders))]
        for i in range(len(self.folders)):
            filename = self.folders[i] + SingleChainExperiment.filenames["output_folder"] + \
                     os.sep + SingleChainExperiment.filenames["mcmc_chain_instance"]
            print "loading", filename
            f = open(filename , "r")
            self.mcmc_chains[i] = load(f)
            f.close()
        
    def post_process(self, ref_quantiles=arange(0.1, 1, 0.1)):
        n = len(self.folders)
        
        # burnin is the same for all chains
        burnin = self.mcmc_chains[0].mcmc_params.burnin
        
        quantiles = zeros((len(self.mcmc_chains), len(ref_quantiles)))
        norm_of_means = zeros(len(self.mcmc_chains))
        
        for i in range(n):
            # burned in view
            burned_in = self.mcmc_chains[i].samples[burnin:, :]
        
            # quantiles
            quantiles[i, :] = self.mcmc_chains[i].mcmc_sampler.distribution.emp_quantiles(burned_in, ref_quantiles)
            
            # norm of means
            norm_of_means[i] = norm(mean(burned_in, 0))

        mean_quantiles = mean(quantiles, 0)
        mean_norm_of_means = mean(norm_of_means)
        std_quantiles = std(quantiles, 0)
        std_norm_of_means = std(norm_of_means)
        
        print "quantiles:"
        for i in range(len(ref_quantiles)):
            print mean_quantiles[i], "+-", std_quantiles[i]
        
        print "norm of means:"
        print mean_norm_of_means, "+-", std_norm_of_means
            
        
#if __name__ == '__main__':
#    experiment_dir = expanduser("~") + os.sep + "mcmc_hammer_experiments" + os.sep
#    folders = [experiment_dir + "MCMCHammerWindow_Ring_" + str(i) + os.sep for i in range(10)]
#    ag = SingleChainAggregator(folders)
#    ag.load_raw_results()
#    ag.post_process()
#    
