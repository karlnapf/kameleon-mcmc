from main.tools.GitTools import GitTools
from os import makedirs
from pickle import dump
import os

class SingleChainExperiment(object):
    # filename dictionaries
    filenames = { "mcmc_chain_instance": "mcmc_chain_instance.bin", \
                  "output_folder": "output", \
                  "gitversion": "gitversion.txt", \
                  "parameters": "parameters.txt"
               }
    
    def __init__(self, mcmc_chain, folder_prefix=""):
        self.mcmc_chain = mcmc_chain
        self.folder_prefix=folder_prefix
        
        # build foldername
        distname = mcmc_chain.mcmc_sampler.distribution.__class__.__name__
        sampler_name = mcmc_chain.mcmc_sampler.__class__.__name__
        foldername = folder_prefix + sampler_name + "_" + distname + "_"
        
        # create new folder for experiment
        num = 0;
        while os.path.exists(foldername + str(num) + os.sep):
            num += 1
        
        foldername = foldername + str(num) + os.sep
        self.foldername = foldername
        makedirs(foldername)
            
        # save git version and branch to file
        f = open(foldername + self.filenames["gitversion"], "w")
        f.write(GitTools.get_hash() + os.linesep)
        f.write(GitTools.get_branch() + os.linesep)
        f.close()
        
        # save some infos to a file
        f = open(foldername + self.filenames["parameters"], "w")
        f.write(str(mcmc_chain.mcmc_params) + os.linesep)
        f.write(str(mcmc_chain.mcmc_sampler) + os.linesep)
        f.write(str(mcmc_chain.mcmc_sampler.distribution) + os.linesep)
        f.write(str(mcmc_chain) + os.linesep)
        f.close()
        
    def run(self):
        self.mcmc_chain.run()
        
        # save distribution instance
        output_folder = self.foldername + self.filenames["output_folder"] + os.sep
        makedirs(output_folder)
        f = open(output_folder + self.filenames["mcmc_chain_instance"], "w")
        dump(self.mcmc_chain, f)
        f.close()
        
#if __name__ == '__main__':
#    distribution = Ring()
#    kernel = GaussianKernel(sigma=1)
#    mcmc_hammer = MCMCHammerWindow(distribution, kernel)
#    
#    start = array([-2, -2])
#    mcmc_params = MCMCParams(start=start, num_iterations=20000, burnin=5000)
#    mcmc_chain = MCMCChain(mcmc_hammer, mcmc_params)
#    mcmc_chain.append_mcmc_output(ProgressOutput())
#    
#    experiment_dir = expanduser("~") + os.sep + "mcmc_hammer_experiments" + os.sep
#    experiment = SingleChainExperiment(mcmc_chain, folder_prefix=experiment_dir)
#    experiment.run()
    
#    mcmc_chain.append_mcmc_output(ProgressOutput())
#    mcmc_chain.run()
#    
#    print distribution.emp_quantiles(mcmc_chain.samples)
#    
#    
#    num_iterations = self.mcmc_params.num_iterations
#    self.samples = zeros((num_iterations, self.mcmc_hammer.distribution.dimension))
#    self.ratios = zeros(num_iterations)
#    self.log_liks = zeros(num_iterations)
#    self.accepteds = zeros(num_iterations)
#    
