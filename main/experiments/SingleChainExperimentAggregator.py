"""
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

Written (W) 2013 Heiko Strathmann
Written (W) 2013 Dino Sejdinovic
"""

from main.experiments.ExperimentAggregator import ExperimentAggregator
from main.kernel.GaussianKernel import GaussianKernel
from main.tools.RCodaTools import RCodaTools
from matplotlib.pyplot import plot, fill_between, savefig, ylim, clf, title
from numpy.lib.npyio import savetxt
from numpy.linalg.linalg import norm
from numpy.ma.core import arange, zeros, mean, std, allclose, sqrt, asarray, \
    array
from numpy.ma.extras import median

class SingleChainExperimentAggregator(ExperimentAggregator):
    def __init__(self, folders, ref_quantiles=arange(0.1, 1, 0.1)):
        ExperimentAggregator.__init__(self, folders)
        self.ref_quantiles = ref_quantiles
    
    def __process_results__(self):
        lines = []
        if len(self.experiments) == 0:
            lines.append("no experiments to process")
            return
        
        # burnin is the same for all chains
        burnin = self.experiments[0].mcmc_chain.mcmc_params.burnin
        
        quantiles = zeros((len(self.experiments), len(self.ref_quantiles)))
        norm_of_means = zeros(len(self.experiments))
        acceptance_rates = zeros(len(self.experiments))
        ess_0 = zeros(len(self.experiments))
        ess_1 = zeros(len(self.experiments))
        ess_minima = zeros(len(self.experiments))
        ess_medians = zeros(len(self.experiments))
        ess_maxima = zeros(len(self.experiments))
        times = zeros(len(self.experiments))
        
        for i in range(len(self.experiments)):
            burned_in = self.experiments[i].mcmc_chain.samples[burnin:, :]
            
            # use precomputed quantiles if they match with the provided ones
            if hasattr(self.experiments[i], "ref_quantiles") and \
               hasattr(self.experiments[i], "quantiles") and \
               allclose(self.ref_quantiles, self.experiments[i].ref_quantiles):
                quantiles[i, :] = self.experiments[i].quantiles
            else:
                try:
                    quantiles[i, :] = self.experiments[i].mcmc_chain.mcmc_sampler.distribution.emp_quantiles(\
                                      burned_in, self.ref_quantiles)
                except NotImplementedError:
                    print "skipping quantile computations, distribution does", \
                          "not support it."
                
            dim = self.experiments[i].mcmc_chain.mcmc_sampler.distribution.dimension
            norm_of_means[i] = norm(mean(burned_in, 0))
            acceptance_rates[i] = mean(self.experiments[i].mcmc_chain.accepteds[burnin:])
            
            # dump burned in samples to disc
            # sample_filename=self.experiments[0].experiment_dir + self.experiments[0].name + "_burned_in.txt"
            # savetxt(sample_filename, burned_in)
            
            # store minimum ess for every experiment
            ess_per_covariate = asarray([RCodaTools.ess_coda(burned_in[:, cov_idx]) for cov_idx in range(dim)])
            ess_0=ess_per_covariate[0]
            ess_1=ess_per_covariate[1]
            ess_minima[i] = min(ess_per_covariate)
            ess_medians[i] = median(ess_per_covariate)
            ess_maxima[i] = max(ess_per_covariate)
            
            # save chain time needed
            ellapsed = self.experiments[i].mcmc_chain.mcmc_outputs[0].times
            times[i] = int(round(sum(ellapsed)))

        mean_quantiles = mean(quantiles, 0)
        std_quantiles = std(quantiles, 0)
        
        sqrt_num_trials=len(self.experiments)
        
        # print median kernel width sigma
        sigma=GaussianKernel.get_sigma_median_heuristic(burned_in.T)
        lines.append("median kernel sigma: "+str(sigma))
        
        lines.append("quantiles:")
        for i in range(len(self.ref_quantiles)):
            lines.append(str(mean_quantiles[i]) + " +- " + str(std_quantiles[i]/sqrt_num_trials))
        
        lines.append("norm of means:")
        lines.append(str(mean(norm_of_means)) + " +- " + str(std(norm_of_means)/sqrt_num_trials))
        
        lines.append("acceptance rate:")
        lines.append(str(mean(acceptance_rates)) + " +- " + str(std(acceptance_rates)/sqrt_num_trials))
        
        lines.append("ess dimension 0:")
        lines.append(str(mean(ess_0)) + " +- " + str(std(ess_0)/sqrt_num_trials))
        
        lines.append("ess dimension 1:")
        lines.append(str(mean(ess_1)) + " +- " + str(std(ess_1)/sqrt_num_trials))
        
        lines.append("minimum ess:")
        lines.append(str(mean(ess_minima)) + " +- " + str(std(ess_minima)/sqrt_num_trials))
        
        lines.append("median ess:")
        lines.append(str(mean(ess_medians)) + " +- " + str(std(ess_medians)/sqrt_num_trials))
        
        lines.append("maximum ess:")
        lines.append(str(mean(ess_maxima)) + " +- " + str(std(ess_maxima)/sqrt_num_trials))
        
        lines.append("times:")
        lines.append(str(mean(times)) + " +- " + str(std(times)/sqrt_num_trials))
        
        # mean as a function of iterations, normalised by time
        step = 1000
        iterations = arange(self.experiments[0].mcmc_chain.mcmc_params.num_iterations - burnin, step=step)
        
        running_means = zeros(len(iterations))
        running_errors = zeros(len(iterations))
        for i in arange(len(iterations)):
            # norm of mean of chain up 
            norm_of_means_yet = zeros(len(self.experiments))
            for j in range(len(self.experiments)):
                samples_yet = self.experiments[j].mcmc_chain.samples[burnin:(burnin + iterations[i] + 1 + step), :]
                norm_of_means_yet[j] = norm(mean(samples_yet, 0))
            
            running_means[i] = mean(norm_of_means_yet)
            error_level = 1.96
            running_errors[i] = error_level * std(norm_of_means_yet) / sqrt(len(norm_of_means_yet))
            
        plot(iterations, running_means/mean(times))
        fill_between(iterations, (running_means - running_errors)/mean(times), \
                     (running_means + running_errors)/mean(times), hold=True, color="gray")
        savefig(self.experiments[0].experiment_dir + self.experiments[0].name + "_running_mean.png")
        clf()
        
        # also store plot X and Y
        savetxt(self.experiments[0].experiment_dir + self.experiments[0].name + "_running_mean_X.txt", \
                iterations)
        savetxt(self.experiments[0].experiment_dir + self.experiments[0].name + "_running_mean_Y.txt", \
                running_means/mean(times))
        savetxt(self.experiments[0].experiment_dir + self.experiments[0].name + "_running_mean_errors.txt", \
                running_errors/mean(times))
        
        # quantile convergence of a single one
        desired_quantile=0.1
        running_quantiles=zeros(len(iterations))
        running_quantile_errors=zeros(len(iterations))
        for i in arange(len(iterations)):
            quantiles_yet = zeros(len(self.experiments))
            for j in range(len(self.experiments)):
                samples_yet = self.experiments[j].mcmc_chain.samples[burnin:(burnin + iterations[i] + 1 + step), :]
                
                # just compute one quantile for now
                quantile=self.experiments[j].mcmc_chain.mcmc_sampler.distribution.emp_quantiles(samples_yet, \
                                                                                          array([desired_quantile]))
            running_quantiles[i] = mean(quantiles_yet)
            error_level = 1.96
            running_quantile_errors[i] = error_level * std(quantiles_yet) / sqrt(len(quantiles_yet))
        
        
        plot(iterations, running_quantiles/mean(times))
        fill_between(iterations, (running_quantiles - running_quantile_errors)/mean(times), \
                     (running_quantiles + running_quantile_errors)/mean(times), hold=True, color="gray")
        
        plot([iterations.min(),iterations.max()], [desired_quantile/mean(times) for _ in range(2)])
        
        title(str(desired_quantile)+"-quantile convergence")
        savefig(self.experiments[0].experiment_dir + self.experiments[0].name + "_running_quantile.png")
        clf()
        
        # also store plot X and Y
        savetxt(self.experiments[0].experiment_dir + self.experiments[0].name + "_running_quantile_X.txt", \
                iterations)
        savetxt(self.experiments[0].experiment_dir + self.experiments[0].name + "_running_quantile_Y.txt", \
                running_quantiles/mean(times))
        savetxt(self.experiments[0].experiment_dir + self.experiments[0].name + "_running_quantile_errors.txt", \
                running_quantile_errors/mean(times))
        savetxt(self.experiments[0].experiment_dir + self.experiments[0].name + "_running_quantile_reference.txt", \
                desired_quantile/mean(times))
        
        # add latex table line
        latex_lines = []
        latex_lines.append("Sampler & Acceptance & ESS2 & Norm(mean) & ")
        for i in range(len(self.ref_quantiles)):
            latex_lines.append('%.1f' % self.ref_quantiles[i] + "-quantile")
            if i < len(self.ref_quantiles) - 1:
                latex_lines.append(" & ")
        latex_lines.append("\\\\")
        lines.append("".join(latex_lines))
        
        latex_lines = []
        latex_lines.append(self.experiments[0].mcmc_chain.mcmc_sampler.__class__.__name__)
        latex_lines.append('$%.3f' % mean(acceptance_rates) + " \pm " + '%.3f$' % (std(acceptance_rates)/sqrt_num_trials))
        latex_lines.append('$%.3f' % mean(norm_of_means) + " \pm " + '%.3f$' % (std(norm_of_means)/sqrt_num_trials))
        for i in range(len(self.ref_quantiles)):
            latex_lines.append('$%.3f' % mean_quantiles[i] + " \pm " + '%.3f$' % (std_quantiles[i]/sqrt_num_trials))
        
        
        lines.append(" & ".join(latex_lines) + "\\\\")
        
        return lines

    def __str__(self):
        s = self.__class__.__name__ + "=["
        s += "ref_quantiles=" + str(self.ref_quantiles)
        s += ", " + ExperimentAggregator.__str__(self)
        s += "]"
        return s
