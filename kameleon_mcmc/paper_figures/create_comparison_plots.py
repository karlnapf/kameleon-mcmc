"""
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

Written (W) 2013 Heiko Strathmann
Written (W) 2013 Dino Sejdinovic
"""

from matplotlib.pyplot import bar, legend, figure, savefig, show, errorbar, ylim
from numpy import asarray
from numpy import arange, zeros
import os
import sys

import latex_plot_init


if __name__ == '__main__':
    """
    Takes a set out output files from SingleChainExperimentAggregator and plots
    comparisons of the quantile errors and other statistics
    
    """
    if len(sys.argv) <= 3:
        print "usage:", str(sys.argv[0]).split(os.sep)[-1], "<outfilename> <directory> <filename(s)>"
        print "example:"
        print "python " + str(sys.argv[0]).split(os.sep)[-1] + " " + \
              "plots/banana_comparison_plot.eps" + \
              "/home/heiko/hammer_plots_before_rebuttal/ " + \
              "AdaptiveMetropolisLearnScale_Banana_results_0_99.txt " + \
              "AdaptiveMetropolis_Banana_results_0_99.txt " + \
              "KameleonWindowLearnScale_Banana_results_0_99.txt " + \
              "StandardMetropolis_Banana_results_0_99.txt"
        exit()
        
    outfile = str(sys.argv[1])
    folder = str(sys.argv[2])
    
    ref_quantiles = arange(0.1, 1.0, 0.1)
    
    # iterate over files and extract statistics for every file
    quantile_errors = []
    quantile_error_stds = []
    acceptance_rates = []
    acceptance_rate_stds = []
    norms_of_means = []
    norm_of_mean_stds = []
    filenames = []
    for filename in sys.argv[3:]:
        filenames.append(filename)
        print folder + filename
        f = open(folder + filename, "r")
        lines = [line.strip() for line in f.readlines()]
        f.close()
        
        quantile_errors.append(zeros(len(ref_quantiles)))
        quantile_error_stds.append(zeros(len(ref_quantiles)))
        quantile_index = lines.index("quantiles:") + 1
        for i in range(len(ref_quantiles)):
            quantile_line = lines[quantile_index]
            quantile_errors[-1][i] = float(quantile_line.split("+-")[0])
            quantile_error_stds[-1][i] = float(quantile_line.split("+-")[1])
            quantile_index += 1
        #quantile_errors[-1] = abs(quantile_errors[-1] - ref_quantiles)+

        acceptance_line = lines[lines.index("acceptance rate:") + 1]
        acceptance_rates.append(float(acceptance_line.split("+-")[0]))
        acceptance_rate_stds.append(float(acceptance_line.split("+-")[1]))
        
        mean_line = lines[lines.index("norm of means:") + 1]
        norms_of_means.append(float(mean_line.split("+-")[0]))
        norm_of_mean_stds.append(float(mean_line.split("+-")[1]))
    
    # combine quantiles and acceptance rates, means
    mean_scaler = 1.0 / 10
    acceptace_scaler = 1.0
    quantile_scaler = 10
    sqrt_num_trials=10
    overal_length = len(quantile_errors[0]) + 2
    to_plot = zeros((len(filenames), overal_length))
    error_bars = zeros((len(filenames), overal_length))
    for i in range(len(to_plot)):
        to_plot[i, 0] = acceptance_rates[i] * acceptace_scaler
        to_plot[i, 1] = norms_of_means[i] * mean_scaler
        to_plot[i, 2:] = quantile_errors[i] * quantile_scaler
        error_bars[i, 0] = acceptance_rate_stds[i] * acceptace_scaler
        error_bars[i, 1] = norm_of_mean_stds[i] * mean_scaler
        error_bars[i, 2:] = quantile_error_stds[i] * quantile_scaler
        
    # plotting things
    bar_width = 0.15
    colours = ['blue', 'red', 'yellow', 'green']
    bar_positions = arange(len(to_plot[0]))
    bar_positions[2:] += 1
    bar_positions[1:] += 1
    bar_positions += 1
    x_tick_marks = ["Accept $\in [0,1]$", \
                    "$||\hat{\mathbb{E}}[X]||/10$"] + \
                    [str(ref_quantile) for ref_quantile in ref_quantiles]
    
    # do barplots with errorbars
    fig = figure(figsize=(6.5, 1.2))
    ax = fig.gca()
    ax.yaxis.grid(True)
    bars = []
    for i in range(len(to_plot)):
        # bars in groups
        bars.append(bar(bar_positions + i * bar_width, to_plot[i], \
            width=bar_width, color=colours[i]))
        
        # error-bars in the middle of bars (scalling from 1 std dev to 1.28 aka 80%)
        errorbar(bar_positions + (i + 0.5) * bar_width, to_plot[i], yerr=error_bars[i]*1.28, \
                 fmt=' ', color="black", capsize=1, linewidth=0.5)
        
    ylim(0, .6)
    ax.set_xticks(bar_positions + 2 * bar_width)
    ax.set_xticklabels(x_tick_marks)
    
    filename_to_plot_name = {"StandardMetropolis_Banana_results_0_99.txt":"SM", \
                           "AdaptiveMetropolis_Banana_results_0_99.txt":"AM-FS", \
                           "AdaptiveMetropolisLearnScale_Banana_results_0_99.txt":"AM-LS", \
                           "KameleonWindowLearnScale_Banana_results_0_99.txt":"KAMH-LS"}
    
    def filename_to_plot_name(filename):
        if filename.find("StandardMetropolis") is not -1:
            return "SM"
        elif filename.find("AdaptiveMetropolisLearnScale") is not -1:
            return "AM-LS"
        elif filename.find("AdaptiveMetropolis") is not -1:
            return "AM-FS"
        elif filename.find("KameleonWindowLearnScale") is not -1:
            return "KAMH-LS"
        else:
            print "cannot process filename"
            exit()
    
    
    l = legend(bars, tuple([filename_to_plot_name(filename) for filename in filenames]), \
           loc="upper center", ncol=4)
    l.draggable(True)
    savefig(outfile, bbox_inches='tight')
    
