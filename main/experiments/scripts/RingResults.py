from main.experiments.SingleChainAggregator import SingleChainAggregator
from posixpath import expanduser
import os

if __name__ == '__main__':
    experiment_dir = expanduser("~") + os.sep + "mcmc_hammer_experiments" + os.sep
    folders = [experiment_dir + "MCMCHammerWindow_Ring_" + str(i) + os.sep for i in range(1)]
    ag = SingleChainAggregator(folders)
    ag.load_raw_results()
    ag.post_process()
    