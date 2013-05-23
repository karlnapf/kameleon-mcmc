from posixpath import expanduser
import os
import sys
to_add=os.sep.join(os.path.abspath(os.path.dirname(sys.argv[0])).split(os.sep)[0:-3])
sys.path.append(to_add)

from main.experiments.SingleChainAggregator import SingleChainAggregator

if __name__ == '__main__':
    assert(len(sys.argv)==3)
    a=int(str(sys.argv[1]))
    b=int(str(sys.argv[2]))
    indices=range(a, b+1)
    print "aggregating experiments", indices
    
    experiment_dir = expanduser("~") + os.sep + "mcmc_hammer_experiments" + os.sep
    folders = [experiment_dir + "MCMCHammerWindow_Ring_" + str(i) + os.sep for i in indices]
    ag = SingleChainAggregator(folders)
    ag.load_raw_results()
    ag.post_process()
    