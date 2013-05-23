import os
import sys
to_add=os.sep.join(os.path.abspath(os.path.dirname(sys.argv[0])).split(os.sep)[0:-3])
sys.path.append(to_add)

from main.experiments.SingleChainAggregator import SingleChainAggregator

if __name__ == '__main__':
    if len(sys.argv)!=4:
        print "usage:", str(sys.argv[0]), "<idx_from> <idx_to> <folder_base>"
        print "example:"
        print "python RingResults.py 0 99 ~/mcmc_hammer_experiments/MCMCHammerWindow_Ring_"
        exit()
        
    a=int(str(sys.argv[1]))
    b=int(str(sys.argv[2]))
    folder_base=str(sys.argv[3])
    indices=range(a, b+1)
    print "aggregating experiments", indices, "of", folder_base
    
    folders = [folder_base + str(i) + os.sep for i in indices]
    ag = SingleChainAggregator(folders)
    ag.load_raw_results()
    ag.post_process()
    