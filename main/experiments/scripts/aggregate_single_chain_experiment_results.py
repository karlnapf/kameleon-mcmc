from main.experiments.SingleChainExperimentAggregator import \
    SingleChainExperimentAggregator
import os
import sys

if __name__ == '__main__':
    if len(sys.argv)!=4:
        print "usage:", str(sys.argv[0]).split(os.sep)[-1], "<idx_from> <idx_to> <folder_base>"
        print "example:"
        print "python " + str(sys.argv[0]).split(os.sep)[-1] + " 0 2 /nfs/home1/ucabhst/mcmc_hammer_experiments/MCMCHammerWindow_Ring_"
        exit()
        
    a=int(str(sys.argv[1]))
    b=int(str(sys.argv[2]))
    folder_base=str(sys.argv[3])
    output_filename=folder_base + os.sep + "results_" + str(a) + "_" + str(b) + ".txt"
    indices=range(a, b+1)
    lines=[]
    
    lines.append("aggregating experiments " + str(indices) +  " of " + folder_base)
    print lines[-1]
    
    folders = [folder_base + str(i) + os.sep for i in indices]
    ag = SingleChainExperimentAggregator(folders)
    results=ag.aggregate()
    print os.linesep.join(results)
    lines+=results
    
    # save to result file
    f=open(output_filename, 'w')
    f.write(os.linesep.join(lines))
    f.close()
