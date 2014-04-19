"""
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

Written (W) 2013 Heiko Strathmann
Written (W) 2013 Dino Sejdinovic
"""

from kameleon_mcmc.experiments.SingleChainExperimentAggregator import \
    SingleChainExperimentAggregator
import os
import sys

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print "usage:", str(sys.argv[0]).split(os.sep)[-1], "<idx_from> <idx_to> <folder_base>"
        print "example:"
        print "python " + str(sys.argv[0]).split(os.sep)[-1] + " 0 2 /nfs/data3/ucabhst/kameleon_experiments/KameleonWindow_Ring_"
        exit()
        
    a = int(str(sys.argv[1]))
    b = int(str(sys.argv[2]))
    folder_base = str(sys.argv[3])
    
    temp = folder_base.split(os.sep)
    output_filename = os.sep.join(temp[:-1]) + os.sep + temp[-1] + "results_" + str(a) + "_" + str(b) + ".txt"
    
    indices = range(a, b + 1)
    lines = []
    
    lines.append("aggregating experiments " + str(indices) + " of " + folder_base)
    print lines[-1]
    
    folders = [folder_base + str(i) + os.sep for i in indices]
    ag = SingleChainExperimentAggregator(folders)
    results = ag.aggregate()
    print os.linesep.join(results)
    lines += results
    
    # save to result file
    f = open(output_filename, 'w')
    f.write(os.linesep.join(lines))
    f.close()
