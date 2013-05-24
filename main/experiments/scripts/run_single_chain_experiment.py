from pickle import load
import sys

if len(sys.argv)!=2:
    print "usage:", str(sys.argv[0]), "<Experiment instance pickle filename>"
    print "example:"
    print "python run_single_chain_experiment.py ~/mcmc_hammer_experiments/MCMCHammerWindow_Ring_0/experiment_instance.bin"
    exit()
    
filename=str(sys.argv[1])
print "running experiment file", filename

f=open(filename, 'r')
experiment=load(f)
f.close()
experiment.run()