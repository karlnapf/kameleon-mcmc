"""
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

Written (W) 2013 Heiko Strathmann
Written (W) 2013 Dino Sejdinovic
"""

from abc import abstractmethod
from kameleon_mcmc.tools.GitTools import GitTools
from os import makedirs
from pickle import dump, load
import os
import sys

class Experiment(object):
    dispatcher_filename=os.path.abspath(__file__)
    
    erperiment_output_filename="experiment_output.bin"
    experiment_output_folder="output"
    gitversion_filename="gitversion.txt"
    parameters_filename="parameters.txt"
    
    def __init__(self, experiment_dir="", name=""):
        self.experiment_dir=experiment_dir
        self.name=name
        
        # create new folder for experiment
        folder_base=experiment_dir + name + "_"
        num = 0;
        while os.path.exists(folder_base + str(num) + os.sep):
            num += 1
        
        self.foldername = folder_base + str(num) + os.sep
        makedirs(self.foldername)
            
        # save git version and branch to file
        f = open(self.foldername + Experiment.gitversion_filename, "w")
        f.write(GitTools.get_hash() + os.linesep)
        f.write(GitTools.get_branch() + os.linesep)
        f.close()
        
        # save some infos to a file
        f = open(self.foldername + Experiment.parameters_filename, "w")
        f.write(str(self))
        f.close()
        
    @abstractmethod
    def __run_experiment__(self):
        raise NotImplementedError()
    
    def run(self):
        print "starting experiment", self.name
        self.__run_experiment__()
        print "saving experiment", self.name
        self.save_results()
        print "experiment", self.name, "done"
    
    def save_results(self):
        output_folder = self.foldername + Experiment.experiment_output_folder + os.sep
        makedirs(output_folder)
        f = open(output_folder + Experiment.erperiment_output_filename, "w")
        dump(self, f)
        f.close()
        
    def __str__(self):
        s=self.__class__.__name__+ "=["
        s += "experiment_dir="+ str(self.experiment_dir)
        s += ", name="+ str(self.name)
        s += ", foldername="+ str(self.foldername)
        s += "]"
        return s
    
if __name__ == '__main__':
    if len(sys.argv)!=2:
        print "usage:", str(sys.argv[0]).split(os.sep)[-1], "<Experiment instance pickle filename>"
        print "example:"
        print "python " + str(sys.argv[0]).split(os.sep)[-1] + " /nfs/home1/ucabhst/mcmc_hammer_experiments/KameleonWindow_Ring_0/experiment_instance.bin"
        exit()
        
    filename=str(sys.argv[1])
    print "running experiment file", filename
    
    try:
        f=open(filename, 'r')
        experiment=load(f)
        f.close()
        experiment.run()
    except IOError:
        print "Could not open file due to IOError"