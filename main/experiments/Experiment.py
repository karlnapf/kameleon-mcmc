from abc import abstractmethod
from main.tools.GitTools import GitTools
from os import makedirs
from pickle import dump
import os

class Experiment(object):
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