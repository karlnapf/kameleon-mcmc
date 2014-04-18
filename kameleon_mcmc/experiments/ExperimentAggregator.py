"""
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

Written (W) 2013 Heiko Strathmann
Written (W) 2013 Dino Sejdinovic
"""

from abc import abstractmethod
from kameleon_mcmc.experiments.ClusterTools import ClusterTools
from kameleon_mcmc.experiments.Experiment import Experiment
from pickle import load
from sets import Set
import os

class ExperimentAggregator(object):
    def __init__(self, folders):
        assert(len(folders) > 0)
        
        self.experiments = []
        
        # load first git version
        if not os.path.exists(folders[0]):
            print folders[0]
            assert(os.path.exists(folders[0]))

        f = open(folders[0] + Experiment.gitversion_filename)
        ref_githash = f.readline().strip()
        ref_gitbranch = f.readline().strip()
        f.close()
        
        to_ignore = Set()
        
        # compare to all others
        for i in range(len(folders)):
            # assert that all experiments ran under the same git version
            if not os.path.exists(folders[i]):
                print "folder", folders[i], "does not exist, ignoring"
                to_ignore.add(i)
                continue
            
            f = open(folders[i] + Experiment.gitversion_filename)
            githash = f.readline().strip()
            gitbranch = f.readline().strip()
            f.close()
            
            # assert that all git hashs are equal
            if not githash == ref_githash:
                print "git hash", githash, "is not equal to reference git hash", ref_githash
            if not gitbranch == ref_gitbranch:
                print "git branch", gitbranch, "is not equal to reference git branch",
            
            #assert(githash == ref_githash)
            #assert(gitbranch == ref_gitbranch)
            
            # this might be false if script was altered after experiment, comment out then
#            if githash != GitTools.get_hash():
#                print "git version in folder", folders[i], "is", githash, "while " \
#                      "git version in folder", folders[0], "is", ref_githash
#                 
#            if gitbranch != GitTools.get_branch():
#                print "git branch in folder", folders[i], "is", gitbranch, "while " \
#                      "git branch in folder", folders[0], "is", ref_gitbranch
                      
        # remove non-existing folders
        folders_cleaned=[]
        for i in range(len(folders)):
            if i not in to_ignore:
                folders_cleaned.append(folders[i])
            
        self.folders=folders_cleaned
    
    def load_raw_results(self):
        self.experiments = []
        for i in range(len(self.folders)):
            filename = self.folders[i] + Experiment.experiment_output_folder + \
                     os.sep + Experiment.erperiment_output_filename
            
            try:
                f = open(filename , "r")
                self.experiments.append(load(f))
                f.close()
            except IOError:
                print "skipping", filename, "due to IOError"
                errorfilename=self.folders[i] + ClusterTools.cluster_error_filename
                try:
                    ef = open(errorfilename)
                    lines=ef.readlines()
                    print "cluster error output"
                    print lines, "\n\n"
                except IOError:
                    print "could not find cluster error file", errorfilename, "due to IOError"
        
        print "loaded", len(self.experiments), "experiments"
                
    def aggregate(self):
        """
        Loads and processes results and returns a list of strings from private
        method __process_results__
        """
        self.load_raw_results()
        return self.__process_results__()
    
    @abstractmethod
    def __process_results__(self):
        """
        Returns a list of strings with results
        """
        raise NotImplementedError()
    
    def __str__(self):
        s=self.__class__.__name__+ "=["
        s += "folders="+ str(self.folders)
        s += ", experiments="+ str(self.experiments)
        s += "]"
        return s
