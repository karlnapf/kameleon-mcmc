from pickle import dump
from popen2 import popen2
import os
import time

class ClusterTools(object):
    cluster_output_filename="cluster_output.txt"
    cluster_error_filename="cluster_error.txt"
    
    @staticmethod
    def submit_experiments(experiment_list, cluster_command="run_experiment.py"):
        
        filenames=[]
        for experiment in experiment_list:
            filename=experiment.foldername+"experiment_instance.bin"
            filenames.append(filename)
            f=open(filename, 'w')
            dump(experiment, f)
            f.close()
            
        for i in range(len(filenames)):
            command="nice -n 10 python " + cluster_command + " " + filenames[i]
            
            job_name = filenames[i].split(os.sep)[-2].split(".")[0]
            walltime = "walltime=2:00:00"
            processors = "nodes=1:ppn=1"
            memory = "pmem=1gb"
            workdir = experiment.foldername
            output=experiment_list[i].foldername + ClusterTools.cluster_output_filename
            error=experiment_list[i].foldername + ClusterTools.cluster_error_filename
            
            job_string = """
            #PBS -S /bin/bash
            #PBS -N %s
            #PBS -l %s
            #PBS -l %s
            #PBS -l %s
            #PBS -o %s
            #PBS -e %s
            export PATH=/nfs/home1/ucabjga/opt/epd/bin:$PATH
            export PYTHONPATH=/nfs/home1/ucabhst/mcmc-hammer
            cd %s
            %s""" % (job_name, walltime, processors, memory, output, error, workdir, command)
        
            # send job_string to qsub
            outpipe, inpipe = popen2('qsub')
            print job_string
            inpipe.write(job_string)
            inpipe.close()
            
            print outpipe.read()
            time.sleep(0.1)