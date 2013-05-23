from pickle import dump
from popen2 import popen2
import os
import time

class ClusterTools(object):
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
            walltime = "walltime=99:00:00"
            processors = "nodes=1:ppn=1"
            memory = "pmem=100mb"
            workdir = experiment.folder_prefix
            mail="heiko.strathmann@gmail.com"
            output=experiment_list[i].foldername + "cluster_output.txt"
            error=experiment_list[i].foldername + "cluster_error.txt"
            
            job_string = """
            #PBS -S /bin/bash
            #PBS -N %s
            #PBS -l %s
            #PBS -l %s
            #PBS -l %s
            #PBS -M %s
            #PBS -o %s
            #PBS -e %s
            #PBS -m abe  # (a = abort, b = begin, e = end)
            cd %s
            %s""" % (job_name, walltime, processors, memory, mail, output, error, workdir, command)
        
            # send job_string to qsub
            outpipe, inpipe = popen2('qsub')
            print job_string
            inpipe.write(job_string)
            inpipe.close()
            
            print outpipe.read()
            time.sleep(0.1)