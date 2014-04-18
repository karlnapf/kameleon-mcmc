"""
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

Written (W) 2013 Heiko Strathmann
"""

from glob import glob
from kameleon_mcmc.mcmc.output.Output import Output
from pickle import dump, load
import os

class StoreChainOutput(Output):
    
    instance_filename_base = "StoreChainOutput_iteration_"
    
    def __init__(self, folder, lag=1):
        Output.__init__(self)
        
        self.folder = folder
        self.lag = lag
        
    def update(self, mcmc_chain, step_output):
        i = mcmc_chain.iteration
        if i % self.lag == 0:
            filename = self.folder + os.sep + self.instance_filename_base + str(i)
            f = open(filename, 'wb')
            dump(mcmc_chain, f)
            f.close()
        
    def prepare(self):
        try:
            os.makedirs(self.folder)
        except OSError:
            pass
        
    def load_last_stored_chain(self):
        filenames = glob(self.folder + os.sep + self.instance_filename_base + "*")
        
        if len(filenames) is 0:
            return None
        
        max_number = 0
        max_filename = None
        for filename in filenames:
            idx = filename.find(self.instance_filename_base) + len(self.instance_filename_base)
            current = int(filename[idx:])
            if current > max_number:
                max_number = current
                max_filename = filename
        
        # if nothing was saved yet    
        if max_filename is None:
            return None
        
        f = open(max_filename)
        chain = load(f)
        f.close()
        
        return chain
    
