"""
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

Written (W) 2013 Heiko Strathmann
Written (W) 2013 Dino Sejdinovic
"""

from abc import abstractmethod
class Output(object):
    def __init__(self):
        pass

    @abstractmethod
    def update(self, mcmc_chain, step_output):
        raise NotImplementedError()
    
    @abstractmethod
    def prepare(self):
        raise NotImplementedError()
    
