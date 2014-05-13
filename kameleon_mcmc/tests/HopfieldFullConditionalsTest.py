"""
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

Written (W) 2013 Heiko Strathmann
Written (W) 2013 Dino Sejdinovic
"""

from numpy.random import randn
from numpy import zeros, fill_diagonal, asarray, arange, reshape
from kameleon_mcmc.distribution.Hopfield import Hopfield
from kameleon_mcmc.distribution.HopfieldFullConditionals import HopfieldFullConditionals

def main():
    d=10
    b=randn(d)
    V=randn(d,d)
    W=V+V.T
    fill_diagonal(W,zeros(d))
    full_hopfield=Hopfield(W,b)
    schedule="in_turns"
    current_state=asarray(zeros(d), dtype=bool)
    conditionals=HopfieldFullConditionals(full_hopfield, \
                                          current_state, \
                                          schedule, \
                                          index_block=arange(d) )
    for ii in arange(50):
        print conditionals.current_idx
        X=conditionals.sample(1)
        print X
        print full_hopfield.log_pdf(reshape(X,(1,d)))
    
    
    
    
    
    
main()