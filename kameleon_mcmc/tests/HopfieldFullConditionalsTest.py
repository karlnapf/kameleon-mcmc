"""
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

Written (W) 2013 Heiko Strathmann
Written (W) 2013 Dino Sejdinovic
"""

from numpy.random import rand, randn
from numpy import zeros, fill_diagonal, asarray, arange, reshape, mean
from matplotlib.pyplot import show, plot, hist
from kameleon_mcmc.distribution.Hopfield import Hopfield
from kameleon_mcmc.distribution.full_conditionals.HopfieldFullConditionals import HopfieldFullConditionals


def main():
    d=20
    b=randn(d)
    V=randn(d,d)
    W=V+V.T
    fill_diagonal(W,zeros(d))
    full_hopfield=Hopfield(W,b)
    schedule="in_turns"
    current_stateA=list(rand(d)<0.5)
    current_stateB=list(rand(d)<0.5)
    conditionalsA=HopfieldFullConditionals(full_hopfield, \
                                          current_stateA, \
                                          schedule, \
                                          index_block=arange(d) )
    conditionalsB=HopfieldFullConditionals(full_hopfield, \
                                          current_stateB, \
                                          schedule, \
                                          index_block=arange(d) )
    n=500000
    burnin=1000
    thin=100
    X=reshape(zeros(n*d)==1,(n,d))
    Y=reshape(zeros(n*d)==1,(n,d))
    for ii in arange(n):
        #print 'now sampling index: ' + str(conditionals.current_idx)
        X[ii]=conditionalsA.sample(1)
        Y[ii]=conditionalsB.sample(1)
        #print X[ii]
    hist(full_hopfield.log_pdf(X[arange(burnin,n,thin)]))
    print mean(X[arange(burnin,n,thin)],0)
    print mean(Y[arange(burnin,n,thin)],0)
    show()
    
    
    
    
    
    
main()