"""
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

Written (W) 2013 Heiko Strathmann
"""

from main.distribution.Gaussian import Gaussian
from main.gp.GaussianProcess import GaussianProcess
from main.gp.covariance.SquaredExponentialCovariance import \
    SquaredExponentialCovariance
from main.gp.inference.LaplaceApproximation import LaplaceApproximation
from main.gp.likelihood.LogitLikelihood import LogitLikelihood
from main.gp.mcmc.PseudoMarginalHyperparameterDistribution import \
    PseudoMarginalHyperparameterDistribution
from main.kernel.GaussianKernel import GaussianKernel
from main.mcmc.MCMCChain import MCMCChain
from main.mcmc.MCMCParams import MCMCParams
from main.mcmc.output.PlottingOutput import PlottingOutput
from main.mcmc.output.StatisticsOutput import StatisticsOutput
from main.mcmc.samplers.AdaptiveMetropolisLearnScale import \
    AdaptiveMetropolisLearnScale
from main.mcmc.samplers.KameleonWindowLearnScale import KameleonWindowLearnScale
from main.mcmc.samplers.StandardMetropolis import StandardMetropolis
from matplotlib.pyplot import figure, plot
from numpy.core.function_base import linspace
from numpy.lib.twodim_base import eye
from numpy.ma.core import sin, asarray, zeros, log, exp, sort, cos, ones
from numpy.random import rand, randn, randint
from numpy.random import seed
from scipy.constants.constants import pi

# cirlce
t=linspace(0,pi/2)
X_model=sin(t)
Y_model=cos(t)
#plot(X_model,Y_model)

# decision surface for sampled data
n_train=50
thetas=linspace(0,pi/2,n_train)
X=sin(thetas)
Y=cos(thetas)

# randomly select labels and distinguish data
offset=0.05
seed(1)
lab=randint(0,2,n_train)*2-1
idx_a=lab>0
idx_b=lab<0
X[idx_a]*=(1.+offset)
Y[idx_a]*=(1.+offset)
X[idx_b]*=(1.-offset)
Y[idx_b]*=(1.-offset)

#plot(X[idx_a],Y[idx_a], 'bo')
#plot(X[idx_b],Y[idx_b], 'ro')

data=asarray(zip(X,Y))

# create example data
x_max=15
n_train=100
noise_levels=linspace(1,0,n_train)*1
amplitude=2

X=sort(rand(n_train)**1)
X=X/X.max()*x_max
#X=linspace(0,x_max,n_train)
omegas=exp(linspace(0,.9,n_train))-1
Y_model=sin(X*omegas)*amplitude
Y=Y_model.copy()

lab=asarray([1.0 if i%2==0 else -1.0 for i in range(n_train)])
idx_a=lab>=0
idx_b=lab<0

y_offsets_a=linspace(1.,0.2,len(Y[idx_a]))
y_offsets_b=linspace(1.,0.2,len(Y[idx_a]))
Y[idx_a]-=randn(sum(idx_a))*noise_levels[idx_a]+y_offsets_a
Y[idx_b]+=randn(sum(idx_b))*noise_levels[idx_b]+y_offsets_b

figure(figsize=(18,2))
plot(X[idx_a], Y[idx_a],"ro")
plot(X[idx_b], Y[idx_b],"bo")
plot(X,Y_model)

data=asarray(zip(X,Y))

# prior on theta is set to something that corresponds to the ML" maximum
theta_prior=Gaussian(mu=2*ones(2), Sigma=eye(2)*5)
target=PseudoMarginalHyperparameterDistribution(data, lab, \
                                                n_importance=1000, prior=theta_prior, \
                                                ridge=1e-5)

kernel = GaussianKernel(sigma=5.0)
sampler=KameleonWindowLearnScale(target, kernel)
#sampler=AdaptiveMetropolisLearnScale(target)
#sampler=StandardMetropolis(target)
params = MCMCParams(start=2.0*ones(target.dimension), num_iterations=100000, burnin=20000)

chain=MCMCChain(sampler, params)
chain.append_mcmc_output(StatisticsOutput(print_from=0, lag=1))
chain.append_mcmc_output(PlottingOutput(plot_from=0))

chain.run()