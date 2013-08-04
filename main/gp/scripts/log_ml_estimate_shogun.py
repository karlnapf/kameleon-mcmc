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
from main.mcmc.samplers.KameleonWindowLearnScale import KameleonWindowLearnScale
from matplotlib.pyplot import figure, plot
from numpy.core.function_base import linspace
from numpy.ma.core import sin, asarray, zeros
from numpy.random import randn

from main.distribution.Distribution import Distribution
from numpy.ma.core import shape, zeros, exp
from shogun.Features import BinaryLabels, RealFeatures
from shogun.GaussianProcess import LaplacianInferenceMethod, LogitLikelihood, \
    ZeroMean
from shogun.Kernel import GaussianKernel

# create example data
x_max=20
n_train=50
noise_levels=linspace(1,0,n_train)*1
amplitude=2

X=linspace(0,x_max,n_train)
omegas=linspace(0,2,n_train)*0.1
Y_model=sin(X*omegas)*amplitude
Y=Y_model.copy()

labels=asarray([1.0 if i%2==0 else -1.0 for i in range(n_train)])
idx_a=labels>=0
idx_b=labels<0

y_offset=0.4
Y[idx_a]-=randn(sum(idx_a))*noise_levels[idx_a]+ y_offset
Y[idx_b]+=randn(sum(idx_b))*noise_levels[idx_b]+ y_offset

figure(figsize=(18,2))
plot(X[idx_a], Y[idx_a],"ro")
plot(X[idx_b], Y[idx_b],"bo")
plot(X,Y_model)

data=asarray(zip(X,Y))


#  create a GP with Laplace approximation
theta_prior=Gaussian()

features=RealFeatures(data.T)
kernel=GaussianKernel(10, 1)
mean=ZeroMean()
labels=BinaryLabels(labels)
model=LogitLikelihood()
inf=LaplacianInferenceMethod(kernel, features, mean, labels, model)

inf.set_scale(exp(0))
kernel.set_width(exp(0))

print -inf.get_negative_marginal_likelihood()
print inf.get_log_ml_estimate(10000, 1e-5)
