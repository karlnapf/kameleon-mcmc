#Code for Kernel Adaptive Metropolis-Hastings.

See http://jmlr.org/proceedings/papers/v32/sejdinovic14.html

Written (W) 2013-2014 Heiko Strathmann and Dino Sejdinovic

Build status:
[![Build Status](https://travis-ci.org/karlnapf/kameleon-mcmc.png)](https://travis-ci.org/karlnapf/kameleon-mcmc)

This software is licensed under a BSD license. See license.txt.

##Description
See kameleon_mcmc.examples for demonstrations how to run the sampler on example distributions.
All experiments in the paper can be reproduced with the scripts in experiments.scripts.
All figures in the paper can be reproduced with scripts in kameleon_mcmc.paper_figures.

The kameleon_mcmc.gp module contains code for sampling GP classification based distributions
over hyperparameters, marginalised over the GP latent variables. The resulting
marginal likelihood is not available in the closed form and has to be estimated.
The Shogun machine learning toolbox is used for this.
See http://shogun-toolbox.org/
