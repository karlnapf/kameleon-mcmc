This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

Written (W) 2013 Heiko Strathmann and Dino Sejdinovic

Code for Adaptive Metropolis-Hastings.

See main.examples for demonstrations how to run the sampler on example distributions.
All experiments in the paper can be reproduced with the scripts in experiments.scripts.
All figures in the paper can be reproduced with scripts in main.paper_figures.

The main.gp module contains code for sampling GP classification based distributions
over hyperparameters, marginalised over the GP latent variables. The resulting
marginal likelihood is not available in closed for and there has to be estimated.
The Shogun machine learning toolbox is used for this.
See http://shogun-toolbox.org/
