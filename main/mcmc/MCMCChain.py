from numpy.ma.core import zeros

class MCMCChain(object):
    def __init__(self, mcmc_sampler, mcmc_params):
        self.mcmc_sampler = mcmc_sampler
        self.mcmc_params = mcmc_params
        self.mcmc_outputs = []
        self.is_initialised = False
    
    def init(self):
        # fields for the chain
        num_iterations = self.mcmc_params.num_iterations
        self.samples = zeros((num_iterations, self.mcmc_sampler.distribution.dimension))
        self.ratios = zeros(num_iterations)
        self.log_liks = zeros(num_iterations)
        self.accepteds = zeros(num_iterations)
        
        # state of the chain
        self.iteration = 0
        
        # init output instances
        for out in self.mcmc_outputs:
            out.prepare()
        
        # init sampler with starting point
        self.mcmc_sampler.init(self.mcmc_params.start.copy())
        self.is_initialised = True
        
    
    def has_finished(self):
        return self.iteration >= self.mcmc_params.num_iterations - 1
    
    def append_mcmc_output(self, output):
        self.mcmc_outputs.append(output)
    
    def run(self):
        if not self.is_initialised:
            self.init()
        
        # run chain
        while self.iteration < self.mcmc_params.num_iterations:
            i = self.iteration
            # store old proposal for outputs
            """
            there is a problem here: as Q is first
            initialised in the step() method,
            on the first call Q_old is always None
            -dino
            """
            Q_old = self.mcmc_sampler.Q
            
            # mcmc step
            sample, proposal, accepted, log_lik, ratio = self.mcmc_sampler.step()
            
            # output updated state
            for out in self.mcmc_outputs:
                out.adapt(self.mcmc_params, proposal, self.samples[0:i], \
                           self.log_liks[0:i], Q_old)

            # collect results
            self.samples[i, :] = sample
            self.ratios[i] = ratio
            self.accepteds[i] = accepted
            self.log_liks[i] = log_lik
            
            # adapt sampler
            self.mcmc_sampler.adapt(self)
            
            # adapt chain state
            self.iteration += 1
