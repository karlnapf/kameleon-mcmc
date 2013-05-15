class Output(object):
    def __init__(self):
        pass

    def update(self, mcmc_params, proposal, samples, log_liks, Q):
        raise NotImplementedError()
    
    def prepare(self, distribution):
        raise NotImplementedError()
    
