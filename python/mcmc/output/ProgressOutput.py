from classes.mcmc.output.Output import Output

class ProgressOutput(Output):
    def __init__(self):
        pass
    
    def update(self, mcmc_params, proposal, samples, log_liks, Q):
        it = len(samples)
        if (it % round(mcmc_params.num_iterations / 10)) == 0:
            print int(round(float(it) / mcmc_params.num_iterations * 100)), \
            "\tpercent done"
    
    def prepare(self, distribution):
        pass
