from main.mcmc.output.Output import Output

class ProgressOutput(Output):
    def __init__(self):
        Output.__init__(self)
    
    def update(self, mcmc_params, proposal_object, samples, log_liks, Q):
        it = len(samples)
        if (it % round(mcmc_params.num_iterations / 10)) == 0:
            print int(round(float(it) / mcmc_params.num_iterations * 100)), \
            "\tpercent done"
    
    def prepare(self):
        pass
