class Distribution(object):
    def __init__(self, dimension):
        self.dimension = dimension
    
    def sampler(self, n=1):
        raise NotImplementedError()
    
    def log_pdf(self, X):
        raise NotImplementedError()
    
