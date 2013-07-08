"""
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

Written (W) 2013 Heiko Strathmann
Written (W) 2013 Dino Sejdinovic
"""

from numpy.ma.core import mean

class StatisticTools(object):
    
    @staticmethod
    def effective_sample_size(data, step_size=1) :
        """ Computes the effective sample size for the given 1D array of points.
        Original source taken from biopy
        Modified by Heiko Strathmann under GPLv3
        """
        num_samples = len(data)
        
        assert len(data) > 1
        
        maxLag = min(num_samples // 3, 1000)
        
        gamma_stat = [0, ] * maxLag
        
        var_stat = 0.0;
        
        normalised_data = data - data.mean()
        
        for lag in range(maxLag) :
            v1 = normalised_data[:num_samples - lag]
            v2 = normalised_data[lag:]
            gamma_stat[lag] = mean(v1*v2)
            
            if lag == 0 :
                var_stat = gamma_stat[0]
            elif lag % 2 == 0 :
                s = gamma_stat[lag - 1] + gamma_stat[lag]
                
                if s > 0 :
                    var_stat += 2.0 * s
                else :
                    break
            
        # auto correlation time
        act = step_size * var_stat / gamma_stat[0]
        
        # effective sample size
        ess = (step_size * num_samples) / act
        
        return ess
