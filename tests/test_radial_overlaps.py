#!/usr/bin/env python
"""
compare analytical and numerical integrals between spherically symmetric Gaussian densities
which serve as upper bounds for the polarization integrals 
"""
import numpy as np
import unittest

from radial_overlap import radial_overlap_numerical, radial_overlap_analytical

class TestRadialOverlap(unittest.TestCase):
    def test_radial_overlaps(self, num_samples=10):
        # Check integrals up to f-functions for Gaussians with random parameters
        lmax = 3
        for li in range(0, lmax+1):
            for lj in range(li, lmax+1):
                for n in range(0, num_samples):
                    xi, yi, zi,  xj, yj, zj = 1.0*2.0*(np.random.rand(6)-0.5)
                    beta_i, beta_j = 0.1 + 3.0*np.random.rand(2)
                    # analytical integrals
                    olap_ana = radial_overlap_analytical(xi,yi,zi, li, beta_i,
                                                         xj,yj,zj, lj, beta_j)
                    # numerical integrals
                    olap_num = radial_overlap_numerical(xi,yi,zi, li, beta_i,
                                                        xj,yj,zj, lj, beta_j)
                    #print(f"{li} {lj}   {olap_ana:e} ?= {olap_num:e}")
                    relative_error = abs(olap_ana - olap_num)/abs(olap_num)
                    absolute_error = abs(olap_ana - olap_num)
                    self.assertTrue( (absolute_error < 1.0e-4) or (relative_error < 1.0e-4) )

if __name__ == "__main__":
    # make random numbers reproducible
    np.random.seed(0)
    
    unittest.main()
