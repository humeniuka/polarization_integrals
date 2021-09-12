#!/usr/bin/env python
"""
compare integrals for diffuse orbitals (small exponents) with exact integrals.
"""
import unittest
import numpy as np

# fast C++ implementation
from polarization_integrals import PolarizationIntegral

try:
    from polarization_ints_numerical import polarization_integral as polarization_integral_numerical, overlap_integral
except ImportError as err:
    print("numerical integrals not available")
    exit(-1)
    
# increase resolution of integration grids
from becke import settings
settings.radial_grid_factor = 3  # increase number of radial points by factor 3
settings.lebedev_order = 23      # angular Lebedev grid of order 23


class TestDiffuseOrbitals(unittest.TestCase):
    def test_diffuse_case_1(self):
        """
        compare polarization integrals for the case `k=2*j` with the reference implementation
        The exponents beta_i and beta_j are scanned from 1.0e-5 to 0.1
        """
        # make random numbers reproducible
        np.random.seed(0)

        # centers of Gaussian orbitals        
        xi,yi,zi = 2.0 * 2.0*np.array(np.random.rand(3)-0.5)
        xj,yj,zj = 2.0 * 2.0*np.array(np.random.rand(3)-0.5)
        # angular momenta
        nxi,nyi,nzi = 1,0,0
        li = nxi+nyi+nzi
        nxj,nyj,nzj = 2,0,0
        lj = nxj+nyj+nzj
        # cutoff function
        alpha = 4.0
        q = 2
        
        # powers of polarization operator
        k = 4
        mx,my,mz = 0,0,0

        # vary exponents
        betas = np.linspace(0.00001, 0.1, 100)

        absolute_errors = np.zeros_like(betas)
        relative_errors = np.zeros_like(betas)
        for ib,beta in enumerate(betas):
            # exponents of radial parts of Gaussians
            beta_i = beta
            beta_j = beta

            # normalization constants for Gaussian orbitals
            norm_i = np.sqrt( overlap_integral(xi,yi,zi, nxi,nyi,nzi, beta_i,
                                               xi,yi,zi+0.000001, nxi,nyi,nzi, beta_i) )
            norm_j = np.sqrt( overlap_integral(xj,yj,zj, nxj,nyj,nzj, beta_j,
                                               xj,yj,zj+0.000001, nxj,nyj,nzj, beta_j) )

            #print(f"norm_i= {norm_i}  norm_j= {norm_j}")
            label=f"k={k} mx={mx} my={my} mz={mz} , q={q} , nxi={nxi} nyi={nyi} nzi={nzi} , nxj={nxj} nyj={nyj} nzj={nzj} beta={beta:e}"
            
            # prepare for integrals between shells with angular momenta li and lj
            pol = PolarizationIntegral(xi,yi,zi, li, beta_i,
                                       xj,yj,zj, lj, beta_j,
                                       k, mx,my,mz,
                                       alpha, q)

            # fast C++ implementation
            pint = pol.compute_pair(nxi,nyi,nzi,
                                    nxj,nyj,nzj)

            # slow numerical integrals
            pint_numerical = polarization_integral_numerical(xi,yi,zi, nxi,nyi,nzi, beta_i,
                                                             xj,yj,zj, nxj,nyj,nzj, beta_j,
                                                             k, mx,my,mz,
                                                             alpha, q)

            # integrals for normalized Gaussian orbitals
            pint /= norm_i * norm_j
            pint_numerical /= norm_i * norm_j
            
            print(label + f"  integral= {pint:+12.7f}  numerical= {pint_numerical:+12.7f}")
            absolute_errors[ib] = abs(pint - pint_numerical)
            relative_errors[ib] = abs(pint - pint_numerical)/abs(pint_numerical)
        
            # numerical accuracy is quite low
            self.assertTrue( (absolute_errors[ib] < 1.0e-3) or (relative_errors[ib] < 1.0e-3) )
            
if __name__ == '__main__':
    unittest.main()
