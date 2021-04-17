#!/usr/bin/env python
"""
compare polarization integrals from C++ implementation with numerical integrals
"""
import unittest
import numpy as np

# fast C++ implementation
from polarization_integrals import PolarizationIntegral
# numerical integral using Becke's multicenter grids
from polarization_ints_numerical import polarization_integral as polarization_integral_numerical

# increase resolution of integration grids
from becke import settings
settings.radial_grid_factor = 3  # increase number of radial points by factor 3
settings.lebedev_order = 23      # angular Lebedev grid of order 23

def partition3(l):
    """
    enumerate all partitions of the integer l into 3 integers nx, ny, nz
    such that nx+ny+nz = l
    """
    for nx in range(0, l+1):
        for ny in range(0, l-nx+1):
            nz = l-nx-ny
            yield (nx,ny,nz)

def kappa(n):
    if (n % 2 == 0):
        return n/2
    else:
        return (n+1)/2

    
class TestPolarizationIntegrals(unittest.TestCase):
    verbosity = 0
    def test_analytical_vs_numerical_small_b(self):
        """test polarization integrals for basis centers on opposite sides of the origin (b=0)"""
        # cutoff function
        alpha = 50.0

        n_samples = 5
        for i in range(0, n_samples):
            if self.verbosity > 0:
                print(f"Random sample {i}")
            # First center is chosen randomly
            xi,yi,zi = 2.0 * 2.0*np.array(np.random.rand(3)-0.5)
            # exponents of radial parts of Gaussians
            beta_i, beta_j = np.random.rand(2)+0.1
            # The second center is chosen such that
            #  b = |beta_i * ri + beta_j * rj| = eps, such that x = b^2 / alpha << 1
            for eps in np.linspace(0.0, 5.0, 10):
                xj,yj,zj = - beta_i/beta_j * np.array([xi,yi,zi]) + eps * 2.0*(np.random.rand(3)-0.5)

                self._compare_all_integrals(xi,yi,zi, beta_i, xj,yj,zj, beta_j, alpha)

    def test_analytical_vs_numerical_large_b(self):
        """test polarization integrals for basis centers at random locations"""
        # cutoff function
        alpha = 50.0

        n_samples = 5
        for i in range(0, n_samples):
            if self.verbosity > 0:
                print(f"Random sample {i}")
            # Both centers are chosen randomly
            xi,yi,zi = 2.0*np.array(np.random.rand(3)-0.5)
            xj,yj,zj = 2.0*np.array(np.random.rand(3)-0.5)
            # exponents of radial parts of Gaussians
            beta_i, beta_j = np.random.rand(2)+0.1
            
            self._compare_all_integrals(xi,yi,zi, beta_i, xj,yj,zj, beta_j, alpha)
                
    def _compare_all_integrals(self, xi,yi,zi, beta_i, xj,yj,zj, beta_j, alpha):
        """
        enumerate all polarization integrals that satisfy the following conditions
        on the integer parameters:

           k           in [3,4,5]
           mx+my+mz    in [0,1,2]
           nxi+nyi+nzi in [0,1,2]   s,p and d-functions
           nxj+nyj+nzj in [0,1,2]

        and compare the numerical with the analytical implementation.
        """
        # Check integrals up to d-function
        l_max = 2
        
        # enumerate powers of polarization operator
        for k in [3,4,5]:
            for mmax in [0,1,2]:
                for mx,my,mz in partition3(0):
                    # power of cutoff function
                    q = max(2, int(kappa(k) - kappa(mx) - kappa(my) + kappa(mz) - 1))
                    # enumerate angular momenta of basis functions
                    for li in range(0, l_max+1):
                        for lj in range(0, l_max+1):
                            # prepare for integrals between shells with angular momenta li and lj
                            pol = PolarizationIntegral(xi,yi,zi, li, beta_i,
                                                       xj,yj,zj, lj, beta_j,
                                                       k, mx,my,mz,
                                                       alpha, q)
                            for nxi,nyi,nzi in partition3(li):
                                for nxj,nyj,nzj in partition3(lj):
                                    label=f"k={k} mx={mx} my={my} mz={mz} , q={q} , nxi={nxi} nyi={nyi} nzi={nzi} , nxj={nxj} nyj={nyj} nzj={nzj}"
                                    with self.subTest(label=label):
                                        # fast C++ implementation
                                        pint = pol.compute_pair(nxi,nyi,nzi,
                                                                nxj,nyj,nzj)

                                        # slow numerical integrals
                                        pint_numerical = polarization_integral_numerical(xi,yi,zi, nxi,nyi,nzi, beta_i,
                                                                                         xj,yj,zj, nxj,nyj,nzj, beta_j,
                                                                                         k, mx,my,mz,
                                                                                         alpha, q)

                                        if self.verbosity > 0:
                                            print(label + f"  integral= {pint:+12.7f}  numerical= {pint_numerical:+12.7f}")
                                        absolute_error = abs(pint - pint_numerical)
                                        relative_error = abs(pint - pint_numerical)/abs(pint_numerical)
                                        # numerical accuracy is quite low
                                        self.assertTrue( (absolute_error < 1.0e-3) or (relative_error < 1.0e-3) )


    
if __name__ == '__main__':
    # make random numbers reproducible
    np.random.seed(0)
    
    unittest.main()
