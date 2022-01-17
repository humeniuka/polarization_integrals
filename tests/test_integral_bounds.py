#!/usr/bin/env python
"""
compare upper bounds of integrals with exact integrals.
"""
import unittest
import numpy as np

# fast C++ implementation
from polarization_integrals import PolarizationIntegral
from upper_bounds import upper_bound, upper_bound_radial

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

def plot_integrals_vs_bounds(exact, bound):
    import matplotlib
    matplotlib.rc('xtick', labelsize=16)
    matplotlib.rc('ytick', labelsize=16)
    matplotlib.rc('legend', fontsize=16)
    matplotlib.rc('axes', labelsize=16)
    
    import matplotlib.pyplot as plt

    plt.title("Cauchy-Schwarz bound on polarization integrals")
    plt.xlabel("exact integrals (log)")
    plt.ylabel("upper bound (log)")
    plt.xscale('log')
    plt.yscale('log')

    exact = np.array(exact)
    bound = np.array(bound)
    
    xmin = min(exact.min(), bound.min())
    xmax = max(exact.max(), bound.max())
    
    plt.plot(exact, bound, "o", fillstyle='none', color='blue', label="Cauchy-Schwarz")
    # straight line ~ perfectly tight bound
    plt.plot([xmin,xmax], [xmin,xmax], color="black", label="perfect correlation")

    # threshold for neglecting one-electron integrals
    threoe = 1.0e-8
    plt.plot([xmin,xmax], [threoe, threoe], color="red", label="1e threshold")
    
    plt.legend()
    plt.tight_layout()

    plt.savefig("/tmp/integral_bounds_Cauchy-Schwarz.png")

    plt.show()
    
class TestUpperBound(unittest.TestCase):
    verbosity = 1
    def test_Cauchy_Schwarz_bound(self):
        # exact integrals
        self.exact = []
        # upper bounds
        self.bound = []
        
        self._small_b()
        self._large_b()

        self.exact = np.array(self.exact)
        self.bound = np.array(self.bound)
        #
        #np.savetxt("/tmp/exact_vs_bound.dat", np.vstack((self.exact, self.bound)).T)
        
        # Disable plotting if test suite is run on cluster without graphics
        #plot_integrals_vs_bounds(self.exact, self.bound)
        
    def _small_b(self):
        """polarization integrals and upper bounds 
        for basis centers on opposite sides of the origin (b=0)"""
        # cutoff function
        alpha = 50.0

        n_samples = 1
        for i in range(0, n_samples):
            if self.verbosity > 0:
                print(f"Random sample {i}")
            # First center is chosen randomly
            xi,yi,zi = 2.0 * 2.0*np.array(np.random.rand(3)-0.5)
            # exponents of radial parts of Gaussians
            beta_i, beta_j = np.random.rand(2)+0.1 + 0.5
            # The second center is chosen such that
            #  b = |beta_i * ri + beta_j * rj| = eps, such that x = b^2 / alpha << 1
            for eps in np.linspace(0.0, 5.0, 10):
                xj,yj,zj = - beta_i/beta_j * np.array([xi,yi,zi]) + eps * 2.0*(np.random.rand(3)-0.5)

                self._compare_all_integrals(xi,yi,zi, beta_i, xj,yj,zj, beta_j, alpha)

    def _large_b(self):
        """polarization integrals and upper bounds for basis centers at random locations"""
        # cutoff function
        alpha = 50.0

        n_samples = 50
        for i in range(0, n_samples):
            if self.verbosity > 0:
                print(f"Random sample {i}")
            # Both centers are chosen randomly within a ball 5 Angstrom
            xi,yi,zi = 5.0 * 2.0*np.array(np.random.rand(3)-0.5)
            xj,yj,zj = 5.0 * 2.0*np.array(np.random.rand(3)-0.5)
            # exponents of radial parts of Gaussians from the interval [0.01, 2.01]
            beta_i, beta_j = 2*np.random.rand(2)+0.01
            
            self._compare_all_integrals(xi,yi,zi, beta_i, xj,yj,zj, beta_j, alpha)

        
            
    def _compare_all_integrals(self, xi,yi,zi, beta_i, xj,yj,zj, beta_j, alpha):
        """
        enumerate all polarization integrals that satisfy the following conditions
        on the integer parameters:

           k           3 or 4
           mx+my+mz    1 or 0
           nxi+nyi+nzi in [0,1,2]   s,p and d-functions
           nxj+nyj+nzj in [0,1,2]

        and check that the numerical integrals are smaller than the upper bounds.
        """
        # Check integrals up to d-function
        l_max = 2
        
        # enumerate powers of polarization operator
        for (k,mmax) in [(3,1), (4,0)]:
            for mx,my,mz in partition3(mmax):
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

                                    # upper bound to pint
                                    pint_bound = upper_bound(xi,yi,zi, nxi,nyi,nzi, beta_i,
                                                             xj,yj,zj, nxj,nyj,nzj, beta_j,
                                                             k, mx,my,mz,
                                                             alpha, q)
                                    
                                    if self.verbosity > 0:
                                        print(label + f"  |integral|= {abs(pint):+12.7f}  upper bound= {pint_bound:+12.7f}")

                                    self.exact.append(pint)
                                    self.bound.append(pint_bound)
                                    
                                    self.assertTrue(abs(pint) <= pint_bound)

if __name__ == '__main__':
    # make random numbers reproducible
    np.random.seed(0)
    
    unittest.main()
