#!/usr/bin/env python
"""
overlap integrals between spherically symmetric Gaussians of the form

       2             2*li                     2
   RGTO (r)  = (r-ri)     exp(-2*beta_i (r-ri) )
          i
"""
# The analytical integrals are compared with numerical integrals
try:
    import becke
except ImportError as err:
    print("""
    Numerical integration requires the `becke` module which can be obtained from 
       `https://github.com/humeniuka/becke_multicenter_integration`
    """)
    raise err

import numpy as np

# fast C++ implementation
from polarization_integrals._polarization import radial_overlap as radial_overlap_analytical

# numerical integration on Becke grid
def radial_overlap_numerical(xi, yi, zi,   li,  beta_i,
                             xj, yj, zj,   lj,  beta_j):
    """
    numerical overlap integral between squares of spherically symmetric
    radial Gaussian basis functions of the form

                       li                      2
      RGTO (r) = (r-ri)    exp(-beta_i (r - ri) )
          i 

    The radial integrals

       (2)  /     2         2
      U   = | RGTO (r)  CGTO (r)
       ij   /     i         j

    are an upper bound for the integrals of the type

       (2)  /     2         2
      S   = | CGTO (r)  CGTO (r)
       ij   /     i         j

    between unnormalized primitive Cartesian basis functions of the form

                       nxi       nyi       nzi                     2
      CGTO (r) = (x-xi)    (y-yi)    (z-zi)    exp(-beta_i (r - ri) )
          i 

    with total angular momentum li = nxi+nyi+nzi.
    """
    # unnormalized bra and ket Gaussian type orbitals
    def RGTOi(x,y,z):
        dx, dy, dz = x-xi, y-yi, z-zi
        dr2 = dx*dx+dy*dy+dz*dz
        dr = np.sqrt(dr2)
        return pow(dr,li) * np.exp(-beta_i * dr2)

    def RGTOj(x,y,z):
        dx, dy, dz = x-xj, y-yj, z-zj
        dr2 = dx*dx+dy*dy+dz*dz
        dr = np.sqrt(dr2)
        return pow(dr,lj) * np.exp(-beta_j * dr2)

    # functions to integrate
    # (i^2|j^2)
    def integrand_density_overlap(x,y,z):
        # product of orbital densities
        return RGTOi(x,y,z)**2 * RGTOj(x,y,z)**2

    # place a spherical grid on each center: ri, 0, rj
    atoms = [(1, (xi, yi, zi)),
             (1, (0.0, 0.0, 0.0)),
             (1, (xj, yj, zj))]

    dolap = becke.integral(atoms, integrand_density_overlap)

    return dolap
