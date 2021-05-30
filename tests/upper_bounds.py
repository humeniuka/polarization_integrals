#!/usr/bin/env python
"""
upper bounds of polarization integrals that can be used for screening negligible ones
"""
try:
    import becke
except ImportError as err:
    print("""
    Numerical integration requires the `becke` module which can be obtained from 
       `https://github.com/humeniuka/becke_multicenter_integration`
    """)
    raise err

import numpy as np

def upper_bound(xi, yi, zi,   nxi, nyi, nzi,  beta_i,
                xj, yj, zj,   nxj, nyj, nzj,  beta_j,
                k, mx, my, mz,
                alpha, q):
    """
    upper bound for polarization integral based on the Cauchy-Schwarz inequality

      (i|Op|j) = <ij,Op> <= ||ij|| ||O||

                          1/2    /  3   2             2     1/2
               = (i^2|j^2)     { | dr Op (r) Cutoff(r)  dr }
                                 /

    Parameters
    ----------
    same as for `polarization_ints_numerical.polarization_integral`

    Returns
    -------
    bound        :    float
      upper bound on the integral <CGTO(i)|x^mx y^my z^mz r^{-k} (1 - exp(-alpha r^2))^q |CGTO(j)>
    """
    # unnormalized bra and ket Gaussian type orbitals
    def CGTOi(x,y,z):
        dx, dy, dz = x-xi, y-yi, z-zi
        dr2 = dx*dx+dy*dy+dz*dz
        return pow(dx, nxi)*pow(dy,nyi)*pow(dz,nzi) * np.exp(-beta_i * dr2)

    def CGTOj(x,y,z):
        dx, dy, dz = x-xj, y-yj, z-zj
        dr2 = dx*dx+dy*dy+dz*dz
        return pow(dx, nxj)*pow(dy,nyj)*pow(dz,nzj) * np.exp(-beta_j * dr2)

    # polarization operator
    def Op(x,y,z):
        r = np.sqrt(x*x+y*y+z*z)
        return pow(x,mx)*pow(y,my)*pow(z,mz) * pow(r, -k)

    # cutoff function, without this the integrals do not exist
    def cutoff(x,y,z):
        r2 = x*x+y*y+z*z
        return pow(1 - np.exp(-alpha*r2), q)

    # functions to integrate
    # (i^2|j^2)
    def integrand_i2j2(x,y,z):
        # product of orbital densities
        return CGTOi(x,y,z)**2 * CGTOj(x,y,z)**2

    def integrand_Op2(x,y,z):
        return (Op(x,y,z) * cutoff(x,y,z))**2

    # place a spherical grid on each center: ri, 0, rj
    atoms = [(1, (xi, yi, zi)),
             (1, (0.0, 0.0, 0.0)),
             (1, (xj, yj, zj))]

    # The integral of `Op(r)^2 Cutoff(r)^2` is a constant and does not depend
    # on the basis functions, only on mx,my,mz and k. It should be only calculated
    # once.
    
    bound = np.sqrt(   becke.integral(atoms, integrand_i2j2)
                     * becke.integral(atoms, integrand_Op2) )

    return bound

        
