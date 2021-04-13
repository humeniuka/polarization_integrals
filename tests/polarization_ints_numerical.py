#!/usr/bin/env python
"""
polarization integrals are computed by numerical integration on multicenter spherical grids
(Becke's integration scheme)
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

def polarization_integral(xi, yi, zi,   nxi, nyi, nzi,  beta_i,
                          xj, yj, zj,   nxj, nyj, nzj,  beta_j,
                          k, mx, my, mz,
                          alpha, q):
    """
    polarization integrals between unnormalized Cartesian GTOs 
    by numerical integration on a multicenter Becke grid

    Parameters
    ----------
    xi,yi,zi     :    floats
      Cartesian positions of center i
    nxi,nyi,nzi  :    int >= 0
      powers of Cartesian primitive GTO i
    beta_i       :    float > 0
      exponent of radial part of orbital i
    xj,yj,zj     :    floats
      Cartesian positions of center j
    nxj,nyj,nzj  :    int >= 0
      powers of Cartesian primitive GTO j
    beta_j       :    float > 0
      exponent of radial part of orbital j
    k, mx,my,mz  :    ints >= 0, k > 2
      powers in the polarization operator `O(x,y,z) = x^mx * y^my * z^mz |r|^{-k}`
    alpha        :    float >> 0
      exponent of cutoff function
    q            :    int
      power of cutoff function

    Returns
    -------
    integ        :    float
      integral <CGTO(i)|x^mx y^my z^mz r^{-k} (1 - exp(-alpha r^2))^q |CGTO(j)>
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

    # function to integrate
    def integrand(x,y,z):
        return CGTOi(x,y,z) * Op(x,y,z) * cutoff(x,y,z) * CGTOj(x,y,z)


    # place a spherical grid on each center: ri, 0, rj
    atoms = [(1, (xi, yi, zi)),
             (1, (0.0, 0.0, 0.0)),
             (1, (xj, yj, zj))]

    integ = becke.integral(atoms, integrand)

    return integ
