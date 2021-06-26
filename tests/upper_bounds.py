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
from radial_overlap import radial_overlap_analytical

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

        
def operator2_inegrals(k, mx,my,mz,
                       alpha, q):
    """
    square of polarization operator O(r) = x^mx y^my z^mz |r|^{-k}

           2    /  3   2             2   
      ||O||   = | dr Op (r) Cutoff(r)  dr
                /

    Returns
    -------
    op2      :  float
      integral ||O||^2
    """
    # polarization operator
    def Op(x,y,z):
        r = np.sqrt(x*x+y*y+z*z)
        return pow(x,mx)*pow(y,my)*pow(z,mz) * pow(r, -k)

    # cutoff function, without this the integrals do not exist
    def cutoff(x,y,z):
        r2 = x*x+y*y+z*z
        return pow(1 - np.exp(-alpha*r2), q)

    # function to integrate
    def integrand_Op2(x,y,z):
        return (Op(x,y,z) * cutoff(x,y,z))**2

    # place a spherical grid at the origin
    atoms = [(1, (0.0, 0.0, 0.0))]

    op2 = becke.integral(atoms, integrand_Op2)

    return op2

def operator2_integrals(cutoff_power, cutoff_exponent, k, mx,my,mz):
    """
    integral of square of polarization operator O(r) = x^mx y^my z^mz |r|^{-k}

           2    /  3   2             2                                            q
      ||O||   = | dr Op (r) Cutoff(r)  dr   with Cutoff(r) = (1 - exp(-alpha r^2))
                /

    for selected values of k, mx,my,mz and q (cutoff power) as a function of the
    cutoff exponent alpha.

    Parameters
    ----------
    cutoff_power    :  int (only 2,4)
      power q in cutoff function 
    cutoff_exponent :  float > 0
      exponent alpha in cutoff function
    mx,my,mz     :  ints >= 0
    k            :  int > 2  (only 3,6)
      powers in polarization operator

    Returns
    -------
    op2      :  float
      integral ||O||^2    
    """
    # These integrals were generated with the Mathematica script 'operator_squared_integrals.nb'
    if   (k == 3 and mx+my+mz == 1 and cutoff_power == 2):
        op2 = 8/3 * (1 - 3*np.sqrt(2) + 2*np.sqrt(3)) * pow(np.pi,3/2) * pow(cutoff_exponent,1/2)
    elif (k == 6 and mx+my+mz == 2 and cutoff_power == 4):
        op2 = 128/75 * (-279 - 30*np.sqrt(2) + 63 * np.sqrt(3) + 175 * np.sqrt(5) - 126*np.sqrt(6) + 49 * np.sqrt(7)) * pow(np.pi,3/2) * pow(cutoff_exponent,5/2)
    else:
        raise NotImplementedError(f"Integral ||O||^2 not implemented for k={k}, mx+my+mz={mx+my+mz} and q={cutoff_power}")
    return op2

def upper_bound_radial(xi, yi, zi,   nxi, nyi, nzi,  beta_i,
                       xj, yj, zj,   nxj, nyj, nzj,  beta_j,
                       k, mx, my, mz,
                       alpha, q):
    """
    upper bound for polarization integral based on the Cauchy-Schwarz inequality

      (i|Op|j) = <ij,Op> <= ||ij|| ||O||

                          1/2    /  3   2             2     1/2
               = (i^2|j^2)     { | dr Op (r) Cutoff(r)  dr }
                                 /

    The Cartesian Gaussians are approximated by spherically symmetric Gaussians with the
    same total angular momentum

          2            2nxi      2nyi      2nzi                    2
      CGTO (r) = (x-xi)    (y-yi)    (z-zi)    exp(-2 beta_i (r-ri) )
          i
                        2li                     2
               <= (r-ri)    exp(-2 beta_i (r-ri) )      with li=nxi+nyi+nzi

    So that the first integral

                  /        2        2
      (i^2|j^2) = | CGTO(i)  CGTO(j)  dr
                  /     

    is bounded by
                   /     2        2
                <= | RGTO (r) RGTO (r)
                   /     i        j

    Parameters
    ----------
    same as for `polarization_ints_numerical.polarization_integral`

    Returns
    -------
    bound        :    float
      upper bound on the integral <CGTO(i)|x^mx y^my z^mz r^{-k} (1 - exp(-alpha r^2))^q |CGTO(j)>
    """    
    op2 = operator2_integrals(q, alpha, k, mx,my,mz)
    # total angular momenta
    li = nxi+nyi+nzi
    lj = nxj+nyj+nzj
    if (li > lj):
        # swap i and j
        (xi,yi,zi, li, beta_i), (xj,yj,zj, lj, beta_j) = (xj,yj,zj, lj, beta_j), (xi,yi,zi, li, beta_i)
    
    # overlap of spherically symmetric radial Gaussians
    #  RGTO_i(r) = (r-ri)^(2*li) exp(-2*beta_i (r-ri)^2
    # and
    #  RGTO_j(r) = (r-rj)^(2*lj) exp(-2*beta_j (r-rj)^2
    i2j2_rad = radial_overlap_analytical(xi, yi, zi, li, beta_i,
                                         xj, yj, zj, lj, beta_j)
    bound = np.sqrt(op2 * i2j2_rad)

    return bound


def plot_operator2_integrals():
    cutoff_power = 2

    integrals = {
        (3,1,0,0, 1*cutoff_power): [],
        (6,2,0,0, 2*cutoff_power): [],
        (6,1,1,0, 2*cutoff_power): []}

    cutoff_exponents = np.linspace(0.01, 50, 250)
    for alpha in cutoff_exponents:
        for (k,mx,my,mz, q) in [(3, 1,0,0, 1*cutoff_power),
                                (6, 2,0,0, 2*cutoff_power),
                                (6, 1,1,0, 2*cutoff_power),
    ]:
            op2 = operator2_inegrals(k, mx,my,mz,
                                     alpha, q)
            integrals[(k,mx,my,mz,q)].append(op2)

    import matplotlib.pyplot as plt
    plt.title("cutoff power $q=%d$" % q)
    plt.xlabel("cutoff exponent / bohr")
    plt.ylabel(r"||O(r)||$^2$")
    for key in integrals.keys():
        l, = plt.plot(cutoff_exponents, np.array(integrals[key]), label="$k=%d$ $m_x=%d$ $m_y=%d$ $m_z=%d$ $q=%d$ (numerical)" % key)
        k,mx,my,mz,q = key
        plt.plot(cutoff_exponents, operator2_integrals(q, cutoff_exponents, k, mx,my,mz),
                 "o", color=l.get_color(), markersize=5,
                 label="$k=%d$ $m_x=%d$ $m_y=%d$ $m_z=%d$ $q=%d$ (exact)" % key)
        
    plt.legend()
    plt.show()
    

if __name__ == "__main__":
    plot_operator2_integrals()
    
