#!/usr/bin/env python
"""
create Latex table with comparison of polarization integrals for
cases 1.1, 2.1 and 2.2 with the 
original analytical (wrong), the corrected analytical and the numerical integration methods.
"""
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
import numpy.linalg as la

# fast C++ implementation
from polarization_integrals import PolarizationIntegral

# numerical integral using Becke's multicenter grids
from polarization_ints_numerical import polarization_integral as polarization_integral_numerical
# increase resolution of integration grids
from becke import settings
settings.radial_grid_factor = 3  # increase number of radial points by factor 3
settings.lebedev_order = 23      # angular Lebedev grid of order 23

from polarization_ints_reference import polarization_integral as polarization_integral_reference


# exponents
beta_i = 1.0
beta_j = 1.0

# cutoff function
alpha = 50.0
q = 2

# centers of GTOs
xi,yi = 0,0
xj,yj = 0,0
# The z-coordiantes of GTOs are varied to get different cases (b > 0 and b == 0)

# powers of Cartesian coordinates
nxi,nyi,nzi = 0,0,0
nxj,nyj,nzj = 0,0,0

# powers in operator
mx,my = 0,0
# mz is varied to get different cases (1, 2.1 or 2.2)

def write_row(k=3, mz=0, zi=0.5, zj=0.5, nzj=0):
    li = nxi+nyi+nzi
    lj = nxj+nyj+nzj
    
    # prepare for integrals between shells with angular momenta li and lj
    pol = PolarizationIntegral(xi,yi,zi, li, beta_i,
                               xj,yj,zj, lj, beta_j,
                               k, mx,my,mz,
                               alpha, q)
    # fast C++ implementation
    pint = pol.compute_pair(nxi,nyi,nzi,
                            nxj,nyj,nzj)
    
    # numerical integrals
    # The numerical integration fails if the centers coincide exactly, so we add a small number
    eps = 1.0e-6
    if (abs(zi) < 10*eps) and (abs(zj) < 10*eps):
        zi -= eps
        zj += eps
        zi_ = zi - eps
        zj_ = zj + eps
    else:
        zi_ = zi
        zj_ = zj

    pint_numerical = polarization_integral_numerical(xi,yi,zi_, nxi,nyi,nzi, beta_i,
                                                     xj,yj,zj_, nxj,nyj,nzj, beta_j,
                                                     k, mx,my,mz,
                                                     alpha, q)

    try:
        # Xiao's reference integrals using the original definition of H(1,a) in Schwerdtfeger's article
        pint_reference = polarization_integral_reference(xi,yi,zi, nxi,nyi,nzi, beta_i,
                                                         xj,yj,zj, nxj,nyj,nzj, beta_j,
                                                         k, mx,my,mz,
                                                         alpha, q,
                                                         original=True)
    except ZeroDivisionError:
        pint_reference = np.nan
        
    #### determine which sub case we are dealing with

    # 1, 2.1 or 2.2 ?
    if (k % 2 == 0):
        case = "     1  "
    else:
        j = k//2
        case = "     2."
        s_max = li+lj+mx+my+mz
        if (s_max - j >= 0):
            case += "1"
        else:
            case += "2"
        
    # b > 0 or b = 0 ?
    
    ri = np.array([xi,yi,zi])
    rj = np.array([xj,yj,zj])
    b = la.norm(beta_i * ri + beta_j * rj)
    if (abs(b) < 1.0e-10):
        case += " (b = 0) "
    else:
        case += "         "
        
    print(f"{case} &  {k} & {mz}     & ${zi}$  & ${zj}$  &  ${pint_reference:+12.7f}$ &  ${pint:+12.7f}$   & ${pint_numerical:+12.7f}$     \\\\")   

# write header
print(f"Case              &  k & $m_3$ & $z_1$  & $z_2$  &       original  &       corrected   &      numerical     \\\\ \\midrule") 

# Case 1  (b > 0)
write_row(k=4, mz=0, zi=-1.0, zj=0.5)
## Case 1  (b == 0)
#write_row(k=4, mz=0, zi=-1.0, zj=1.0)
# Case 1  (b == 0, all centers coincide)
#write_row(k=4, mz=0, zi=0.0, zj=0.0)

# Case 2.1  (b > 0)
write_row(k=3, mz=1, zi=-1.0, zj=0.5)
## Case 2.1  (b == 0)
#write_row(k=3, mz=1, zi=-1.0, zj=1.0)    
# Case 2.1  (b == 0, all centers coincide)
#write_row(k=3, mz=1, zi=0.0, zj=0.0)    

# Case 2.2  (b > 0)
write_row(k=3, mz=0, zi=-1.0, zj=0.5)
## Case 2.2  (b == 0)
#write_row(k=3, mz=0, zi=-1.0, zj=1.0)
# Case 2.2  (b == 0, all centers coincide)
write_row(k=3, mz=0, zi=0.0, zj=0.0)
