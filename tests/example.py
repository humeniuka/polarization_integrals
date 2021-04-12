#!/usr/bin/env python
"""
example from the docstring of the class `PolarizationIntegral`
"""

from polarization_integrals import PolarizationIntegral

# s- and p-shell
li, lj = 0,1
# Op = r^{-4}
k, mx,my,mz = 4, 0,0,0
# cutoff function
alpha = 50.0
q = 4

I = PolarizationIntegral(0.0, 0.0, 0.0,  li,  0.5,  
                         0.0, 0.0, 1.0,  lj,  0.5,
                         k, mx,my,mz,
                         alpha, q)

# <s|Op|px>
print( I.compute_pair(0,0,0,  1,0,0) )
# <s|Op|py>
print( I.compute_pair(0,0,0,  0,1,0) )
# <s|Op|pz>
print( I.compute_pair(0,0,0,  0,0,1) )

