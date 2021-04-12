
Requirements
------------
 - python3
 - pybind11

Getting Started
---------------
The package is installed by running

.. code-block:: bash

   $ pip install -e .

To check the proper functioning of the code, it is recommended to run a set of tests with
   
.. code-block:: bash

   $ cd tests
   $ python -m unittest

Example
-------
The following code evaluates the matrix elements of the operator `r^{-4}` between an
unnormalized s-orbital at the origin and a shell of unnnormalized p-orbitals at the
point rj=(0.0, 0.0, 1.0).

First a `PolarizationIntegral` is created, which needs to know the centers, angular momenta 
and radial exponents of the two shells, as well as the polarization operator and the cutoff function:

.. code-block:: python

   $ from polarization_integrals import PolarizationIntegral

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

			    
Then the integrals can be evaluated for all combinations of unnormalized primitives
from each shell. 
			    
.. code-block:: python
		
   # <s|Op|px>
   print( I.compute_pair(0,0,0,  1,0,0) )
   # <s|Op|py>
   print( I.compute_pair(0,0,0,  0,1,0) )
   # <s|Op|pz>
   print( I.compute_pair(0,0,0,  0,0,1) )

