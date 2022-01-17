
library for evaluating polarization integrals according to Schwerdtfeger et al. (1988) (Reference [CPP]).

Polarization Integrals
----------------------
The polarization integrals are defined as

.. math::

   \langle \text{CGTO}_i \vert \frac{x^{m_x} y^{m_y} z^{m_z}}{r^k} \left(1 - e^{-\alpha r} \right)^q \vert \text{CGTO}_j \rangle

between unnormalized primitive Cartesian Gaussian functions 

.. math::

   \text{CGTO}_i(x,y,z) = (x - x_i)^{n_{xi}} (y - y_i)^{n_{yi}} (z - z_i)^{n_{zi}} \exp\left(-\beta_i (\vec{r} - \vec{r}_i)^2 \right)

and

.. math::
   
   \text{CGTO}_j(x,y,z) = (x - x_j)^{n_{xj}} (y - y_j)^{n_{yj}} (z - z_j)^{n_{zj}} \exp\left(-\beta_j (\vec{r} - \vec{r}_j)^2 \right)

   
for :math:`k > 2`. The power of the cutoff function :math:`q` has to satisfy

.. math::
   
  q \geq \kappa(k) - \kappa(m_x) - \kappa(m_y) - \kappa(m_z) - 1

where

.. math::

   \kappa = \begin{cases}
              \frac{n}{2}   \quad \text{, if n is even } \\
              \frac{n+1}{2} \quad \text{, if n is odd  }
	    \end{cases}
   


Requirements
------------
 - python3
 - pybind11

For parallel evaluation of integrals on a GPU you need in addition

 - CUDA Toolkit (nvcc)
 - a CUDA supported GPU device


Getting Started
---------------
The package is installed by running

.. code-block:: bash

   $ pip install -e .

Tests
-----
To check the proper functioning of the code, it is recommended to run a set of tests.
In order to compute the numerical integrals needed for comparison one first has to install
the python package `becke` from https://github.com/humeniuka/becke_multicenter_integration .
Then the test suite is run with
   
.. code-block:: bash

   $ cd tests
   $ python -m unittest

Example
-------
The following code evaluates the matrix elements of the operator :math:`r^{-4}` between an
unnormalized s-orbital at the origin and a shell of unnnormalized p-orbitals at the
point :math:`\vec{r}_j=(0.0, 0.0, 1.0)`.

First a `PolarizationIntegral` is created, which needs to know the centers, angular momenta 
and radial exponents of the two shells, as well as the polarization operator and the cutoff function:

.. code-block:: python

   from polarization_integrals import PolarizationIntegral

   # s- and p-shell
   li, lj = 0, 1
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


GPU Support
-----------
Polarization integrals can be calculated in parallel on a GPU which supports CUDA.
The kernels with python binding are located in the folder `src_gpu/`.
The kernels and python wrapper are compiled with

.. code-block:: bash

   $ cd src_gpu
   $ make

The correctness of the GPU integrals should be verified by comparison with the CPU implementation
by running a set of tests with

.. code-block:: bash

   $ cd tests_gpu
   $ python -m unittest


References
----------
[CPP] P. Schwerdtfeger, H. Silberbach,
      'Multicenter integrals over long-range operators using Cartesian Gaussian functions',
      Phys. Rev. A 37, 2834
      https://doi.org/10.1103/PhysRevA.37.2834
[CPP-Erratum] Phys. Rev. A 42, 665
      https://doi.org/10.1103/PhysRevA.42.665
[CPP-Erratum2] Phys. Rev. A 103, 069901
      https://doi.org/10.1103/PhysRevA.103.069901
