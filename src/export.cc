/*
  create python bindings for class `PolarizationIntegral`
 */
#include <pybind11/pybind11.h>

#include "polarization.h"

namespace py = pybind11;
using namespace pybind11::literals;

// <for testing only
extern double test_d_func(double x, int p, double w0);
extern double test_d_func_zero_limit(double x, int p, double w0);
extern double test_g_func(double x, int p);
extern double m_func(double x);
extern double test_h_func(double x, int p);
extern double test_h_func_large_x(double x, int p);
extern double test_h_func_small_x(double x, int p);
// >

PYBIND11_MODULE(_polarization, m) {
  py::class_<PolarizationIntegral>(m, "PolarizationIntegral")
    .def(py::init<
	 double, double, double, int, double,
	 double, double, double, int, double,
	 int, int, int, int,
	 double, int
	 >(),
	 R"LITERAL(
polarization integrals between two shells of Cartesian Gaussian-type orbitals.

This class provides integrals between two shells with angular momenta `li` and `lj`. 
The member function `compute_pair(...)` calculates the integrals between two primitives from each shell.

Parameters
----------
xi,yi,zi     :    floats
  Cartesian positions of center i
li           :    int >= 0
  angular momentum of shell i, li=nxi+nyi+nzi
beta_i       :    float > 0
  exponent of radial part of orbital i
xj,yj,zj     :    floats
  Cartesian positions of center j
lj           :    int >= 0
  angular momentum of shell j, lj=nxj+nyj+nzj
beta_j       :    float > 0
  exponent of radial part of orbital j
k, mx,my,mz  :    ints >= 0, k > 2
  powers in the polarization operator `O(x,y,z) = x^mx * y^my * z^mz |r|^{-k}`
alpha        :    float >> 0
  exponent of cutoff function
q            :    int
  power of cutoff function


Example
-------
The following code evaluates the matrix elements of the operator r^{-4} between an
unnormalized s-orbital at the origin and a shell of unnnormalized p-orbitals at the
point rj=(0.0, 0.0, 1.0).

First a `PolarizationIntegral` is created, which needs to know the centers, angular momenta 
and radial exponents of the two shells, as well as the polarization operator and the cutoff function:

>>> from polarization_integrals import PolarizationIntegral
>>> li, lj = 0,1
>>> k, mx,my,mz = 4, 0,0,0
>>> alpha = 50.0
>>> q = 4
>>> I = PolarizationIntegral(0.0, 0.0, 0.0,  li,  0.5,  
...                          0.0, 0.0, 1.0,  lj,  0.5,
...                          k, mx,my,mz,
...                          alpha, q)

Then the integrals are calculated for all combinations of primitives from each shell:
`<s|r^{-4}|px>`, `<s|r^{-4}|py>` and `<s|r^{-4}|pz>`:

>>> I.compute_pair(0,0,0,  1,0,0)
0.0
>>> I.compute_pair(0,0,0,  0,1,0)
0.0
>>> I.compute_pair(0,0,0,  0,0,1)
-29.316544716559417


Polarization Integrals
----------------------
The polarization integrals are defined as

             mx  my  mz
            x   y   z           - alpha r  q
  <CGTO   | ----------- (1 - exp          )   |CGTO  >
       i        r^k                                j

between unnormalized primitive Cartesian Gaussian functions 

                        nxi       nyi       nzi                     2
   CGTO (x,y,z) = (x-xi)    (y-yi)    (z-zi)    exp(-beta_i (r - ri)  )
       i

and

                        nxj       nyj       nzj                     2
   CGTO (x,y,z) = (x-xj)    (y-yj)    (z-zj)    exp(-beta_j (r - rj)  )
       j

for k > 2. The power of the cutoff function q has to satisfy

  q >= kappa(k/2) - kappa(mx/2) - kappa(my/2) - kappa(nz/2) - 1

                      n/2   if n is even
where kappa(n) = {
                    (n+1)/2 if n is odd


References
----------
[CPP] P. Schwerdtfeger, H. Silberbach,
      'Multicenter integrals over long-range operators using Cartesian Gaussian functions',
      Phys. Rev. A 37, 2834
      https://doi.org/10.1103/PhysRevA.37.2834
[CPP-Erratum] Phys. Rev. A 42, 665
      https://doi.org/10.1103/PhysRevA.42.665

)LITERAL",
	 "xi"_a, "yi"_a, "zi"_a, "li"_a, "beta_i"_a,
	 "xj"_a, "yj"_a, "zj"_a, "lj"_a, "beta_j"_a,
	 "k"_a, "mx"_a, "my"_a, "mz"_a,
	 "alpha"_a, "q"_a)
    .def("compute_pair", &PolarizationIntegral::compute_pair,
	 R"LITERAL(
compute polarization integrals between two unnormalized CGTOs with powers `nxi,nyi,nzi` and `nxj,nyj,nzj`.

The primitives have to belong to shells with angular momenta `li` and `lj`, i.e.
the powers have to satisfy `nxi+nyi+nzi = li` and `nxj+nyj+nzj = lj`, 
where `li` and `lj` are the angular momenta specified when creating the `PolarizationIntegral` object. 
)LITERAL",
	 "nxi"_a, "nyi"_a, "nzi"_a,
	 "nxj"_a, "nyj"_a, "nzj"_a);
  // <only for testing
  m.def("test_d_func", &test_d_func, "export implementation of d(p+1/2,x) for testing");
  m.def("test_d_func_zero_limit", &test_d_func_zero_limit, "export implementation of \tilde{d}(p+1/2,x) for testing");
  m.def("test_g_func", &test_g_func, "export implementation of g(p+1/2,x) for testing");
  m.def("m_func", &m_func, "export implementation of m(x) for testing");
  m.def("test_h_func", &test_h_func, "export implementation of H(p,x) for testing");
  m.def("test_h_func_large_x", &test_h_func_large_x, "export implementation of H(p,x) (for large x) for testing");
  m.def("test_h_func_small_x", &test_h_func_small_x, "export implementation of H(p,x) (for small x) for testing");
  // >
}

