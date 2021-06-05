/*
  create python bindings for polarization integrals on the GPU

  Polarization integrals are processed in batches for lists of primitive pairs.
 */
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

#include "polarization.h"

namespace py = pybind11;
using namespace pybind11::literals;

std::vector<double> polarization_prim_pairs_wrapper(
				     // array of pairs of primitives
				     const std::vector<PrimitivePair> &pairs_vector,
				     // operator    O(r) = x^mx y^my z^mz |r|^-k 
				     int k, int mx, int my, int mz,
				     // cutoff function F2(r) = (1 - exp(-alpha r^2))^q
				     double alpha,  int q) {
  // This code only works on a GPU, check if we have one available.
  int count;
  cudaError_t err;
  err = cudaGetDeviceCount(&count);
  if ((err != cudaSuccess) || (count == 0)) {
    throw std::runtime_error("No CUDA device found!");
  }
  // access underlying data of primitive pairs
  const PrimitivePair *pairs = pairs_vector.data();
  int npair = pairs_vector.size();
  // determine size of buffer for integrals
  int buffer_size = integral_buffer_size(pairs, pairs_vector.size());
  // integrals are stored in this buffer
  std::vector<double> buffer_vector(buffer_size);
  double *buffer = buffer_vector.data();

  // allocate memory for pairs of primitives on GPU
  PrimitivePair *pairs_;
  cudaMalloc((void **) &pairs_, sizeof(PrimitivePair) * npair);
  // copy primitive data to device
  cudaMemcpy(pairs_, pairs, sizeof(PrimitivePair) * npair,  cudaMemcpyHostToDevice);

  // allocate memory for integrals on the GPU
  double *buffer_;
  cudaMalloc((void **) &buffer_, sizeof(double) * buffer_size);

  // do the integrals
  polarization_prim_pairs(pairs_, npair,
			  buffer_,
			  k, mx, my, mz, alpha, q);

  // copy integrals from GPU to CPU
  cudaMemcpy(buffer, buffer_, sizeof(double) * buffer_size,  cudaMemcpyDeviceToHost);

  // release dynamic memory on GPU
  cudaFree(pairs_);
  cudaFree(buffer_);
  

  return buffer_vector;
}


PYBIND11_MODULE(_polarization_gpu, m) {
  py::class_<Primitive>(m, "Primitive")
    .def(py::init<double, double, int, 
	 double, double, double, int>(),
	 R"LITERAL(
Gaussian-type primitive basis function

Parameters
----------
exp      :   float > 0.0
  Gaussian exponent
coef     :   float
  contraction coefficient
l        :   int >= 0
  angular momentum
x,y,z    :   floats
  Cartesian positions of center
shellIdx :   int >= 0
  index of shell to which this primitive belongs

)LITERAL",
	 "coef"_a, "exp"_a, "l"_a,
	 "x"_a, "y"_a, "z"_a, "shellIdx"_a)
    .def_readwrite("coef", &Primitive::coef)
    .def_readwrite("exp",  &Primitive::exp)
    .def_readwrite("l",    &Primitive::l)
    .def_readwrite("x",    &Primitive::x)
    .def_readwrite("y",    &Primitive::y)
    .def_readwrite("z",    &Primitive::z)
    .def_readwrite("shellIdx", &Primitive::shellIdx);

  py::class_<PrimitivePair>(m, "PrimitivePair")
    .def(py::init<Primitive, Primitive, int>(),
	 R"LITERAL(
A pair of Gaussian-type primitives for which polarization integrals should be calculated.

Parameters
----------
primA     :  Primitive
  bra primitive
primB     :  Primitive
  ket primitive
bufferIdx :  int >= 0
  index into output buffer where integrals are stored. In the buffer there has to 
  be enough space to accomodate the `N(primA.l) * N(primB.l)` matrix elements from the 
  combinations of all angular momentum components in each primitive, where
  `N(l)=(l+1)(l+2)/2` is the number of Cartesian basis functions with angular momentum l.

)LITERAL",
	 "primA"_a, "primB"_a, "bufferIdx"_a)
    .def_readwrite("primA",     &PrimitivePair::primA)
    .def_readwrite("primB",     &PrimitivePair::primB)
    .def_readwrite("bufferIdx", &PrimitivePair::bufferIdx);

  m.def("polarization_prim_pairs", &polarization_prim_pairs_wrapper, 
	R"LITERAL(
compute AO polarization integrals for pairs of primitives on the GPU

Parameters
----------
pairs       :    list of PrimitivePair
  pairs of bra and ket Gaussian-type primitives
k, mx,my,mz :    int
  powers defining the polarization operator  
   O(r) = x^mx y^my z^mz |r-rO|^-k 
alpha       :    float
  exponent of cutoff function cutoff(r)=(1-exp(-alpha r^2))^q
q           :    int
  power of cutoff function

Returns
-------
integrals   :    list of float
  polarization integrals, the integrals belonging to the pair of primitives
  pairs[i] start at `integrals[pairs[i].bufferIdx]` (see below for details).


Details
-------
The polarization integral is

                                  mx my mz
                                 x  y  z          - alpha r^2  q
  buffer[ij] = coef  coef  <AO | ----------- (1 - exp         )   |AO  >
                   i     j    i      r^k                             j

The coordinate system has to be shifted such that the polarizable atom lies at the origin.

The integrals for the pair of primitives `pair = pairs[ipair]` starts at index `ij = pair.bufferIdx`. 
The number of integrals per pair depends on the angular momenta of the primitives. 
Since a primitive with angular momentum l has `N(l)=(l+1)(l+2)/2)` Cartesian components, 
there are `N(prim.primA.l) * N(prim.primB.l)` integrals for each pair of primitives. 
The cartesian angular momentum components are generated in the following order:
 
     l               angular functions          (l+1)(l+2)/2
   ----------------------------------------------------------
     0               s                             1
     1               px,py,pz                      3
     2               dxy,dxz,dyz,dxx,dyy,dzz       6

Example: If the pair contains a p-function in the bra and a d-function in the ket primitive,
  the 18 integrals are stored in buffer starting at position `pair.bufferIdx` in the following
  order

     <px|Op|dxy>, <px|Op|dxz>, <px|Op|dyz>, <px|Op|dxx>, <px|Op|dyy>, <px|Op|dzz>,
     <py|Op|dxy>, ...
     ...
     <pz|Op|dxy>, ...                                                 <pz|Op|dzz>

 i.e. row-major order with the bra orbitals as rows and ket orbitals as columns.

NOTE: The primitives should be sorted by the angular momenta (by the key `(primA.l, primB.l)`
  to ensure that most threads execute the same instructions (no warp divergence).



)LITERAL",
	"pairs"_a, "k"_a, "mx"_a, "my"_a, "mz"_a,
	"alpha"_a, "q"_a);
}

