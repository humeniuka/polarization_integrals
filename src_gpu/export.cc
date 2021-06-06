/*
  create python bindings for polarization integrals on the GPU

  Polarization integrals are processed in batches for lists of primitive pairs.
 */
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

// By default this code is compiled for double precision. To switch to single precision
// define the macro SINGLE_PRECISION.
// The python module will be called _polarization_gpu.so or _polarization_gpu_sp.so 
// depending on the precision of the floating point data type.

#ifdef SINGLE_PRECISION
#  define REAL_TYPE float
#  define MODULE_NAME _polarization_gpu_sp
#else
#  define REAL_TYPE double
#  define MODULE_NAME _polarization_gpu
#endif

// Note that here `REAL` is a data type, while `real` is a template parameter.
typedef REAL_TYPE REAL;

#include "polarization.h"

namespace py = pybind11;
using namespace pybind11::literals;

template <typename real>
std::vector<real> polarization_prim_pairs_wrapper(
				     // array of pairs of primitives
                                     const std::vector<PrimitivePair<real>> &pairs_vector,
				     // operator    O(r) = x^mx y^my z^mz |r|^-k 
				     int k, int mx, int my, int mz,
				     // cutoff function F2(r) = (1 - exp(-alpha r^2))^q
				     real alpha,  int q) {
  // This code only works on a GPU, check if we have one available.
  int count;
  cudaError_t err;
  err = cudaGetDeviceCount(&count);
  if ((err != cudaSuccess) || (count == 0)) {
    throw std::runtime_error("No CUDA device found!");
  }
  // access underlying data of primitive pairs
  const PrimitivePair<real> *pairs = pairs_vector.data();
  int npair = pairs_vector.size();
  // determine size of buffer for integrals
  int buffer_size = integral_buffer_size<real>(pairs, pairs_vector.size());
  // integrals are stored in this buffer
  std::vector<real> buffer_vector(buffer_size);
  real *buffer = buffer_vector.data();

  // allocate memory for pairs of primitives on GPU
  PrimitivePair<real> *pairs_;
  cudaMalloc((void **) &pairs_, sizeof(PrimitivePair<real>) * npair);
  // copy primitive data to device
  cudaMemcpy(pairs_, pairs, sizeof(PrimitivePair<real>) * npair,  cudaMemcpyHostToDevice);

  // allocate memory for integrals on the GPU
  real *buffer_;
  cudaMalloc((void **) &buffer_, sizeof(real) * buffer_size);

  // do the integrals
  polarization_prim_pairs<real>(pairs_, npair,
				buffer_,
				k, mx, my, mz, alpha, q);
  
  // copy integrals from GPU to CPU
  cudaMemcpy(buffer, buffer_, sizeof(real) * buffer_size,  cudaMemcpyDeviceToHost);

  // release dynamic memory on GPU
  cudaFree(pairs_);
  cudaFree(buffer_);
  

  return buffer_vector;
}

PYBIND11_MODULE(MODULE_NAME, m) {
  py::class_<Primitive<REAL>>(m, "Primitive")
    .def(py::init<REAL, REAL, int, 
	 REAL, REAL, REAL, int>(),
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
    .def_readwrite("coef", &Primitive<REAL>::coef)
    .def_readwrite("exp",  &Primitive<REAL>::exp)
    .def_readwrite("l",    &Primitive<REAL>::l)
    .def_readwrite("x",    &Primitive<REAL>::x)
    .def_readwrite("y",    &Primitive<REAL>::y)
    .def_readwrite("z",    &Primitive<REAL>::z)
    .def_readwrite("shellIdx", &Primitive<REAL>::shellIdx);

  py::class_<PrimitivePair<REAL>>(m, "PrimitivePair")
    .def(py::init<Primitive<REAL>, Primitive<REAL>, int>(),
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
    .def_readwrite("primA",     &PrimitivePair<REAL>::primA)
    .def_readwrite("primB",     &PrimitivePair<REAL>::primB)
    .def_readwrite("bufferIdx", &PrimitivePair<REAL>::bufferIdx);

  m.def("polarization_prim_pairs", &polarization_prim_pairs_wrapper<REAL>, 
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

