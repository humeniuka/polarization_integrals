
#ifndef _polarization_integrals_h
#define _polarization_integrals_h

#include <cuda_runtime.h>  // for double3

// Primitive Cartesian Gaussian 
//
//                        lx       ly       lz                 2
//   AO(r) = coef * (x-x0)   (y-y0)   (z-z0)    exp(-exp (r-r0) )
//
// 
struct Primitive
{
  double coef;  // normalized contraction coefficient
  double exp;   // Gaussian exponent
  int l;        // total angular momentum l=lx+y+lz : S=0, P=1, D=2, ...
  // cartesian coordinates of center
  double x;     // x0
  double y;     // y0
  double z;     // z0
  int shellIdx; // index of shell to which this primitive belongs
};

// pair of two primitive Gaussians for which the integral should be calculated
struct PrimitivePair
{
  Primitive primA; // bra primitive 
  Primitive primB; // ket primitive   
  // The matrix elements for the integrals between all angular components
  // from each primitive are placed into the buffer beginning at this position
  int bufferIdx;
};

// struct for atomic orbital
struct AtomicOrbital {
  int lx, ly, lz;       // cartesian powers
};

//! Determine number of AO functions in a shell of momentum l=lx+ly+lz.
#define ANGL_FUNCS(l) ((l+1)*(l+2)/2)


void polarization_prim_pairs(// array of pairs of primitives
		  const PrimitivePair *pairs,
		  // number of pairs
		  int npair,
		  // output buffer, the required size of the buffer for all integrals
		  // can be determined with the help of `integral_buffer_size(...)`.
		  double *buffer,
		  // position of polarizable atom
		  double3 origin,
		  // operator    O(r) = x^mx y^my z^mz |r|^-k 
		  int k, int mx, int my, int mz,
		  // cutoff function F2(r) = (1 - exp(-alpha r^2))^q
		  double alpha,  int q
  /*
   AO polarization integrals for pairs of primitives

                          mx  my  mz
                        x'  y'  z'          - alpha r'^2  q
     buffer[ij] = <AO | ----------- (1 - exp             )   |AO  >
                     i      r'^k                                j


   with r' = r - origin. The integrals for the pair of primitives `pair = pairs[ipair]`
   starts at index `ij = pair.bufferIdx`. The number of integrals per pair depends on the
   angular momenta of the primitives. Since a primitive with angular momentum l 
   has `N(l)=(l+1)(l+2)/2)` Cartesian components, there are `N(prim.primA.l) * N(prim.primB.l)`
   integrals for each pair of primitives. The cartesian angular momentum components are 
   generated in the following order:
 
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

  */
			);


int integral_buffer_size(// array of pairs of primitives
			 const PrimitivePair *pairs,
			 // number of pairs in array
			 int npair
  /*
    compute the size of a buffer that can hold all the polarization integrals.
    Each pair of primitives needs a block of size `N(prim.primA.l) * N(prim.primB.l)`.
   */
			 );

class PolarizationIntegral {
  // center of basis functions for shell i
  double xi,yi,zi;
  // angular momentum of shell i
  int li;
  // exponent of Gaussians i
  double beta_i;
  // center of basis functions for shell j
  double xj,yj,zj;
  // angular momentum of shell j
  int lj;
  // exponent of Gaussians j
  double beta_j;
  // operator x^mx y^my z^mz |r|^{-k}
  int k, mx,my,mz;
  // cutoff function (1-exp(-alpha*r^2))^q
  double alpha;
  int q;

  //
  double bx,by,bz, b;

  int l_max;
  // k = 2*j or 2*j+1
  int j;
  // minimum and maximum value of s = lx+ly+lz - (zeta_x+zeta_y+zeta_z)/2
  int s_min, s_max;

  double *integs;
  double *f;

 public:
  // constructor
  __device__ PolarizationIntegral(
		       // unnormalized Cartesian Gaussian phi_i(r) = (x-xi)^nxi (y-yi)^nyi (z-zi)^nzi exp(-beta_i * (r-ri)^2), total angular momentum is li = nxi+nyi+nzi
		       double xi, double yi, double zi,    int li,  double beta_i,
		       // unnormalized Cartesian Gaussian phi_j(r) = (x-xj)^nxj (y-yj)^nyj (z-zj)^nzj exp(-beta_j * (r-rj)^2), the total angular momentum is lj = nxj+nyj+nzj
		       double xj, double yj, double zj,    int lj,  double beta_j,
		       // operator    O(r) = x^mx y^my z^mz |r|^-k 
		       int k,   int mx, int my, int mz,
		       // cutoff function F2(r) = (1 - exp(-alpha r^2))^q
		       double alpha, int q );

  // destructor
  __device__ ~PolarizationIntegral();

  // computes the integral between two unnormalized Cartesian Gaussian-type orbitals
  // belonging to the shells at the centers ri and rj with angular momenta li and lj.
  __device__ double compute_pair(int nxi, int nyi, int nzi,
				 int nxj, int nyj, int nzj);
  
};

#endif
