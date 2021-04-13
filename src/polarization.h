
#ifndef _polarization_integrals_h
#define _polarization_integrals_h

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
  PolarizationIntegral(
		       // unnormalized Cartesian Gaussian phi_i(r) = (x-xi)^nxi (y-yi)^nyi (z-zi)^nzi exp(-beta_i * (r-ri)^2), total angular momentum is li = nxi+nyi+nzi
		       double xi, double yi, double zi,    int li,  double beta_i,
		       // unnormalized Cartesian Gaussian phi_j(r) = (x-xj)^nxj (y-yj)^nyj (z-zj)^nzj exp(-beta_j * (r-rj)^2), the total angular momentum is lj = nxj+nyj+nzj
		       double xj, double yj, double zj,    int lj,  double beta_j,
		       // operator    O(r) = x^mx y^my z^mz |r|^-k 
		       int k,   int mx, int my, int mz,
		       // cutoff function F2(r) = (1 - exp(-alpha r^2))^q
		       double alpha, int q );

  // destructor
  ~PolarizationIntegral();

  // computes the integral between two unnormalized Cartesian Gaussian-type orbitals
  // belonging to the shells at the centers ri and rj with angular momenta li and lj.
  double compute_pair(int nxi, int nyi, int nzi,
		      int nxj, int nyj, int nzj);
  
};

#endif
