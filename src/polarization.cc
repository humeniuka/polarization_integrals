/*
Polarization Integrals for QM-MM-2e-Pol

This library provides and efficient implementation of the special functions defined in [CPP]

 -  d(p+1/2, x) and d(-p+1/2,x)         (eqns. 25 and 29)
 -  gamma(p+1/2,x)                      (eqn. 33)
 -  H(p,x)                              (eqns. 37 and 38)

and the polarization integrals

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



[CPP] P. Schwerdtfeger, H. Silberbach,
      "Multicenter integrals over long-range operators using Cartesian Gaussian functions",
      Phys. Rev. A 37, 2834
      https://doi.org/10.1103/PhysRevA.37.2834
[CPP-Erratum] Phys. Rev. A 42, 665
      https://doi.org/10.1103/PhysRevA.42.665

External Libraries
------------------
 -  Faddeeva package (http://ab-initio.mit.edu/Faddeeva.cc and http://ab-initio.mit.edu/Faddeeva.hh) 
*/
#include <cmath>
#include <cassert>
#include <iostream>

#include "Faddeeva.hh"
extern double Faddeeva::erf(double x);
extern double Faddeeva::Dawson(double x);

#include "polarization.h"

using namespace std;

void d_func(double x, int p_min, int p_max, double w0, double *d) {
  /*
    Arguments
    ---------
    x            : double >= 0
      upper limit of integration
    p_min, p_max : int, p_min <= 0, pmax >= 0
      defines range of values for integer p, p_min <= p <= p_max
    w0           : double or array with same shape as x
      To avoid overflow in the exponential function, exp(-w0) * d(p+1/2,x) is calculated.
    d            : pointer to allocated array of doubles of size |p_min|+p_max+1
      The integrals 
        d[p] = exp(-w0) * d(p+1/2,x) 
      are stored in this output array. The memory has to be allocated and
      released by the caller. 
      `d` should point to the pmin-th element of the array, so that the elements
      p=p_min,p_min+1,...,-1,0,1,...,p_max can be accessed as d[p]

    Details
    -------

    d_func(...) evaluates the integrals (eqns. (23) and (29) in Ref. [CPP] with eta=0)
    
                                   /x      p-1/2
    d(a,x) = d(p+1/2,x) = exp(-w0) |  dw  w      exp(w)
                                   /0

    for a = p+1/2 with p an integer.
    The prefactor exp(-w0) allows to avoid overflows in the exponential.
    
    The function values are generated iteratively for all integers p in the
    range p = p_min,p_min+1,..., 0 ,1,2,...,p_max 

    1) by upward iteration for p=0,1,2,...,p_max (a=1/2,3/2,...,p_max+1/2)

      Starting value (p=0)

          d[0] = d(1/2,x) = 2 exp(x-w0) dawson(sqrt(x))
      
      Iteration  (p -> p+1)

                    p+1/2
          d[p+1] = x      exp(x-w0)  - (p+1/2) d[p]

    2) and by downward iteration for p=0,-1,-2,...,p_min

      Iteration (-p -> -(p+1))
                           -(p+1/2)
          d[-(p+1)] = - ( x         exp(x-w0) - d[-p] ) / (p+1/2)

   */
  assert((p_min <= 0) && (p_max >= 0));
  int p;
  double xp, ixp;
  // constants during iteration
  double expx = exp(x-w0);
  double sqrtx = sqrt(x);
  double dwsn = Faddeeva::Dawson(sqrtx);
  
  // initialization p=0
  d[0] = 2*expx * dwsn;
  
  // 1) upward iteration starting from p=0
  xp = sqrtx * expx;
  for(p=0; p<p_max; p++) {
    d[p+1] = xp - (p+0.5)*d[p];
    // x^(p+1/2)  * exp(x-w0)
    xp *= x;
  }

  // 2) downward iteration starting from p=0
  ixp = 1/sqrtx * expx;
  for(p=0; p > p_min; p--) {
    d[p-1] = -(ixp - d[p])/(-p+0.5);
    // x^(-(p+1/2)) * exp(x-w0)
    ixp /= x;
  }
  
  // returns nothing, output is in array d
}

void d_func_zero_limit(double x, int p_min, int p_max, double w0, double *d) {
  /*
    The function \tilde{d} also computes d(p+1/2,x), however without the factor x^{p+1/2}:

      ~             p+1/2
      d(p+1/2,x) = x      d(p+1/2,x)          for all integers p

    This ensures that \tilde{d} has a finite value in the limit x -> 0.

    Arguments
    ---------
    x            : double >= 0
      upper limit of integration
    p_min, p_max : int, p_min <= 0, pmax >= 0
      defines range of values for integer p, p_min <= p <= p_max
    w0           : double or array with same shape as x
      To avoid overflow in the exponential function, exp(-w0) * \tilde{d}(p+1/2,x) is calculated.
    d            : pointer to allocated array of doubles of size |p_min|+p_max+1
      The integrals 
        d[p] = exp(-w0) * \tilde{d}(p+1/2,x) 
      are stored in this output array. The memory has to be allocated and
      released by the caller. 
      `d` should point to the pmin-th element of the array, so that the elements
      p=p_min,p_min+1,...,-1,0,1,...,p_max can be accessed as d[p]

   */
  assert((p_min <= 0) && (p_max >= 0));
  int p, k;
  double y;
  // constants during iteration
  double expx = exp(x-w0);

  // zero output array for indices p=0,...,p_max,
  // the memory locations pmin,...,-1 are overwritten, anyway.
  fill(d, d+p_max+1, 0.0);
  
  /*
   1) For p >= 0, \tilde{d} is calculated from the Taylor expansion around x=0.
    
          ~          inf     x^k
          d (x) = sum     ------------
           p         k=0  k! (p+k+1/2)
    
      The Taylor expansion is truncated at k_max = 20
  */
  const int k_max = 20;
  // y = x^k / k! * exp(-w0)
  y = exp(-w0);
  for(k=0; k < k_max; k++) {
    for(p=0; p <= p_max; p++) {
      d[p] += y/(p+k+0.5);
    }
    y *= x/(k+1);
  }

  /*
   2) For -p < 0, \tilde{d} is obtained by downward iteration starting from p=0
      according to the prescription
   
         ~              1       x        ~
         d       = - ------- ( e   -  x  d   )
          -(p+1)      p+1/2               -p
  */
  for(p=0; p < -p_min; p++) {
    d[-(p+1)] = - (expx - x*d[-p])/(p+0.5);
  }

  // returns nothing, output is in array d  
}
  
inline int kappa(int n) {
  if (n % 2 == 0) {
    return n/2;
  } else {
    return (n+1)/2;
  }
}

PolarizationIntegral::PolarizationIntegral(
		   // unnormalized Cartesian Gaussian phi_i(r) = (x-xi)^nxi (y-yi)^nyi (z-zi)^nzi exp(-beta_i * (r-ri)^2), total angular momentum is li = nxi+nyi+nzi
		   double xi_, double yi_, double zi_,    int li_,  double beta_i_,
		   // unnormalized Cartesian Gaussian phi_j(r) = (x-xj)^nxj (y-yj)^nyj (z-zj)^nzj exp(-beta_j * (r-rj)^2), the total angular momentum is lj = nxj+nyj+nzj
		   double xj_, double yj_, double zj_,    int lj_,  double beta_j_,
		   // operator    O(r) = x^mx y^my z^mz |r|^-k 
		   int k_,   int mx_, int my_, int mz_,
		   // cutoff function F2(r) = (1 - exp(-alpha r^2))^q
		   double alpha_, int q_ )
  // initialize member variable with arguments
  : xi{xi_}, yi{yi_}, zi{zi_}, li{li_}, beta_i{beta_i_},
    xj{xj_}, yj{yj_}, zj{zj_}, lj{lj_}, beta_j{beta_j_},
    k{k_}, mx{mx_}, my{my_}, mz{mz_},
    alpha{alpha_}, q{q_} {
  
  assert("Integrals are only implemented for the case k > 2!" && k > 2);
  // check that exponent of operator k and cutoff power q are compatible, otherwise the integrals
  // do not exist
  assert("Integrals do not exist for this combination of k and q!" && (q >= kappa(k) - kappa(mx) - kappa(my) - kappa(mz) - 1));
  
  // eqn. (15)
  bx = beta_i*xi + beta_j*xj;
  by = beta_i*yi + beta_j*yj;
  bz = beta_i*zi + beta_j*zj;
  b = sqrt(bx*bx+by*by+bz*bz);
  double ri2, rj2;
  ri2 = xi*xi+yi*yi+zi*zi;
  rj2 = xj*xj+yj*yj+zj*zj;

  l_max = li+lj + max(mx, max(my,mz));

  s_max = li+lj+mx+my+mz;
  j = k/2;

  /* Precalculate the factors
                   (2*i-1)!!
	   f[i] = ----------     for i=0,...,lmax
	             2^i
	 by the iteration

           f[0] = 1
           f[1] = 1/2
         f[i+1] = (i+1/2) f[i]
  */
  f = new double[l_max+1];
  
  f[0] = 1.0;
  f[1] = 0.5;
  int i;
  for (i=1; i < (l_max+1)/2; i++) {
    f[i+1] = (i+0.5)*f[i];
  }
  
  /* Precalculate unique integrals J
     The factor exp(-w0) = exp(-beta_i*ri^2 - beta_j*rj^2) is pulled into the integral 
  */
  double w0 = beta_i * ri2 + beta_j * rj2;
  // allocate zeroed memory 
  integs = new double[s_max+1]();
  double c = pow(M_PI,1.5)/tgamma(k/2.0);
  
  double b2, b_pow, b2jm3;
  b2 = b*b;
  b2jm3 = pow(b,2*j-3);
  /*
    outer loop is
    sum_(mu to q) binomial(q,mu) (-1)^mu
  */
  int mu, p_min, p_max, s;
  double a_mu, x, invx;
  double a_mu_jm32, a_mu_pow; 
  
  double test_binom_nu = 0.0;
  double binom_jm1_nu = 1.0;
  int nu;

  // threshold for switching to implementation for small x = b^2/a_mu
  const double x_small = 1.0e-2;
  
  p_min = min(0, -j+1);
  p_max = s_max;
  double *darr = new double[-p_min+p_max+1]();
  // Shift pointer to array element for p=0, so that
  // the indices of d can be positive or negative.
  double *d = (darr-p_min);
  
  double test_binom_mu = 0.0;
  double binom_q_mu = 1.0;
  for (mu=0; mu <= q; mu++) {
    // eqn. (15)
    a_mu = beta_i + beta_j + mu*alpha;
    x = b2/a_mu;

    if(x < x_small) {
      /* x = (b^2/a_mu) -> 0 limit */
      if (k % 2 == 0) {
	// Case 1: k=2*j
	/*
	                     q                    mu   -s+j-3/2     j-1                     nu
	   integs[s] = c  sum     binom(q,mu) (-1)    a          sum      binom(j-1,nu) (-1)    \tilde{d}(s-j+nu+1 + 1/2,x)
                             mu=0                      mu           nu=0                  
	*/
	a_mu_jm32 = pow(a_mu, j-1.5);

	// compute integrals \tilde{d}(p+1/2,x)
	d_func_zero_limit(x, p_min, p_max, w0, d);
	// array d contains \tilde{d}_p = x^{-p-1/2} d_p

	test_binom_nu = 0.0;
	binom_jm1_nu = 1.0;
	for(nu=0; nu <= j-1; nu++) {
	  // a_mu_pow = a_mu^{-s+j-3/2}
	  a_mu_pow = a_mu_jm32;
	  for (s=0; s<=s_max; s++) {
	    // eqn. (22)
	    integs[s] += c * binom_q_mu * binom_jm1_nu * a_mu_pow * d[s-j+nu+1];
	    // 
	    assert((s-j+nu+1 <= p_max) && (p_min <= s-j+nu+1 ));
	    assert(abs(a_mu_pow - pow(a_mu,-s+j-1.5))/abs(a_mu_pow) < 1.0e-10);

	    a_mu_pow /= a_mu;
	  }
	  test_binom_nu += binom_jm1_nu;
	  // update binomial coefficients for next iteration
	  //  B_{n,k+1} = x (n-k)/(k+1) B_{n,k}
	  binom_jm1_nu *= ((-1) * (j-1-nu))/(nu + 1.0);
	}
	assert(abs(test_binom_nu) < 1.0e-10);
      } else {
	// Case 2: k=2*j+1
	assert(1 == 2 && "not implemented yet!");
      }
    } else { // x > x_small
      invx = 1.0/x;
    
      if (k % 2 == 0) {
	// Case 1: k=2*j, eqn. (22)
	/*
	                     q                    mu   -2*s+2*j-3     j-1                     a   nu                   b^2
	   integs[s] = c  sum     binom(q,mu) (-1)    b            sum      binom(j-1,nu) (- --- )   d(s-j+nu+1 + 1/2, --- )
                             mu=0                                     nu=0                   b^2                        a
	*/
	// compute integrals d(p+1/2,x)
	d_func(x, p_min, p_max, w0, d);
	
	test_binom_nu = 0.0;
	binom_jm1_nu = 1.0;
	for(nu=0; nu <= j-1; nu++) {
	  b_pow = b2jm3;
	  for (s=0; s<=s_max; s++) {
	    // eqn. (22)
	    integs[s] += c * binom_q_mu * binom_jm1_nu * b_pow * d[s-j+nu+1];
	    
	    assert((s-j+nu+1 <= p_max) && (p_min <= s-j+nu+1 ));
	    assert(abs(b_pow - pow(b,-2*s+2*j-3))/abs(b_pow) < 1.0e-10);
	    
	    b_pow /= b2;
	  }
	  test_binom_nu += binom_jm1_nu;
	  // update binomial coefficients for next iteration
	  //  B_{n,k+1} = x (n-k)/(k+1) B_{n,k}
	  binom_jm1_nu *= ((-invx) * (j-1-nu))/(nu + 1.0);
	}
	assert(abs(test_binom_nu - pow(1 - invx, j-1)) < 1.0e-10);
      } else {
	assert(1 == 2 && "Case 2 (k=2*j+1) not implemented yet");
	// Case 2: k=2*j+1
	for (s=0; s<=s_max; s++) {
	  if (s-j >= 0) {
	    // Subcase 1: s-j >= 0, eqn. (32)
	    integs[s] = c;
	  } else {
	    // Subcase 2: s-j < 0, eqn. (39)
	    integs[s] = c;
	  }
	}
      }
    }
    test_binom_mu += binom_q_mu;
    // update binomial coefficient
    binom_q_mu *= ((-1)*(q-mu))/(mu + 1.0);
  } // end of loop over mu
  delete[] darr;

  // 0 = (1-1)^q = sum_{mu=0}^q binom(q,mu) (-1)^mu
  assert(abs(test_binom_mu) < 1.0e-10);
}

PolarizationIntegral::~PolarizationIntegral() {
  // release memory
  delete[] f;
  delete[] integs;
}

double PolarizationIntegral::compute_pair(int nxi, int nyi, int nzi,
					  int nxj, int nyj, int nzj) {

  int eta_xi, eta_xj, eta_yi, eta_yj, eta_zi, eta_zj;
  // binom_xi_pow = binomial(nxi,eta_xi) (-xi)^(nxi - eta_xi)
  double binom_xi_pow, binom_xj_pow, binom_yi_pow, binom_yj_pow, binom_zi_pow, binom_zj_pow;
  int lx, ly, lz;
  // products of binomial coefficients and powers of centers
  double fxx, fxxyy, fxxyyzz;

  // binom_bx_pow = binomial(lx,zeta_x) bx^(lx - zeta_x)
  double binom_bx_pow, binom_by_pow, binom_bz_pow;
  int zeta_x, zeta_y, zeta_z;
  // products of binomial coefficients and powers of centers
  double gxy, gxyz;
  // If zeta_x is even, then even_x = true
  bool even_x, even_y, even_z;

  // accumulates polarization integrals
  double op = 0.0;

  // Variables beginning with test_... are only needed for the consistency of the code
  // and can be removed later.
  double test_binomial_6, test_binomial_3;

  // maximum values for lx,ly,lz
  int lx_max, ly_max, lz_max, l_max_;
  // maximum value of s = lx+ly+lz - (zeta_x+zeta_y+zeta_z)/2
  int s, s_max_;

  assert("Total angular momentum for bra orbital differs from that used to create PolarizationIntegral instance!" && (nxi+nyi+nzi == li));
  assert("Total angular momentum for ket orbital differs from that used to create PolarizationIntegral instance!" && (nxj+nyj+nzj == lj));
  
  lx_max = nxi+nxj+mx;
  ly_max = nyi+nyj+my;
  lz_max = nzi+nzj+mz;
  l_max_ = max(lx_max, max(ly_max, lz_max));
  assert(l_max_ <= l_max);
  
  s_max_ = lx_max+ly_max+lz_max;
  assert(s_max_ <= s_max);
  
  // six nested for loops from binomial expansion of the cartesian basis functions (eqn. (9))
  test_binomial_6 = 0.0;
  // x-loop    
  binom_xi_pow = 1.0;
  for(eta_xi=nxi; eta_xi >= 0; eta_xi--) {
    binom_xj_pow = 1.0;
    for(eta_xj=nxj; eta_xj >= 0; eta_xj--) {
      lx = eta_xi + eta_xj + mx;
      fxx = binom_xi_pow * binom_xj_pow;
      // y-loop
      binom_yi_pow = 1.0;
      for(eta_yi=nyi; eta_yi >= 0; eta_yi--) {
	binom_yj_pow = 1.0;
	for(eta_yj=nyj; eta_yj >= 0; eta_yj--) {
	  ly = eta_yi + eta_yj + my;
	  fxxyy = fxx * binom_yi_pow * binom_yj_pow;
	  // z-loop
	  binom_zi_pow = 1.0;
	  for(eta_zi=nzi; eta_zi >= 0; eta_zi--) {
	    binom_zj_pow = 1.0;
	    for(eta_zj=nzj; eta_zj >= 0; eta_zj--) {
	      lz = eta_zi + eta_zj + mz;
	      fxxyyzz = fxxyy * binom_zi_pow * binom_zj_pow;

	      // The six for-loops calculate the expression
	      //  (1-xi)^nxi (1-yi)^nyi (1-zi)^nzi (1-xj)^nxj (1-yj)^nyj (1-zj)^nzj
	      // using the binomial expansion theorem.
	      test_binomial_6 += fxxyyzz;
	      
	      // three nested loops (eqn. (20))
	      test_binomial_3 = 0.0;
	      // bx-loop
	      binom_bx_pow = 1.0;
	      for(zeta_x=lx; zeta_x >= 0; zeta_x--) {
		even_x = (zeta_x % 2 == 0);
		// by-loop
		binom_by_pow = 1.0;
		for(zeta_y=ly; zeta_y >= 0; zeta_y--) {
		  even_y = (zeta_y % 2 == 0);
		  gxy = binom_bx_pow * binom_by_pow;
		  // bz-loop
		  binom_bz_pow = 1.0;
		  for(zeta_z=lz; zeta_z >= 0; zeta_z--) {
		    even_z = (zeta_z % 2 == 0);
		    gxyz = gxy * binom_bz_pow;

		    // The three for-loops calculate the expression
		    //  (1+bx)^lx (1+by)^ly (1+bz)^lz
		    // using the binomial expansion theorem.
		    test_binomial_3 += gxyz;

		    if (even_x && even_y && even_z) {
		      gxyz *= f[zeta_x/2] * f[zeta_y/2] * f[zeta_z/2];
		      
		      s = lx+ly+lz-(zeta_x+zeta_y+zeta_z)/2;
		      assert((0 <= s) && (s <= s_max));
		      
		      op += fxxyyzz * gxyz * integs[s];
		    }
		    
		    // update binomial coefficients for next iteration
		    //  B_{n,k-1} = x k / (n-k+1) B_{n,k}
		    binom_bz_pow *= (bz * zeta_z)/(lz - zeta_z + 1.0);
		  }
		  binom_by_pow *= (by * zeta_y)/(ly - zeta_y + 1.0);
		}
		binom_bx_pow *= (bx * zeta_x)/(lx - zeta_x + 1.0);
	      }
	      assert(abs(test_binomial_3 - pow(1+bx,lx)*pow(1+by,ly)*pow(1+bz,lz))/abs(test_binomial_3) < 1.0e-10);	      	      
	      // update binomial coefficients for next iteration
	      //  B_{n,k-1} = x k / (n-k+1) B_{n,k}
	      binom_zj_pow *= ((-zj) * eta_zj)/(nzj - eta_zj + 1.0);
	    }
	    binom_zi_pow *= ((-zi) * eta_zi)/(nzi - eta_zi + 1.0);
	  }
	  binom_yj_pow *= ((-yj) * eta_yj)/(nyj - eta_yj + 1.0);
	}
	binom_yi_pow *= ((-yi) * eta_yi)/(nyi - eta_yi + 1.0);
      }
      binom_xj_pow *= ((-xj) * eta_xj)/(nxj - eta_xj + 1.0);
    }
    binom_xi_pow *= ((-xi) * eta_xi)/(nxi - eta_xi + 1.0);
  }

  // consistency check
  assert(abs(test_binomial_6 - pow(1-xi,nxi)*pow(1-yi,nyi)*pow(1-zi,nzi) * pow(1-xj,nxj)*pow(1-yj,nyj)*pow(1-zj,nzj))/abs(test_binomial_6) < 1.0e-10);

  return op;
}

// for testing only
double test_d_func(double x, int p, double w0) {
  int p_min, p_max;
  double dret; 
  p_min = min(0, p);
  p_max = max(0, p);
  double *darr = new double[-p_min+p_max+1]();
  double *d = (darr-p_min);
  d_func(x, p_min, p_max, w0, d);

  dret = d[p];
  delete[] darr;
  
  return dret;
}

double test_d_func_zero_limit(double x, int p, double w0) {
  int p_min, p_max;
  double dret; 
  p_min = min(0, p);
  p_max = max(0, p);
  double *darr = new double[-p_min+p_max+1]();
  double *d = (darr-p_min);
  d_func_zero_limit(x, p_min, p_max, w0, d);

  dret = d[p];
  delete[] darr;
  
  return dret;
}
