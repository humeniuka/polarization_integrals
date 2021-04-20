/*
Polarization Integrals for QM-MM-2e-Pol

This library provides and efficient implementation of the special functions defined in [CPP]

 -  d(p+1/2, x) and d(-p+1/2,x)         (eqns. 25 and 29)
 -  gamma(p+1/2,x)                      (eqn. 33)
 -  H(p,x)                              (eqns. 37 and 38)

and the polarization integrals

             mx  my  mz
            x   y   z           - alpha r^2  q
  <CGTO   | ----------- (1 - exp            )   |CGTO  >
       i        r^k                                  j

between unnormalized primitive Cartesian Gaussian functions 

                        nxi       nyi       nzi                     2
   CGTO (x,y,z) = (x-xi)    (y-yi)    (z-zi)    exp(-beta_i (r - ri)  )
       i

and

                        nxj       nyj       nzj                     2
   CGTO (x,y,z) = (x-xj)    (y-yj)    (z-zj)    exp(-beta_j (r - rj)  )
       j

for k > 2. The power of the cutoff function q has to satisfy

  q >= kappa(k/2) - kappa(mx/2) - kappa(my/2) - kappa(mz/2) - 1

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
// uncomment the following line to turn off assertions 
#undef NDEBUG
//

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
    w0           : double
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
    x            : double
      upper limit of integration
    p_min, p_max : int, p_min <= 0, pmax >= 0
      defines range of values for integer p, p_min <= p <= p_max
    w0           : double
      To avoid overflow in the exponential function, exp(-w0) * \tilde{d}(p+1/2,x) is calculated.
    d            : pointer to allocated array of doubles of size |p_min|+p_max+1
      The integrals 
        d[p] = exp(-w0) * \tilde{d}(p+1/2,x) 
      are stored in this output array. The memory has to be allocated and
      released by the caller. 
      `d` should point to the p_min-th element of the array, so that the elements
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
    y *= x/(k+1.0);
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

void g_func(double x, int p_max, double *g) {
  /*
    evaluates the integral

                          /x      p-1/2
        gamma(p+1/2, x) = |  dw  w      exp(-w)
                          /0

    iteratively for all p=0,1,...,pmax. The closed-form expression for gamma is given in eqn. (33) in [CPP].

    Arguments
    ----------
    x    :  double >= 0
      upper integration limit
    pmax :  int >= 0
      integral is evaluated for all integers p=0,1,...,pmax
    g    :  pointer to first element of doubly array of size pmax+1
      The integrals gamma(p+1/2,x) are stored in g in the order p=0,1,...,pmax
  */
  assert(p_max >= 0);
  // constants during iteration
  double sqrtx = sqrt(x);
  double expmx = exp(-x);
  /* 
     The definition of the error function in eqn. (5) [CPP] differs from the usual definition
     (see https://en.wikipedia.org/wiki/Error_function ):

      - eqn. (5)             erf(x) = integral(exp(-t^2), t=0...z)
      - standard definition  erf(x) = 2/sqrt(pi) integral(exp(-t^2), t=0...z)
  */
  // sqrt(pi)
  const double SQRT_PI = 2.0/M_2_SQRTPI;
  
  // initialization p=0
  g[0] = SQRT_PI * erf(sqrtx);

  // upward iteration starting from p=0
  int p;
  // xp = x^(p+1/2) * exp(-x)
  double xp = sqrtx * expmx;
  for(p=0; p<p_max; p++) {
    g[p+1] = -xp + (p+0.5)*g[p];
    xp *= x;
  }

  // returns nothing, result is in memory location pointed to by g
}


double m_func(double x) {
  /*
    calculates

                      /x     t^2 
    m*(x) = exp(-x^2) | dt  e    erf(t)  =  exp(-x^2) m(x)
                      /0

  For x < 6, m*(x) is computed by piecewise Taylor expansions of m(x) 
  and for x > 6 by the approximation m*(x) = erf(x) dawson(x).

  Note that m*(x) differs by the inclusion of the factor exp(-x^2) from 
  the definition in Schwerdtfeger's article. This avoids overflows for 
  large values of x.
  */
  // return value m*(x)
  double m;
  if (x >= 6.0) {
    m = Faddeeva::erf(x) * Faddeeva::Dawson(x);
  } else {
    // hard-coded values m(xi) and m'(xi) at the expansion points
    const double x0s[] = { 0., 0.5, 1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5, 5., 5.5 };
    const double m0s[] = {
            0.0,
            0.1536288593561963,
            0.8153925207417932,
            3.230822260808236,
            15.47726567802307,
            114.4689886643230,
            1443.356844958733,
            31266.84178403753,
            1.149399290121673e6,
            7.107314602194292e7,
            7.354153746369724e9,
            1.269164846178781e12 };
    const double m1s[] = {
            0.0,
            0.6683350724948156,
            2.290698252303238,
            9.166150419904208,
            54.34275435683373,
            517.8020183042809,
            8102.904926424203,
            208981.1335760574,
            8.886110383508415e6,
            6.229644420759607e8,
            7.200489933727517e10,
            1.372170497746480e13 };

    // select the expansion point i if x0(i) <= x < x0(i+1)
    int i = (int) (2*x);
    double x0 = x0s[i];

    // Taylor expansion is truncated after n_max+1 terms
    const int n_max = 20;
    double m_deriv[n_max+1];
    m_deriv[0] = m0s[i];
    m_deriv[1] = m1s[i];
    m_deriv[2] = 2*x0*m_deriv[1] + M_2_SQRTPI; // M_2_SQRTPI = 2/sqrt(pi)

    /*
      compute derivatives of m(x) at the expansion point iteratively
                    d^k m(x)
       m_deriv[k] = -------- (x0)
                     d x^k
    */
    int n;
    for(n=3; n <= n_max; n++) {
      m_deriv[n] = 2*(n-2)*m_deriv[n-2] + 2*x0*m_deriv[n-1];
    }

    // evaluate Taylor series
    double dx = x-x0;
    // y holds (x-x0)^n / n!
    double y = dx;   
    // accumulator
    m = m_deriv[0];
    for(n=1; n <= n_max; n++) {
      m += y * m_deriv[n];
      y *= dx/(n+1);
    }

    // include factor exp(-x^2) to avoid overflows
    m *= exp(-x*x);
  }
  
  return m;
}

void h_func_large_x(double x, int p_min, int p_max, double h1_add, double *work, double *h) {
  /*
    compute H(p,x) for small x

    Arguments are the same as for `h_func(...)`.
  */
  assert((p_min <= 0) && (p_max >= 0));
  int p, k;
  double acc, y;
  
  // 1) upward recursion for positive p
  double sqrtx = sqrt(x);
  double expmx = exp(-x);
  // initial values for upward recursion:
  //   H(0,x) in eqn. (37b)
  h[0] = (1.0/sqrtx) * Faddeeva::erf(sqrtx) / M_2_SQRTPI;
  if (p_max > 0) {
    //   H(1,x) in eqn. (37c) with correction `h1_add * exp(-x) = -1/2 log(a_mu) exp(-x)`
    // NOTE: m*(x) is modified to include the factor exp(-x^2), therefore
    //       exp(-x) m(sqrt(x)) = m*(sqrt(x))
    h[1] = 2.0 / M_2_SQRTPI * m_func(sqrtx)  +  h1_add * expmx;
  }
  for(p=2; p <= p_max; p++) {
    // compute H(p,x) from H(p-1,x) and H(p-2,x)
    // eqn. (37a)
    h[p] = (2*(p+x)-3)/(2*(p-1)) * h[p-1] - x/(p-1) * h[p-2];
  }
  
  // 2) For negative p we need gamma(k+1/2,x) for k=0,1,...,|p_min|
  double *g = work;
  // compute gamma integrals
  g_func(x, -p_min, g);

  /*
    eqn. (38)
                                                             k 
     H(-p,x) = 1/(2 sqrt(x)) sum(k to p) binomial(p,k) (-1/x)   gamma(k+1/2,x)
             = 1/(2 sqrt(x)) sum(k to p) B_{p,k} g[k]
  */
  double invmx = -1/x;
  for(p=1; p <= -p_min; p++) {
    acc = 0.0;
    // y holds B_{p,k} = binomial(p,k) (-1/x)^k
    y = 1.0;
    for(k=0; k <= p; k++) {
      acc += y * g[k];
      // B_{p,k+1} = (-1/x) (p-k)/(k+1) B_{p,k}
      y *= invmx * (p-k)/(k+1.0);
    }
    acc *= 1.0/(2*sqrtx);

    h[-p] = acc;
  }
  // returns nothing, results are in output array h
}

void h_func_small_x(double x, int p_min, int p_max, double h1_add, double *work, double *h) {
  /*
    compute H(p,x) for small x

    Arguments are the same as for `h_func(...)`.
  */
  assert((p_min <= 0) && (p_max >= 0));
  /*
    1) upward recursion for positive p
    Taylor expansion of H(0,x) around x=0, which is truncated at kmax
    
                      kmax    (-2x)^k
      H(0,x) = 1 + sum     -------------
                      k=1  (2k+1) (2k)!!
  */
  int k, p;
  const int k_max = 20;
  double y, yk, h0;
  y = (-2*x);
  // yk = (-2x)^k / (2k)!!
  yk = y/2;
  h0 = 1.0;
  for(k=1; k<=k_max; k++) {
    h0 += yk/(2*k+1);
    yk *= y / (2*(k+1));
  }        
  // initial values for upward recursion
  h[0] = h0;

  // H(1,x), eqn. (37c)
  double sqrtx, expmx;
  if (p_max > 0) {
    sqrtx = sqrt(x);
    expmx = exp(-x);
    //   H(1,x) in eqn. (37c) with correction `h1_add * exp(-x) = -1/2 log(a_mu) exp(-x)`
    // NOTE: m*(x) is modified to include the factor exp(-x^2), therefore
    //       exp(-x) m(sqrt(x)) = m*(sqrt(x))
    h[1] = 2.0 / M_2_SQRTPI * m_func(sqrtx)  +  h1_add * expmx;
  }
  for(p=2; p <= p_max; p++) {
    // compute H(p,x) from H(p-1,x) and H(p-2,x)
    // eqn. (37a)
    h[p] = (2*(p+x)-3)/(2*(p-1)) * h[p-1] - x/(p-1) * h[p-2];
  }
  
  // 2) for negative p we need \tilde{d}(k+1/2,-x) for k=0,1,...,|pmin|
  double *d_tilde = work;
  d_func_zero_limit(-x, 0, -p_min, 0.0, d_tilde);
  /*
    For small x we rewrite eqn. (38) as
                      p
     H(-p,x) = 1/2 sum    binom(p,k) (-1)^k \tilde{d}(k+1/2,-x)
                      k=0
  */
  double acc;
  for(p=1; p <= -p_min; p++) {
    acc = 0.0;
    // y holds B_{p,k} = binomial(p,k) (-1)^k
    y = 1.0;
    for(k=0; k <= p; k++) {
      acc += y * d_tilde[k];
      // B_{p,k+1} = (-1) (p-k)/(k+1) B_{p,k}
      y *= (-1) * (p-k)/(k+1.0);
    }
    acc *= 0.5;
    h[-p] = acc;
  }
  // returns nothing, output is in array h
}

void h_func(double x, int p_min, int p_max, double h1_add, double *work, double *h) {
  /*
    evaluates the non-diverging part of the integral

               /1          -p
      H(p,x) = | dt (1-t^2)   exp(-x t^2) 
               /0

    according to the recursion relations in  eqns. (37) in [CPP] for p > 0 
    and eqn. (38) in [CPP] for p < 0 for all integers p in the range [pmin, pmax]

  ERRATUM:
    There is a mistake in eqn. (48) in [CPP], since `epsilon` is not independent of a_mu. 
    The correct limit is 

                   q                    mu     -1                       
         lim    sum     binom(q,mu) (-1)   tanh  (sqrt( eta/(a_mu+eta) ))
       eta->inf    mu=0
    
                    q                  mu
         = -1/2  sum   binom(q,mu) (-1)   log(a_u)
                    mu

    This error affects eqn. (37), the definition of H(1,x) has to be modified to

      H(1,x) =  exp(-x) {  sqrt(pi) m(sqrt(x))  - 1/2 * log(a_mu) }
      
   where 

      a_mu = beta_i + beta_j + mu*alpha.

   The correction term `h1_add = -1/2 log(a_mu)` can be passed into the function as the 4th argument.


    Arguments
    ---------
    x       :   double > 0
      second argument
    p_min    :  int <= 0
      lower limit for p
    p_max    :  int >= 0
      upper limit for p
    h1_add   :  double
      `h1_add=-1/2 log(a_mu)` is multiplied by exp(-x) and is added to H(1,x) 
      in order to correct a mistake in eqn. (37) in [CPP]
    work     : pointer to allocated array of doubles of size |p_min|+p_max+1
      temporary work space
      `work` should point to first element of the array.
    h        : pointer to allocated array of doubles of size |p_min|+p_max+1
      The integrals 
        h[p] = H(p,x)
      are stored in this output array. The memory has to be allocated and
      released by the caller. 
      `h` should point to the p_min-th element of the array, so that the elements
      p=p_min,p_min+1,...,-1,0,1,...,p_max can be accessed as h[p]
  */
  assert(x >= 0.0);
  // threshold for switching between two algorithms
  const double x_small = 1.5;
  if (x < x_small) {
    h_func_small_x(x, p_min, p_max, h1_add, work, h);
  } else {
    h_func_large_x(x, p_min, p_max, h1_add, work, h);
  }
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

  s_min = mx+my+mz;
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
  
  double w0 = beta_i * ri2 + beta_j * rj2;
  // allocate zeroed memory 
  integs = new double[s_max+1]();
  double c = pow(M_PI,1.5)/tgamma(k/2.0);
  
  double b2, b_pow, b2jm3;
  b2 = b*b;
  b2jm3 = pow(b,2*j-3);

  // Variables for case 2, subcase 1
  // binom(s-j,nu)
  double binom_smj_nu;
  
  /*
    outer loop is
    sum_(mu to q) binomial(q,mu) (-1)^mu
  */
  int mu, p_min, p_max, s;
  double a_mu, x, invx;
  double a_mu_jm32, a_mu_pow; 
  
  //double test_binom_nu;
  double binom_jm1_nu, binom_j_nu;
  int nu;

  // h1_add = -1/2 log(a_mu)
  double h1_add;
  
  // threshold for switching to implementation for small x = b^2/a_mu
  const double x_small = 1.0e-2;
  
  p_min = min(0, -j+1);
  p_max = s_max;
  double *darr = new double[-p_min+p_max+1]();
  // Shift pointer to array element for p=0, so that
  // the indices of d can be both positive or negative.
  double *d = (darr-p_min);
  // d and g are different names for the same memory location.
  // Depending on which case we are in, the array stores d(p+1/2,x) or g(p+1/2,x).
  double *g = (darr-p_min);

  double *harr = new double[s_max+j+1]();
  // h[p] has elements at positions p=-s_max,...,j.
  // Shift pointer to array element for p=0, so that
  // the indices of d can be both positive or negative.
  double *h = (harr-(-s_max));

  // temporary work space
  double *work = new double[max(-p_min+p_max+1,s_max+j+1)];

  /* 
     Precalculate unique integrals J
     The factor exp(-w0) = exp(-beta_i*ri^2 - beta_j*rj^2) is pulled into the integral 
  */
  
  //double test_binom_mu = 0.0;
  double binom_q_mu = 1.0;
  for (mu=0; mu <= q; mu++) {
    // eqn. (15)
    a_mu = beta_i + beta_j + mu*alpha;
    x = b2/a_mu;

    if(x < x_small) {
      /* x = (b^2/a_mu) -> 0 limit */
      if (k % 2 == 0) {
	//cout << "Case 1 (x < x_small)" << endl;
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

	//test_binom_nu = 0.0;
	binom_jm1_nu = 1.0;
	for(nu=0; nu <= j-1; nu++) {
	  // a_mu_pow = a_mu^{-s+j-3/2}
	  a_mu_pow = a_mu_jm32;
	  for (s=0; s<=s_max; s++) {
	    // eqn. (22)
	    integs[s] += c * binom_q_mu * binom_jm1_nu * a_mu_pow * d[s-j+nu+1];
	    // 
	    assert((s-j+nu+1 <= p_max) && (p_min <= s-j+nu+1 ));
	    //assert(abs(a_mu_pow - pow(a_mu,-s+j-1.5))/abs(a_mu_pow) < 1.0e-10);

	    a_mu_pow /= a_mu;
	  }
	  //test_binom_nu += binom_jm1_nu;
	  
	  // update binomial coefficients for next iteration
	  //  B_{n,k+1} = x (n-k)/(k+1) B_{n,k}
	  binom_jm1_nu *= ((-1) * (j-1-nu))/(nu + 1.0);
	}
	//assert(abs(test_binom_nu - pow(1-1,j-1)) < 1.0e-10);
      } else {
	// Case 2: k=2*j+1 and x < x_small
	assert(k == 2*j+1);

	double expx, expxmw0;
	if (s_max-j >= 0) {
	  /* 
	   compute integrals from Taylor expansion for small x

	     \tilde{g}(p+1/2, x) = x^{-p-1/2} gamma(p+1/2,x)
	                         = \tilde{d}(p+1/2, -x)

	  */
	  d_func_zero_limit(-x, 0, p_max, w0, d);
	  // Now d[p] = \tilde{d}(p+1/2,-x)

	  // the factor exp(-w0) is taken care of by d_func_zero_limit()
	  expx = exp(x);
	}
	if (s_min-j < 0) {
	  expxmw0 = exp(x-w0);
	  h1_add = -0.5 * log(a_mu);
	  // compute integrals H(p,x)
	  h_func(x, -s_max, j, h1_add, work, h);
	}
	// a_mu^{j-s-1}
	a_mu_pow = pow(a_mu, j-1);

	for (s=0; s<=s_max; s++) {
	  if (s < s_min) {
	    // The integrals for s < s_min are not needed, but we have to start the
	    // loop from s=0 to get the binomial coefficients right.
	  } else if (s-j >= 0) {
	    //cout << "Subcase 2a (x < x_small)" << endl;
	    // Subcase 2a: s-j >= 0, eqn. (32)
	    /*
	                       q                    mu   j-s-1 
	      integs[s] = c sum     binom(q,mu) (-1)    a      exp(x)
	                       mu=0                              
                                   
                                    s-j                    nu             
			         sum     binom(s-j,nu) (-1)    \tilde{d}(j+nu+1/2, -x)
                                    nu=0
	    */
	    //test_binom_nu = 0.0;
	    binom_smj_nu = 1.0;
	    for(nu=0; nu <= s-j; nu++) {
	      assert(j+nu <= s_max);
	      integs[s] += c * binom_q_mu * a_mu_pow * expx * binom_smj_nu * d[j+nu];

	      //test_binom_nu += binom_smj_nu;
	      //assert(abs(a_mu_pow - pow(a_mu, j-s-1)) < 1.0e-10);
	      
	      // update binomial coefficient
	      binom_smj_nu *= ((-1) * (s-j-nu))/(nu + 1.0);
	    }
	    //assert(abs(test_binom_nu - pow(1-1,s-j)) < 1.0e-10);
	  } else {
	    assert(s-j < 0);
	    assert(q >= j-s); 
	    //cout << "Subcase 2b (x < x_small)" << endl;
	    // Subcase 2b: s-j < 0 for x < x_small, eqn. (39)
	    /*
	                         q                    mu  j-s-1
	      integs[s] = 2 c sum     binom(q,mu) (-1)   a       exp(x - w0)
	                         mu=0

                                     j                    nu
				  sum     binom(j,nu) (-1)   H(j-s-nu, x)
				     nu=0
	    */
	    //test_binom_nu = 0.0;
	    binom_j_nu = 1.0;
	    for(nu=0; nu <= j; nu++) {
	      assert((-s_max <= j-s-nu) && (j-s-nu <= j));

	      integs[s] += 2 * c * binom_q_mu * a_mu_pow * expxmw0 * binom_j_nu * h[j-s-nu];
	      
	      //test_binom_nu += binom_j_nu;
	      //assert(abs(a_mu_pow - pow(a_mu, j-s-1)) < 1.0e-10);
	      
	      // update binomial coefficient
	      binom_j_nu *= ((-1) * (j-nu))/(nu + 1.0);
	    }
	    //assert(abs(test_binom_nu - pow(1-1, j)) < 1.0e-10);
	  }

	  a_mu_pow /= a_mu;
	} // end loop over s
      }
    } else {
      /* x > x_small */
      invx = 1.0/x;
    
      if (k % 2 == 0) {
	//cout << "Case 1 (x >= x_small)" << endl;
	// Case 1: k=2*j, eqn. (22)
	/*
	                     q                    mu   -2*s+2*j-3     j-1                     a   nu                   b^2
	   integs[s] = c  sum     binom(q,mu) (-1)    b            sum      binom(j-1,nu) (- --- )   d(s-j+nu+1 + 1/2, --- )
                             mu=0                                     nu=0                   b^2                        a
	*/
	// compute integrals d(p+1/2,x)
	d_func(x, p_min, p_max, w0, d);
	
	//test_binom_nu = 0.0;
	binom_jm1_nu = 1.0;
	for(nu=0; nu <= j-1; nu++) {
	  b_pow = b2jm3;
	  for (s=0; s<=s_max; s++) {
	    // eqn. (22)
	    integs[s] += c * binom_q_mu * binom_jm1_nu * b_pow * d[s-j+nu+1];
	    
	    assert((s-j+nu+1 <= p_max) && (p_min <= s-j+nu+1 ));
	    //assert(abs(b_pow - pow(b,-2*s+2*j-3))/abs(b_pow) < 1.0e-10);
	    
	    b_pow /= b2;
	  }
	  //test_binom_nu += binom_jm1_nu;
	  
	  // update binomial coefficients for next iteration
	  //  B_{n,k+1} = x (n-k)/(k+1) B_{n,k}
	  binom_jm1_nu *= ((-invx) * (j-1-nu))/(nu + 1.0);
	}
	//assert(abs(test_binom_nu - pow(1 - invx, j-1)) < 1.0e-10);
      } else {
	// Case 2: k=2*j+1
	assert(k == 2*j+1);

	if (s_max-j >= 0) {
	  // compute integrals gamma(p+1/2,x)
	  g_func(x, p_max, g);
	}
	if (s_min-j < 0) {
	  h1_add = -0.5 * log(a_mu);
	  // compute integrals H(p,x)
	  h_func(x, -s_max, j, h1_add, work, h);
	}
	double expxmw0 = exp(x-w0);
	double powinvx_expxmw0 = pow(invx, j+0.5) * expxmw0;
	// a_mu^{j-s-1}
	a_mu_pow = pow(a_mu, j-1);

	for (s=0; s<=s_max; s++) {
	  if (s < s_min) {
	    // The integrals for s < s_min are not needed, but we have to start the
	    // loop from s=0 to get the binomial coefficients right.
	  } else if (s-j >= 0) {
	    //cout << "Subcase 2a (x >= x_small)" << endl;
	    // Subcase 2a: s-j >= 0, eqn. (32)
	    /*
	                       q                    mu   j-s-1  -j-1/2
	      integs[s] = c sum     binom(q,mu) (-1)    a      x       exp(x - w0)
	                       mu=0                              
                                   
                                    s-j                       nu             
			         sum     binom(s-j,nu) (- 1/x)    g(j+nu+1/2, x)
                                    nu=0
	    */
	    //test_binom_nu = 0.0;
	    binom_smj_nu = 1.0;
	    for(nu=0; nu <= s-j; nu++) {
	      assert(j+nu <= s_max);
	      integs[s] += c * binom_q_mu * a_mu_pow * powinvx_expxmw0 * binom_smj_nu * g[j+nu];

	      //test_binom_nu += binom_smj_nu;
	      //assert(abs(a_mu_pow - pow(a_mu, j-s-1)) < 1.0e-10);
	      
	      // update binomial coefficient
	      binom_smj_nu *= ((-invx) * (s-j-nu))/(nu + 1.0);
	    }
	    //assert(abs(test_binom_nu - pow(1-invx, s-j)) < 1.0e-10);
	  } else {
	    assert(s-j < 0);
	    assert(q >= j-s); 
	    //cout << "Subcase 2b (x >= x_small)" << endl;
	    // Subcase 2b: s-j < 0, eqn. (39)
	    /*
	                         q                    mu  j-s-1
	      integs[s] = 2 c sum     binom(q,mu) (-1)   a       exp(x - w0)
	                         mu=0

                                     j                    nu
				  sum     binom(j,nu) (-1)   H(j-s-nu, x)
				     nu=0
	    */
	    //test_binom_nu = 0.0;
	    binom_j_nu = 1.0;
	    for(nu=0; nu <= j; nu++) {
	      assert((-s_max <= j-s-nu) && (j-s-nu <= j));

	      integs[s] += 2 * c * binom_q_mu * a_mu_pow * expxmw0 * binom_j_nu * h[j-s-nu];
	      
	      //test_binom_nu += binom_j_nu;
	      //assert(abs(a_mu_pow - pow(a_mu, j-s-1)) < 1.0e-10);
	      
	      // update binomial coefficient
	      binom_j_nu *= ((-1) * (j-nu))/(nu + 1.0);
	    }
	    //assert(abs(test_binom_nu - pow(1-1, j)) < 1.0e-10);
	  }
	  a_mu_pow /= a_mu;
	} // end loop over s
      }
    }
    //test_binom_mu += binom_q_mu;
    
    // update binomial coefficient
    binom_q_mu *= ((-1)*(q-mu))/(mu + 1.0);
  } // end of loop over mu

  // release memory
  delete[] darr;
  delete[] harr;
  delete[] work;

  // 0 = (1-1)^q = sum_{mu=0}^q binom(q,mu) (-1)^mu
  //assert(abs(test_binom_mu - pow(1-1,q)) < 1.0e-10);
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
  //double test_binomial_6, test_binomial_3;

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
  //test_binomial_6 = 0.0;
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
	      //test_binomial_6 += fxxyyzz;
	      
	      // three nested loops (eqn. (20))
	      
	      //test_binomial_3 = 0.0;
	      
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
		    //test_binomial_3 += gxyz;

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
	      //assert(abs(test_binomial_3 - pow(1+bx,lx)*pow(1+by,ly)*pow(1+bz,lz))/abs(test_binomial_3) < 1.0e-10);	      	      
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
  //assert(abs(test_binomial_6 - pow(1-xi,nxi)*pow(1-yi,nyi)*pow(1-zi,nzi) * pow(1-xj,nxj)*pow(1-yj,nyj)*pow(1-zj,nzj))/abs(test_binomial_6) < 1.0e-10);

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

double test_g_func(double x, int p) {
  int p_max = max(0, p);
  double *g = new double[p_max+1]();
  g_func(x, p_max, g);

  double gret = g[p];
  delete[] g;
  
  return gret;
}

double test_h_func(double x, int p) {
  /*
    compute H(p,x) for large x
   */
  int p_min, p_max;
  double hret; 
  p_min = min(0, p);
  p_max = max(0, p);
  // output array
  double *harr = new double[-p_min+p_max+1]();
  // shift pointer to p=0 element
  double *h = (harr-p_min);
  // temporary work space
  double *work = new double[-p_min+p_max+1];
  
  h_func(x, p_min, p_max, 0.0, work, h);
  
  hret = h[p];

  delete[] harr;
  delete[] work;
  
  return hret;
}

double test_h_func_large_x(double x, int p) {
  /*
    compute H(p,x) for large x
   */
  int p_min, p_max;
  double hret; 
  p_min = min(0, p);
  p_max = max(0, p);
  // output array
  double *harr = new double[-p_min+p_max+1]();
  // shift pointer to p=0 element
  double *h = (harr-p_min);
  // temporary work space
  double *work = new double[-p_min+p_max+1];
  
  h_func_large_x(x, p_min, p_max, 0.0, work, h);
  
  hret = h[p];

  delete[] harr;
  delete[] work;
  
  return hret;
}

double test_h_func_small_x(double x, int p) {
  /*
    compute H(p,x) for small x
   */
  int p_min, p_max;
  double hret; 
  p_min = min(0, p);
  p_max = max(0, p);
  // output array
  double *harr = new double[-p_min+p_max+1]();
  // shift pointer to p=0 element
  double *h = (harr-p_min);
  // temporary work space
  double *work = new double[-p_min+p_max+1];
  
  h_func_small_x(x, p_min, p_max, 0.0, work, h);
  
  hret = h[p];

  delete[] harr;
  delete[] work;
  
  return hret;
}
