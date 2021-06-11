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

Author
------
Alexander Humeniuk  (alexander.humeniuk@gmail.com)

*/
//

//////////////////// INCLUDES //////////////////////////////////////

#include <stdio.h>
#include <stdexcept>
#include <type_traits> // for std::is_same

#include <cuda.h>
#include <cuda_runtime.h>

#include "polarization.h"
// Dawson function from Faddeeva library
#include "Dawson_real.cu"

//////////////////// MACROS FOR DEBUGGING /////////////////////////

#if defined(DEBUG) || defined(_DEBUG)

// macro for printing
#define print(...) {          \
    printf(__VA_ARGS__);      \
}
// macro for assertions
//   assert(condition)
#undef assert
#define assert(condition) {					                           \
  if (!(condition)) {                                                                      \
    printf("*** Assertion failed at line number %d in file %s ***\n", __LINE__, __FILE__); \
  }                                                                                        \
}

#else // DEBUG
// silence debugging information
#define print(format, ...) {}
#undef assert
#define assert(condition) {}

#endif // DEBUG

// It is not possible to abort the execution from within the kernel. All we can do 
// is print an error message.
#define BUG(...) {                                                               \
    printf(__VA_ARGS__);                                                         \
    printf("\n");                                                                \
    printf("BUG at line number %d in file %s\n", __LINE__, __FILE__);            \
}

// Check if there has been a CUDA error.
// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
inline
cudaError_t checkCuda(cudaError_t err)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s, file %s\n", cudaGetErrorString(err), __FILE__);
    assert(err == cudaSuccess);
  }
#endif
  return err;
}

/////////////////// SPECIAL FUNCTIONS //////////////////////////////

template <typename real>
__device__ void d_func(real x, const int p_min, const int p_max, real w0, real *d) {
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
  real xp, ixp;
  // constants during iteration
  real expx = exp(x-w0);
  real sqrtx = sqrt(x);
  real dwsn = Dawson<real>(sqrtx);
  
  // initialization p=0
  d[0] = 2*expx * dwsn;
  
  // 1) upward iteration starting from p=0
  xp = sqrtx * expx;
  for(p=0; p<p_max; p++) {
    d[p+1] = xp - (p+((real) 0.5))*d[p];
    // x^(p+1/2)  * exp(x-w0)
    xp *= x;
  }

  // 2) downward iteration starting from p=0
  ixp = 1/sqrtx * expx;
  for(p=0; p > p_min; p--) {
    d[p-1] = -(ixp - d[p])/(-p+((real) 0.5));
    // x^(-(p+1/2)) * exp(x-w0)
    ixp /= x;
  }
  
  // returns nothing, output is in array d
}

template <typename real>
__device__ void d_func_zero_limit(real x, const int p_min, const int p_max, real w0, real *d) {
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
  real y;
  // constants during iteration
  real expx = exp(x-w0);

  // zero output array for indices p=0,...,p_max,
  // the memory locations pmin,...,-1 are overwritten, anyway.
  //fill(d, d+p_max+1, 0.0);
  for(p=0; p <= p_max; p++) {
    d[p] = 0.0;
  }
  
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
      d[p] += y/(p+k+((real) 0.5));
    }
    y *= x/(k+((real) 1.0));
  }

  /*
   2) For -p < 0, \tilde{d} is obtained by downward iteration starting from p=0
      according to the prescription
   
         ~              1       x        ~
         d       = - ------- ( e   -  x  d   )
          -(p+1)      p+1/2               -p
  */
  for(p=0; p < -p_min; p++) {
    d[-(p+1)] = - (expx - x*d[-p])/(p+((real) 0.5));
  }
  // returns nothing, output is in array d  
}

template <typename real>
__device__ void g_func(real x, const int p_max, real *g) {
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
  real sqrtx = sqrt(x);
  real expmx = exp(-x);
  /* 
     The definition of the error function in eqn. (5) [CPP] differs from the usual definition
     (see https://en.wikipedia.org/wiki/Error_function ):

      - eqn. (5)             erf(x) = integral(exp(-t^2), t=0...z)
      - standard definition  erf(x) = 2/sqrt(pi) integral(exp(-t^2), t=0...z)
  */
  // sqrt(pi)
  const real SQRT_PI = (real) (2.0/M_2_SQRTPI);
  
  // initialization p=0
  g[0] = SQRT_PI * erf(sqrtx);

  // upward iteration starting from p=0
  int p;
  // xp = x^(p+1/2) * exp(-x)
  real xp = sqrtx * expmx;
  for(p=0; p<p_max; p++) {
    g[p+1] = -xp + (p+((real) 0.5))*g[p];
    xp *= x;
  }

  // returns nothing, result is in memory location pointed to by g
}

// constants needed by m_func(...)
//
// hard-coded values m(xi) and m'(xi) at the expansion points
// ... with double precision ...
__constant__  const double x0s_dp[] = { 0., 0.5, 1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5, 5., 5.5 };
__constant__  const double m0s_dp[] = {
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
__constant__ const double m1s_dp[] = {
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
// ... and with single precision ...
__constant__  const float x0s_sp[] = { 0., 0.5, 1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5, 5., 5.5 };
__constant__  const float m0s_sp[] = {
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
__constant__ const float m1s_sp[] = {
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


template <typename real>
__device__ real m_func(real x) {
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
  real m;
  if (x >= (real) 6.0) {
    m = erf(x) * Dawson<real>(x);
  } else {
    // select the expansion point i if x0(i) <= x < x0(i+1)
    int i = (int) (2*x);
    real x0;
    // load constants with different precision based on data type (one of the
    // branches should be eliminated at compile time.)
    if (std::is_same<real, float>::value) {
      x0 = x0s_sp[i];
    } else {
      x0 = x0s_dp[i];
    }


    // Taylor expansion is truncated after n_max+1 terms
    const int n_max = 20;
    real m_deriv[n_max+1];
    // load constants with different precision based on data type (one of the
    // branches should be eliminated at compile time.)
    if (std::is_same<real, float>::value) {
      m_deriv[0] = m0s_sp[i];
      m_deriv[1] = m1s_sp[i];
    } else {
      m_deriv[0] = m0s_dp[i];
      m_deriv[1] = m1s_dp[i];
    }
    m_deriv[2] = 2*x0*m_deriv[1] + ((real) M_2_SQRTPI); // M_2_SQRTPI = 2/sqrt(pi)

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
    real dx = x-x0;
    // y holds (x-x0)^n / n!
    real y = dx;   
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

template <typename real>
__device__ void h_func_large_x(real x, const int p_min, const int p_max, real h1_add, real *work, real *h) {
  /*
    compute H(p,x) for small x

    Arguments are the same as for `h_func(...)`.
  */
  assert((p_min <= 0) && (p_max >= 0));
  int p, k;
  real acc, y;
  
  // 1) upward recursion for positive p
  real sqrtx = sqrt(x);
  real expmx = exp(-x);
  // initial values for upward recursion:
  //   H(0,x) in eqn. (37b)
  h[0] = (1/sqrtx) * erf(sqrtx) / ((real) M_2_SQRTPI);
  if (p_max > 0) {
    //   H(1,x) in eqn. (37c) with correction `h1_add * exp(-x) = -1/2 log(a_mu) exp(-x)`
    // NOTE: m*(x) is modified to include the factor exp(-x^2), therefore
    //       exp(-x) m(sqrt(x)) = m*(sqrt(x))
    h[1] = ((real) (2.0 / M_2_SQRTPI)) * m_func<real>(sqrtx)  +  h1_add * expmx;
  }
  for(p=2; p <= p_max; p++) {
    // compute H(p,x) from H(p-1,x) and H(p-2,x)
    // eqn. (37a)
    h[p] = (2*(p+x)-3)/(2*(p-1)) * h[p-1] - x/(p-1) * h[p-2];
  }
  
  // 2) For negative p we need gamma(k+1/2,x) for k=0,1,...,|p_min|
  real *g = work;
  // compute gamma integrals
  g_func<real>(x, -p_min, g);

  /*
    eqn. (38)
                                                             k 
     H(-p,x) = 1/(2 sqrt(x)) sum(k to p) binomial(p,k) (-1/x)   gamma(k+1/2,x)
             = 1/(2 sqrt(x)) sum(k to p) B_{p,k} g[k]
  */
  real invmx = -1/x;
  for(p=1; p <= -p_min; p++) {
    acc = 0.0;
    // y holds B_{p,k} = binomial(p,k) (-1/x)^k
    y = 1.0;
    for(k=0; k <= p; k++) {
      acc += y * g[k];
      // B_{p,k+1} = (-1/x) (p-k)/(k+1) B_{p,k}
      y *= invmx * (p-k)/(k+((real) 1.0));
    }
    acc *= 1/(2*sqrtx);

    h[-p] = acc;
  }
  // returns nothing, results are in output array h
}

template <typename real>
__device__ void h_func_small_x(real x, const int p_min, const int p_max, real h1_add, real *work, real *h) {
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
  real y, yk, h0;
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
  real sqrtx, expmx;
  if (p_max > 0) {
    sqrtx = sqrt(x);
    expmx = exp(-x);
    //   H(1,x) in eqn. (37c) with correction `h1_add * exp(-x) = -1/2 log(a_mu) exp(-x)`
    // NOTE: m*(x) is modified to include the factor exp(-x^2), therefore
    //       exp(-x) m(sqrt(x)) = m*(sqrt(x))
    h[1] = ((real) (2.0 / M_2_SQRTPI)) * m_func<real>(sqrtx)  +  h1_add * expmx;
  }
  for(p=2; p <= p_max; p++) {
    // compute H(p,x) from H(p-1,x) and H(p-2,x)
    // eqn. (37a)
    h[p] = (2*(p+x)-3)/(2*(p-1)) * h[p-1] - x/(p-1) * h[p-2];
  }
  
  // 2) for negative p we need \tilde{d}(k+1/2,-x) for k=0,1,...,|pmin|
  real *d_tilde = work;
  d_func_zero_limit<real>(-x, 0, -p_min, (real) 0.0, d_tilde);
  /*
    For small x we rewrite eqn. (38) as
                      p
     H(-p,x) = 1/2 sum    binom(p,k) (-1)^k \tilde{d}(k+1/2,-x)
                      k=0
  */
  real acc;
  for(p=1; p <= -p_min; p++) {
    acc = (real) 0.0;
    // y holds B_{p,k} = binomial(p,k) (-1)^k
    y = 1.0;
    for(k=0; k <= p; k++) {
      acc += y * d_tilde[k];
      // B_{p,k+1} = (-1) (p-k)/(k+1) B_{p,k}
      y *= (-1) * (p-k)/(k+((real) 1.0));
    }
    acc *= (real) 0.5;
    h[-p] = acc;
  }
  // returns nothing, output is in array h
}

template <typename real>
__device__ void h_func(real x, const int p_min, const int p_max, real h1_add, real *work, real *h) {
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
  const real x_small = 1.5;
  if (x < x_small) {
    h_func_small_x<real>(x, p_min, p_max, h1_add, work, h);
  } else {
    h_func_large_x<real>(x, p_min, p_max, h1_add, work, h);
  }
}

////////////////// POLARIZATION INTEGRALS /////////////////////////

template <typename real>
inline __device__ real power(real x, int n) {
  /*
    x^n  for n >= -1
   */
  assert(n >= -1);
  switch (n) {
  case -1:
    return ((real) 1.0) / x;
  case 0: 
    return (real) 1.0;
  case 1:
    return x;
  case 2:
    return x*x;
  case 3:
    return x*x*x;
  default:
    real p = 1.0;
    for(int i = 0; i < n; i++) {
      p *= x;
    }
    return p;
  }
}
template <typename real,
    // total angular momentum of bra, li = nxi+nyi+nzi, and ket, lj = nxj+nyj+nzj
    int li, int lj,
    // operator    O(r) = x^mx y^my z^mz |r|^-k 
    int k,   int mx, int my, int mz,
    // power of cutoff function F2(r) = (1 - exp(-alpha r^2))^q
    int q  >
__device__ PolarizationIntegral<real, li, lj, k, mx, my, mz, q>::PolarizationIntegral(
		   // unnormalized Cartesian Gaussian phi_i(r) = (x-xi)^nxi (y-yi)^nyi (z-zi)^nzi exp(-beta_i * (r-ri)^2), total angular momentum is li = nxi+nyi+nzi
		   real xi_, real yi_, real zi_,  real beta_i_,
		   // unnormalized Cartesian Gaussian phi_j(r) = (x-xj)^nxj (y-yj)^nyj (z-zj)^nzj exp(-beta_j * (r-rj)^2), the total angular momentum is lj = nxj+nyj+nzj
		   real xj_, real yj_, real zj_,  real beta_j_,
		   // exponent of cutoff function F2(r) = (1 - exp(-alpha r^2))^q
		   real alpha_) : 
		      // member initializer list
		      xi(xi_), yi(yi_), zi(zi_), beta_i(beta_i_),
		      xj(xj_), yj(yj_), zj(zj_), beta_j(beta_j_),
		      alpha(alpha_),
		      l_max(L_MAX), s_min(S_MIN), s_max(S_MAX),
		      j(J)
{
  /*
  // initialize member variable with arguments
  xi = xi_; yi = yi_; zi = zi_;
  beta_i = beta_i_;
  xj = xj_; yj = yj_; zj= zj_;
  beta_j = beta_j_;
  alpha = alpha_;
  */
  print("PolarizationIntegral::PolarizationIntegral\n");

  // eqn. (15)
  bx = beta_i*xi + beta_j*xj;
  by = beta_i*yi + beta_j*yj;
  bz = beta_i*zi + beta_j*zj;
  b = sqrt(bx*bx+by*by+bz*bz);
  real ri2, rj2;
  ri2 = xi*xi+yi*yi+zi*zi;
  rj2 = xj*xj+yj*yj+zj*zj;

  // The following definitions have been moved to the member initializer list
  //l_max = li+lj + max(mx, max(my,mz));   
  //s_min = mx+my+mz;
  //s_max = li+lj+mx+my+mz;
  //j = k/2;

  /* Precalculate the factors
                   (2*i-1)!!
	   f[i] = ----------     for i=0,...,lmax
	             2^i
	 by the iteration

           f[0] = 1
           f[1] = 1/2
         f[i+1] = (i+1/2) f[i]
  */  
  f[0] = 1.0;
  f[1] = 0.5;
  int i;
  for (i=1; i < (l_max+1)/2; i++) {
    f[i+1] = (i+((real) 0.5))*f[i];
  }
  
  real w0 = beta_i * ri2 + beta_j * rj2;
  // memset is not needed, integs is automatically initialized to 0.
  //memset(integs, 0.0, (s_max+1)*sizeof(real));

  real c = ((real) pow((real) M_PI,(real) 1.5))/tgamma(k/((real) 2.0));
  
  real b2, b_pow, b2jm3;
  b2 = b*b;
  b2jm3 = power(b,2*j-3);

  // Variables for case 2, subcase 1
  // binom(s-j,nu)
  real binom_smj_nu;
  
  /*
    outer loop is
    sum_(mu to q) binomial(q,mu) (-1)^mu
  */
  int mu, s;
  const int p_min = P_MIN;
  const int p_max = P_MAX;
  real a_mu, x, invx;
  real a_mu_jm32, a_mu_pow; 
  
  //real test_binom_nu;
  real binom_jm1_nu, binom_j_nu;
  int nu;

  // h1_add = -1/2 log(a_mu)
  real h1_add;
  
  // threshold for switching to implementation for small x = b^2/a_mu
  const real x_small = 1.0e-2;
  
  real darr[-p_min+p_max+1];
  // memset is not needed because array is automatically initialized to 0
  //memset(darr, 0.0, (-p_min+p_max+1)*sizeof(real));
  // Shift pointer to array element for p=0, so that
  // the indices of d can be both positive or negative.
  real *d = (darr-p_min);
  // d and g are different names for the same memory location.
  // Depending on which case we are in, the array stores d(p+1/2,x) or g(p+1/2,x).
  real *g = (darr-p_min);

  real harr[S_MAX+J+1];
  // memset is not needed because array is automatically initialized to 0.
  //memset(harr, 0.0, (s_max+j+1)*sizeof(real));
  // h[p] has elements at positions p=-s_max,...,j.
  // Shift pointer to array element for p=0, so that
  // the indices of d can be both positive or negative.
  real *h = (harr-(-s_max));

  // temporary work space
  real work[Max(-P_MIN+P_MAX+1, S_MAX+J+1)];

  /* 
     Precalculate unique integrals J
     The factor exp(-w0) = exp(-beta_i*ri^2 - beta_j*rj^2) is pulled into the integral 
  */

  //real test_binom_mu = 0.0;
  real binom_q_mu = 1.0;
  for (mu=0; mu <= q; mu++) {
    // eqn. (15)
    a_mu = beta_i + beta_j + mu*alpha;
    x = b2/a_mu;

    if(x < x_small) {
      print("x < x_small\n");
      /* x = (b^2/a_mu) -> 0 limit */
      if (k % 2 == 0) {
	print("Case 1: k=2*j and x < x_small\n");
	// Case 1: k=2*j
	/*
	                     q                    mu   -s+j-3/2     j-1                     nu
	   integs[s] = c  sum     binom(q,mu) (-1)    a          sum      binom(j-1,nu) (-1)    \tilde{d}(s-j+nu+1 + 1/2,x)
                             mu=0                      mu           nu=0                  
	*/
	a_mu_jm32 = power(a_mu, j-1)/sqrt(a_mu);

	// compute integrals \tilde{d}(p+1/2,x)
	d_func_zero_limit<real>(x, p_min, p_max, w0, d);
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
	  binom_jm1_nu *= ((-1) * (j-1-nu))/(nu + ((real) 1.0));
	}
	//assert(abs(test_binom_nu - pow(1-1,j-1)) < 1.0e-10);
      } else {
	// Case 2: k=2*j+1 and x < x_small
	print("Case 2: k=2*j+1 and x < x_small\n");
	assert(k == 2*j+1);

	real expx, expxmw0;
	if (s_max-j >= 0) {
	  /* 
	   compute integrals from Taylor expansion for small x

	     \tilde{g}(p+1/2, x) = x^{-p-1/2} gamma(p+1/2,x)
	                         = \tilde{d}(p+1/2, -x)

	  */
	  d_func_zero_limit<real>(-x, 0, p_max, w0, d);
	  // Now d[p] = \tilde{d}(p+1/2,-x)

	  // the factor exp(-w0) is taken care of by d_func_zero_limit()
	  expx = exp(x);
	}
	if (s_min-j < 0) {
	  expxmw0 = exp(x-w0);
	  h1_add = ((real) -0.5) * log(a_mu);
	  // compute integrals H(p,x)
	  h_func<real>(x, -s_max, j, h1_add, work, h);
	}
	// a_mu^{j-s-1}
	a_mu_pow = power(a_mu, j-1);

	for (s=0; s<=s_max; s++) {
	  if (s < s_min) {
	    // The integrals for s < s_min are not needed, but we have to start the
	    // loop from s=0 to get the binomial coefficients right.
	  } else if (s-j >= 0) {
	    print("Subcase 2a (x < x_small)\n");
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
	      binom_smj_nu *= ((-1) * (s-j-nu))/(nu + ((real) 1.0));
	    }
	    //assert(abs(test_binom_nu - pow(1-1,s-j)) < 1.0e-10);
	  } else {
	    assert(s-j < 0);
	    assert(q >= j-s); 
	    print("Subcase 2b (x < x_small)\n");
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
	      binom_j_nu *= ((-1) * (j-nu))/(nu + ((real) 1.0));
	    }
	    //assert(abs(test_binom_nu - pow(1-1, j)) < 1.0e-10);
	  }

	  a_mu_pow /= a_mu;
	} // end loop over s
      }
    } else {
      //print("x >= x_small\n");
      /* x > x_small */
      invx = 1/x;
    
      if (k % 2 == 0) {
	//print("Case 1: k=2*j and x >= x_small\n");
	// Case 1: k=2*j, eqn. (22)
	/*
	                     q                    mu   -2*s+2*j-3     j-1                     a   nu                   b^2
	   integs[s] = c  sum     binom(q,mu) (-1)    b            sum      binom(j-1,nu) (- --- )   d(s-j+nu+1 + 1/2, --- )
                             mu=0                                     nu=0                   b^2                        a
	*/
	// compute integrals d(p+1/2,x)
	d_func<real>(x, p_min, p_max, w0, d);
	
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
	  binom_jm1_nu *= ((-invx) * (j-1-nu))/(nu + ((real) 1.0));
	}
	//assert(abs(test_binom_nu - pow(1 - invx, j-1)) < 1.0e-10);
      } else {
	print("Case 2: k=2*j+1 and x >= x_small\n");
	// Case 2: k=2*j+1
	assert(k == 2*j+1);

	if (s_max-j >= 0) {
	  // compute integrals gamma(p+1/2,x)
	  g_func<real>(x, p_max, g);
	}
	if (s_min-j < 0) {
	  h1_add = ((real) -0.5) * log(a_mu);
	  // compute integrals H(p,x)
	  h_func<real>(x, -s_max, j, h1_add, work, h);
	}
	real expxmw0 = exp(x-w0);
	real sqrtx = sqrt(x);
	real powinvx_expxmw0 = power(invx, j) / sqrtx * expxmw0;
	// a_mu^{j-s-1}
	a_mu_pow = power(a_mu, j-1);

	for (s=0; s<=s_max; s++) {
	  if (s < s_min) {
	    // The integrals for s < s_min are not needed, but we have to start the
	    // loop from s=0 to get the binomial coefficients right.
	  } else if (s-j >= 0) {
	    print("Subcase 2a (x >= x_small)\n");;
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
	      binom_smj_nu *= ((-invx) * (s-j-nu))/(nu + ((real) 1.0));
	    }
	    //assert(abs(test_binom_nu - pow(1-invx, s-j)) < 1.0e-10);
	  } else {
	    assert(s-j < 0);
	    assert(q >= j-s); 
	    print("Subcase 2b (x >= x_small)\n");
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
	      binom_j_nu *= ((-1) * (j-nu))/(nu + ((real) 1.0));
	    }
	    //assert(abs(test_binom_nu - pow(1-1, j)) < 1.0e-10);
	  }
	  a_mu_pow /= a_mu;
	} // end loop over s
      }
    }
    //test_binom_mu += binom_q_mu;
    
    // update binomial coefficient
    binom_q_mu *= ((-1)*(q-mu))/(mu + ((real) 1.0));
  } // end of loop over mu

  // 0 = (1-1)^q = sum_{mu=0}^q binom(q,mu) (-1)^mu
  //assert(abs(test_binom_mu - pow(1-1,q)) < 1.0e-10);
}

template <typename real,
    // total angular momentum of bra, li = nxi+nyi+nzi, and ket, lj = nxj+nyj+nzj
    int li, int lj,
    // operator    O(r) = x^mx y^my z^mz |r|^-k 
    int k,   int mx, int my, int mz,
    // power of cutoff function F2(r) = (1 - exp(-alpha r^2))^q
    int q  >
__device__ PolarizationIntegral<real, li, lj, k, mx, my, mz, q>::~PolarizationIntegral() {
}

template <typename real,
    // total angular momentum of bra, li = nxi+nyi+nzi, and ket, lj = nxj+nyj+nzj
    int li, int lj,
    // operator    O(r) = x^mx y^my z^mz |r|^-k 
    int k,   int mx, int my, int mz,
    // power of cutoff function F2(r) = (1 - exp(-alpha r^2))^q
    int q  >
template <int nxi, int nyi, int nzi,   int nxj, int nyj, int nzj>
__device__ real PolarizationIntegral<real, li, lj, k, mx, my, mz, q>::compute_pair(void) {
  /* see header file polarization.h */
  print("PolarizationIntegral::compute_pair\n");
  int eta_xi, eta_xj, eta_yi, eta_yj, eta_zi, eta_zj;
  // binom_xi_pow = binomial(nxi,eta_xi) (-xi)^(nxi - eta_xi)
  real binom_xi_pow, binom_xj_pow, binom_yi_pow, binom_yj_pow, binom_zi_pow, binom_zj_pow;
  int lx, ly, lz;
  // products of binomial coefficients and powers of centers
  real fxx, fxxyy, fxxyyzz;

  // binom_bx_pow = binomial(lx,zeta_x) bx^(lx - zeta_x)
  real binom_bx_pow, binom_by_pow, binom_bz_pow;
  int zeta_x, zeta_y, zeta_z;
  // products of binomial coefficients and powers of centers
  real gxy, gxyz;
  // If zeta_x is even, then even_x = true
  bool even_x, even_y, even_z;

  // accumulates polarization integrals
  real op = 0.0;

  // Variables beginning with test_... are only needed for the consistency of the code
  // and can be removed later.
  //real test_binomial_6, test_binomial_3;

  // maximum values for lx,ly,lz
  int lx_max, ly_max, lz_max;
  // maximum of all l#_max
  int l_max_ __attribute__((unused));

  int s;
  // maximum value of s = lx+ly+lz - (zeta_x+zeta_y+zeta_z)/2
  int s_max_ __attribute__((unused));

  assert("Total angular momentum for bra orbital differs from that used to create PolarizationIntegral instance!" && (nxi+nyi+nzi == li));
  assert("Total angular momentum for ket orbital differs from that used to create PolarizationIntegral instance!" && (nxj+nyj+nzj == lj));
  
  lx_max = nxi+nxj+mx;
  ly_max = nyi+nyj+my;
  lz_max = nzi+nzj+mz;
  l_max_ = max(lx_max, max(ly_max, lz_max));
  assert(l_max_ <= l_max);
  
  s_max_ = lx_max+ly_max+lz_max;
  assert(s_max_ <= s_max);

  // silence warnings about variables l_max_ and s_max_ not being used when assertions are turned off
  ((void) l_max_);
  ((void) s_max_);

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
		    binom_bz_pow *= (bz * zeta_z)/(lz - zeta_z + ((real) 1.0));
		  }
		  binom_by_pow *= (by * zeta_y)/(ly - zeta_y + ((real) 1.0));
		}
		binom_bx_pow *= (bx * zeta_x)/(lx - zeta_x + ((real) 1.0));
	      }
	      //assert(abs(test_binomial_3 - pow(1+bx,lx)*pow(1+by,ly)*pow(1+bz,lz))/abs(test_binomial_3) < 1.0e-10);	      	      
	      // update binomial coefficients for next iteration
	      //  B_{n,k-1} = x k / (n-k+1) B_{n,k}
	      binom_zj_pow *= ((-zj) * eta_zj)/(nzj - eta_zj + ((real) 1.0));
	    }
	    binom_zi_pow *= ((-zi) * eta_zi)/(nzi - eta_zi + ((real) 1.0));
	  }
	  binom_yj_pow *= ((-yj) * eta_yj)/(nyj - eta_yj + ((real) 1.0));
	}
	binom_yi_pow *= ((-yi) * eta_yi)/(nyi - eta_yi + ((real) 1.0));
      }
      binom_xj_pow *= ((-xj) * eta_xj)/(nxj - eta_xj + ((real) 1.0));
    }
    binom_xi_pow *= ((-xi) * eta_xi)/(nxi - eta_xi + ((real) 1.0));
  }
  // consistency check
  //assert(abs(test_binomial_6 - pow(1-xi,nxi)*pow(1-yi,nyi)*pow(1-zi,nzi) * pow(1-xj,nxj)*pow(1-yj,nyj)*pow(1-zj,nzj))/abs(test_binomial_6) < 1.0e-10);

  return op;
}

// ordering of angular momentum components
__constant__ AtomicOrbital s_orbitals[] = { {0, 0, 0} };
__constant__ AtomicOrbital p_orbitals[] = { {1, 0, 0}, {0, 1, 0}, {0, 0, 1} };
// ordering of d-functions in TeraChem: dxy,dxz,dyz,dxx,dyy,dzz
__constant__ AtomicOrbital d_orbitals[] = {
  {1, 1, 0}, {1, 0, 1}, {0, 1, 1},
  {2, 0, 0}, {0, 2, 0}, {0, 0, 2}};

inline __device__ const AtomicOrbital *ao_ordering(int angl) {
  switch(angl) {
  case 0:
    return s_orbitals;
  case 1: 
    return p_orbitals;
  case 2:
    return d_orbitals;
  default:
    BUG("Error: Angular momentum (%d) not supported!\n", angl);
    // To allow higher angular momenta, please add a list of AtomicOrbital above
    // with the desired ordering of basis functions (e.g. f_orbitals[] = {...}; ) 
    // and included it as a case (e.g. case 3: return f_orbitals; ).
    return NULL;
  }
}

template <typename real,
	  // operator    O(r) = x^mx y^my z^mz |r|^-k 
	  int k, int mx, int my, int mz,
	  // cutoff power
	  int q>
__launch_bounds__(BLOCK_SIZE)
__global__ void polarization_prim_pairs_kernel(
		  // array of pairs of primitives
                  const PrimitivePair<real> *pairs,
		  // number of pairs, length of array `pairs`
		  int npair,
		  // output
		  real *buffer,
		  // cutoff function F2(r) = (1 - exp(-alpha r^2))^q
		  real alpha) {
  /*
    AO polarization integrals for pairs of primitives

                                     mx my mz
                                    x  y  z           - alpha r^2  q
     buffer[ij] = coef  coef  <AO | --------- (1 - exp            )   |AO  >
                      i     j    i    r^k                                j

   See header file for more information.
  */
  print("start polarization_prim_pairs_kernel\n");
  // Each thread in a linear grid handles one pair of primitives
  int ipair = blockIdx.x * blockDim.x + threadIdx.x;
  if (ipair >= npair) {
    // This thread has nothing to do
    return;
  }
  PrimitivePair<real> pair = pairs[ipair];

  Primitive<real> *primA = &(pair.primA);
  Primitive<real> *primB = &(pair.primB);

  // The coordinate system is assumed to have been shifted so that the 
  // polarizable site lies at the origin.

  // Polarization integrals can reuse data if angular momentum quantum numbers
  // L(I)=angmomI and L(J)=angmomJ do not change. 

  const int lA = primA->l;
  const int lB = primB->l;

  real cc = primA->coef * primB->coef;

  // The integrals should be placed in the buffer at the positions
  //   bufferIdx, bufferIdx + 1, ..., bufferIdx + num_angomA * num_angmomB
  int ij = pair.bufferIdx;

  // unrolled loops over all angular momentum components (lA,mA,nA) of primitive A
  // and all angular momentum components (lB,mB,nB) of primitive B
/**** BEGIN of automatically generated code (with code_generator.py) *****/
  if (lA == 0) {
    if (lB == 0) {
      // ss integrals
      PolarizationIntegral<real, 0, 0, k, mx, my, mz, q> integrals
        (primA->x, primA->y, primA->z, primA->exp,
         primB->x, primB->y, primB->z, primB->exp,
         alpha);
            
      buffer[ij+ 0* 1+ 0] = cc * integrals.template compute_pair<0,0,0, 0,0,0>();
    } else if (lB == 1) {
      // sp integrals
      PolarizationIntegral<real, 0, 1, k, mx, my, mz, q> integrals
        (primA->x, primA->y, primA->z, primA->exp,
         primB->x, primB->y, primB->z, primB->exp,
         alpha);
            
      buffer[ij+ 0* 3+ 0] = cc * integrals.template compute_pair<0,0,0, 1,0,0>();
      buffer[ij+ 0* 3+ 1] = cc * integrals.template compute_pair<0,0,0, 0,1,0>();
      buffer[ij+ 0* 3+ 2] = cc * integrals.template compute_pair<0,0,0, 0,0,1>();
    } else if (lB == 2) {
      // sd integrals
      PolarizationIntegral<real, 0, 2, k, mx, my, mz, q> integrals
        (primA->x, primA->y, primA->z, primA->exp,
         primB->x, primB->y, primB->z, primB->exp,
         alpha);
            
      buffer[ij+ 0* 6+ 0] = cc * integrals.template compute_pair<0,0,0, 1,1,0>();
      buffer[ij+ 0* 6+ 1] = cc * integrals.template compute_pair<0,0,0, 1,0,1>();
      buffer[ij+ 0* 6+ 2] = cc * integrals.template compute_pair<0,0,0, 0,1,1>();
      buffer[ij+ 0* 6+ 3] = cc * integrals.template compute_pair<0,0,0, 2,0,0>();
      buffer[ij+ 0* 6+ 4] = cc * integrals.template compute_pair<0,0,0, 0,2,0>();
      buffer[ij+ 0* 6+ 5] = cc * integrals.template compute_pair<0,0,0, 0,0,2>();
    }
  } else if (lA == 1) {
    if (lB == 0) {
      // ps integrals
      PolarizationIntegral<real, 1, 0, k, mx, my, mz, q> integrals
        (primA->x, primA->y, primA->z, primA->exp,
         primB->x, primB->y, primB->z, primB->exp,
         alpha);
            
      buffer[ij+ 0* 1+ 0] = cc * integrals.template compute_pair<1,0,0, 0,0,0>();
      buffer[ij+ 1* 1+ 0] = cc * integrals.template compute_pair<0,1,0, 0,0,0>();
      buffer[ij+ 2* 1+ 0] = cc * integrals.template compute_pair<0,0,1, 0,0,0>();
    } else if (lB == 1) {
      // pp integrals
      PolarizationIntegral<real, 1, 1, k, mx, my, mz, q> integrals
        (primA->x, primA->y, primA->z, primA->exp,
         primB->x, primB->y, primB->z, primB->exp,
         alpha);
            
      buffer[ij+ 0* 3+ 0] = cc * integrals.template compute_pair<1,0,0, 1,0,0>();
      buffer[ij+ 0* 3+ 1] = cc * integrals.template compute_pair<1,0,0, 0,1,0>();
      buffer[ij+ 0* 3+ 2] = cc * integrals.template compute_pair<1,0,0, 0,0,1>();
      buffer[ij+ 1* 3+ 0] = cc * integrals.template compute_pair<0,1,0, 1,0,0>();
      buffer[ij+ 1* 3+ 1] = cc * integrals.template compute_pair<0,1,0, 0,1,0>();
      buffer[ij+ 1* 3+ 2] = cc * integrals.template compute_pair<0,1,0, 0,0,1>();
      buffer[ij+ 2* 3+ 0] = cc * integrals.template compute_pair<0,0,1, 1,0,0>();
      buffer[ij+ 2* 3+ 1] = cc * integrals.template compute_pair<0,0,1, 0,1,0>();
      buffer[ij+ 2* 3+ 2] = cc * integrals.template compute_pair<0,0,1, 0,0,1>();
    } else if (lB == 2) {
      // pd integrals
      PolarizationIntegral<real, 1, 2, k, mx, my, mz, q> integrals
        (primA->x, primA->y, primA->z, primA->exp,
         primB->x, primB->y, primB->z, primB->exp,
         alpha);
            
      buffer[ij+ 0* 6+ 0] = cc * integrals.template compute_pair<1,0,0, 1,1,0>();
      buffer[ij+ 0* 6+ 1] = cc * integrals.template compute_pair<1,0,0, 1,0,1>();
      buffer[ij+ 0* 6+ 2] = cc * integrals.template compute_pair<1,0,0, 0,1,1>();
      buffer[ij+ 0* 6+ 3] = cc * integrals.template compute_pair<1,0,0, 2,0,0>();
      buffer[ij+ 0* 6+ 4] = cc * integrals.template compute_pair<1,0,0, 0,2,0>();
      buffer[ij+ 0* 6+ 5] = cc * integrals.template compute_pair<1,0,0, 0,0,2>();
      buffer[ij+ 1* 6+ 0] = cc * integrals.template compute_pair<0,1,0, 1,1,0>();
      buffer[ij+ 1* 6+ 1] = cc * integrals.template compute_pair<0,1,0, 1,0,1>();
      buffer[ij+ 1* 6+ 2] = cc * integrals.template compute_pair<0,1,0, 0,1,1>();
      buffer[ij+ 1* 6+ 3] = cc * integrals.template compute_pair<0,1,0, 2,0,0>();
      buffer[ij+ 1* 6+ 4] = cc * integrals.template compute_pair<0,1,0, 0,2,0>();
      buffer[ij+ 1* 6+ 5] = cc * integrals.template compute_pair<0,1,0, 0,0,2>();
      buffer[ij+ 2* 6+ 0] = cc * integrals.template compute_pair<0,0,1, 1,1,0>();
      buffer[ij+ 2* 6+ 1] = cc * integrals.template compute_pair<0,0,1, 1,0,1>();
      buffer[ij+ 2* 6+ 2] = cc * integrals.template compute_pair<0,0,1, 0,1,1>();
      buffer[ij+ 2* 6+ 3] = cc * integrals.template compute_pair<0,0,1, 2,0,0>();
      buffer[ij+ 2* 6+ 4] = cc * integrals.template compute_pair<0,0,1, 0,2,0>();
      buffer[ij+ 2* 6+ 5] = cc * integrals.template compute_pair<0,0,1, 0,0,2>();
    }
  } else if (lA == 2) {
    if (lB == 0) {
      // ds integrals
      PolarizationIntegral<real, 2, 0, k, mx, my, mz, q> integrals
        (primA->x, primA->y, primA->z, primA->exp,
         primB->x, primB->y, primB->z, primB->exp,
         alpha);
            
      buffer[ij+ 0* 1+ 0] = cc * integrals.template compute_pair<1,1,0, 0,0,0>();
      buffer[ij+ 1* 1+ 0] = cc * integrals.template compute_pair<1,0,1, 0,0,0>();
      buffer[ij+ 2* 1+ 0] = cc * integrals.template compute_pair<0,1,1, 0,0,0>();
      buffer[ij+ 3* 1+ 0] = cc * integrals.template compute_pair<2,0,0, 0,0,0>();
      buffer[ij+ 4* 1+ 0] = cc * integrals.template compute_pair<0,2,0, 0,0,0>();
      buffer[ij+ 5* 1+ 0] = cc * integrals.template compute_pair<0,0,2, 0,0,0>();
    } else if (lB == 1) {
      // dp integrals
      PolarizationIntegral<real, 2, 1, k, mx, my, mz, q> integrals
        (primA->x, primA->y, primA->z, primA->exp,
         primB->x, primB->y, primB->z, primB->exp,
         alpha);
            
      buffer[ij+ 0* 3+ 0] = cc * integrals.template compute_pair<1,1,0, 1,0,0>();
      buffer[ij+ 0* 3+ 1] = cc * integrals.template compute_pair<1,1,0, 0,1,0>();
      buffer[ij+ 0* 3+ 2] = cc * integrals.template compute_pair<1,1,0, 0,0,1>();
      buffer[ij+ 1* 3+ 0] = cc * integrals.template compute_pair<1,0,1, 1,0,0>();
      buffer[ij+ 1* 3+ 1] = cc * integrals.template compute_pair<1,0,1, 0,1,0>();
      buffer[ij+ 1* 3+ 2] = cc * integrals.template compute_pair<1,0,1, 0,0,1>();
      buffer[ij+ 2* 3+ 0] = cc * integrals.template compute_pair<0,1,1, 1,0,0>();
      buffer[ij+ 2* 3+ 1] = cc * integrals.template compute_pair<0,1,1, 0,1,0>();
      buffer[ij+ 2* 3+ 2] = cc * integrals.template compute_pair<0,1,1, 0,0,1>();
      buffer[ij+ 3* 3+ 0] = cc * integrals.template compute_pair<2,0,0, 1,0,0>();
      buffer[ij+ 3* 3+ 1] = cc * integrals.template compute_pair<2,0,0, 0,1,0>();
      buffer[ij+ 3* 3+ 2] = cc * integrals.template compute_pair<2,0,0, 0,0,1>();
      buffer[ij+ 4* 3+ 0] = cc * integrals.template compute_pair<0,2,0, 1,0,0>();
      buffer[ij+ 4* 3+ 1] = cc * integrals.template compute_pair<0,2,0, 0,1,0>();
      buffer[ij+ 4* 3+ 2] = cc * integrals.template compute_pair<0,2,0, 0,0,1>();
      buffer[ij+ 5* 3+ 0] = cc * integrals.template compute_pair<0,0,2, 1,0,0>();
      buffer[ij+ 5* 3+ 1] = cc * integrals.template compute_pair<0,0,2, 0,1,0>();
      buffer[ij+ 5* 3+ 2] = cc * integrals.template compute_pair<0,0,2, 0,0,1>();
    } else if (lB == 2) {
      // dd integrals
      PolarizationIntegral<real, 2, 2, k, mx, my, mz, q> integrals
        (primA->x, primA->y, primA->z, primA->exp,
         primB->x, primB->y, primB->z, primB->exp,
         alpha);
            
      buffer[ij+ 0* 6+ 0] = cc * integrals.template compute_pair<1,1,0, 1,1,0>();
      buffer[ij+ 0* 6+ 1] = cc * integrals.template compute_pair<1,1,0, 1,0,1>();
      buffer[ij+ 0* 6+ 2] = cc * integrals.template compute_pair<1,1,0, 0,1,1>();
      buffer[ij+ 0* 6+ 3] = cc * integrals.template compute_pair<1,1,0, 2,0,0>();
      buffer[ij+ 0* 6+ 4] = cc * integrals.template compute_pair<1,1,0, 0,2,0>();
      buffer[ij+ 0* 6+ 5] = cc * integrals.template compute_pair<1,1,0, 0,0,2>();
      buffer[ij+ 1* 6+ 0] = cc * integrals.template compute_pair<1,0,1, 1,1,0>();
      buffer[ij+ 1* 6+ 1] = cc * integrals.template compute_pair<1,0,1, 1,0,1>();
      buffer[ij+ 1* 6+ 2] = cc * integrals.template compute_pair<1,0,1, 0,1,1>();
      buffer[ij+ 1* 6+ 3] = cc * integrals.template compute_pair<1,0,1, 2,0,0>();
      buffer[ij+ 1* 6+ 4] = cc * integrals.template compute_pair<1,0,1, 0,2,0>();
      buffer[ij+ 1* 6+ 5] = cc * integrals.template compute_pair<1,0,1, 0,0,2>();
      buffer[ij+ 2* 6+ 0] = cc * integrals.template compute_pair<0,1,1, 1,1,0>();
      buffer[ij+ 2* 6+ 1] = cc * integrals.template compute_pair<0,1,1, 1,0,1>();
      buffer[ij+ 2* 6+ 2] = cc * integrals.template compute_pair<0,1,1, 0,1,1>();
      buffer[ij+ 2* 6+ 3] = cc * integrals.template compute_pair<0,1,1, 2,0,0>();
      buffer[ij+ 2* 6+ 4] = cc * integrals.template compute_pair<0,1,1, 0,2,0>();
      buffer[ij+ 2* 6+ 5] = cc * integrals.template compute_pair<0,1,1, 0,0,2>();
      buffer[ij+ 3* 6+ 0] = cc * integrals.template compute_pair<2,0,0, 1,1,0>();
      buffer[ij+ 3* 6+ 1] = cc * integrals.template compute_pair<2,0,0, 1,0,1>();
      buffer[ij+ 3* 6+ 2] = cc * integrals.template compute_pair<2,0,0, 0,1,1>();
      buffer[ij+ 3* 6+ 3] = cc * integrals.template compute_pair<2,0,0, 2,0,0>();
      buffer[ij+ 3* 6+ 4] = cc * integrals.template compute_pair<2,0,0, 0,2,0>();
      buffer[ij+ 3* 6+ 5] = cc * integrals.template compute_pair<2,0,0, 0,0,2>();
      buffer[ij+ 4* 6+ 0] = cc * integrals.template compute_pair<0,2,0, 1,1,0>();
      buffer[ij+ 4* 6+ 1] = cc * integrals.template compute_pair<0,2,0, 1,0,1>();
      buffer[ij+ 4* 6+ 2] = cc * integrals.template compute_pair<0,2,0, 0,1,1>();
      buffer[ij+ 4* 6+ 3] = cc * integrals.template compute_pair<0,2,0, 2,0,0>();
      buffer[ij+ 4* 6+ 4] = cc * integrals.template compute_pair<0,2,0, 0,2,0>();
      buffer[ij+ 4* 6+ 5] = cc * integrals.template compute_pair<0,2,0, 0,0,2>();
      buffer[ij+ 5* 6+ 0] = cc * integrals.template compute_pair<0,0,2, 1,1,0>();
      buffer[ij+ 5* 6+ 1] = cc * integrals.template compute_pair<0,0,2, 1,0,1>();
      buffer[ij+ 5* 6+ 2] = cc * integrals.template compute_pair<0,0,2, 0,1,1>();
      buffer[ij+ 5* 6+ 3] = cc * integrals.template compute_pair<0,0,2, 2,0,0>();
      buffer[ij+ 5* 6+ 4] = cc * integrals.template compute_pair<0,0,2, 0,2,0>();
      buffer[ij+ 5* 6+ 5] = cc * integrals.template compute_pair<0,0,2, 0,0,2>();
    }
  }
/**** END of automatically generated code *****/
}

inline int kappa(int n) {
  if (n % 2 == 0) {
    return n/2;
  } else {
    return (n+1)/2;
  }
}

#ifdef PINNED_MEMORY
// Memory transfer and calculation are partially overlapped
template <typename real,
	  // operator    O(r) = x^mx y^my z^mz |r|^-k 
	  int k, int mx, int my, int mz,
	  // cutoff power in cutoff function F2(r) = (1 - exp(-alpha r^2))^q
	  int q>
void polarization_prim_pairs(// pointer to array of pairs of primitives in pinned (!) CPU memory
			     const PrimitivePair<real> *pairs,
			     // number of pairs
			     int npair,
			     // pointer to output buffer in pinned (!) CPU memory
			     real *buffer,
			     // cutoff exponent in cutoff function F2(r) = (1 - exp(-alpha r^2))^q
			     real alpha) {
  assert("Integrals are only implemented for the case k > 2!" && k > 2);
  // check that exponent of operator k and cutoff power q are compatible, otherwise the integrals
  // do not exist
  assert("Integrals do not exist for this combination of k and q!" && (q >= kappa(k) - kappa(mx) - kappa(my) - kappa(mz) - 1));
  print("start polarization_prim_pairs\n");

  // allocate memory for primitive pairs on the GPU
  PrimitivePair<real> *pairs_;
  checkCuda( cudaMalloc((void **) &pairs_, sizeof(PrimitivePair<real>) * npair) );

  // memory for output buffer on the GPU
  real *buffer_;
  // How much memory do we need to hold all integrals?
  int buffer_size = integral_buffer_size<real>(pairs, npair);
  checkCuda( cudaMalloc((void **) &buffer_, sizeof(real) * buffer_size) );

  // overlap data transfer and kernal computation by dividing the data
  // into blocks and repeating the operations {copy H->D, kernel, copy D->H}
  // for each of them.
  //  see https://developer.nvidia.com/blog/how-overlap-data-transfers-cuda-cc/
  const int num_streams = 4;
  cudaStream_t streams[num_streams];
  for (int s = 0; s < num_streams; s++)
    checkCuda( cudaStreamCreate(&streams[s]) );

  // offset to beginning of current block in input and output arrays
  int offset_inp = 0;
  int offset_out = 0;
  for(int s = 0; s < num_streams; s++) {
    // 
    int block_size_inp = min((npair+num_streams-1) / num_streams, npair - offset_inp);
    
    // Host -> Device copy
    checkCuda( cudaMemcpyAsync(pairs_+offset_inp, pairs+offset_inp, 
			       sizeof(PrimitivePair<real>) * block_size_inp,
			       cudaMemcpyHostToDevice, streams[s]) );
    // kernel computation
    // grid is a linear array
    dim3 blockDim(BLOCK_SIZE);
    dim3 gridDim((npair + blockDim.x - 1)/blockDim.x);
    // launch kernel
    print("launch kernel with  %d blocks of size %d each\n", gridDim.x, blockDim.x);
    polarization_prim_pairs_kernel<real, k, mx, my, mz, q>
      <<<gridDim,blockDim, 0, streams[s]>>>(pairs_+offset_inp, block_size_inp, 
					    buffer_, 
					    alpha);
    // check for errors
    checkCuda( cudaGetLastError() );
    // Device -> Host copy
    // Compute number of integrals generated by this block
    int block_size_out = integral_buffer_size<real>(pairs+offset_inp, block_size_inp);
    checkCuda( cudaMemcpyAsync(buffer+offset_out, buffer_+offset_out,
			       sizeof(real) * block_size_out,
			       cudaMemcpyDeviceToHost, streams[s]) );

    print("Stream %d finished\n", s);
    print("   input:   starting offset = %d  block size = %d\n", offset_inp, block_size_inp);
    print("   output:  starting offset = %d  block size = %d\n", offset_out, block_size_out);

    offset_inp += block_size_inp;
    offset_out += block_size_out;
  }
  assert(offset_inp == npair);
  assert(offset_out == buffer_size);
  // clean up
  for (int s = 0; s < num_streams; s++) {
    checkCuda( cudaStreamDestroy(streams[s]) );
  }
  // release dynamic GPU memory
  cudaFree(pairs_);
  cudaFree(buffer_);
}

#else // PINNED_MEMORY

// Memory transfer and calculation occur sequentially.
template <typename real,
	  // operator    O(r) = x^mx y^my z^mz |r|^-k 
	  int k, int mx, int my, int mz,
	  // cutoff power in cutoff function F2(r) = (1 - exp(-alpha r^2))^q
	  int q>
void polarization_prim_pairs(// pointer to array of pairs of primitives in pinned (!) CPU memory
			     const PrimitivePair<real> *pairs,
			     // number of pairs
			     int npair,
			     // pointer to output buffer in pinned (!) CPU memory
			     real *buffer,
			     // cutoff exponent in cutoff function F2(r) = (1 - exp(-alpha r^2))^q
			     real alpha) {
  assert("Integrals are only implemented for the case k > 2!" && k > 2);
  // check that exponent of operator k and cutoff power q are compatible, otherwise the integrals
  // do not exist
  assert("Integrals do not exist for this combination of k and q!" && (q >= kappa(k) - kappa(mx) - kappa(my) - kappa(mz) - 1));
  print("start polarization_prim_pairs\n");

  // allocate memory for primitive pairs on the GPU
  PrimitivePair<real> *pairs_;
  checkCuda( cudaMalloc((void **) &pairs_, sizeof(PrimitivePair<real>) * npair) );

  // memory for output buffer on the GPU
  real *buffer_;
  // How much memory do we need to hold all integrals?
  int buffer_size = integral_buffer_size<real>(pairs, npair);
  checkCuda( cudaMalloc((void **) &buffer_, sizeof(real) * buffer_size) );

  // Host -> Device copy
  checkCuda( cudaMemcpy(pairs_, pairs, sizeof(PrimitivePair<real>) * npair, cudaMemcpyHostToDevice) );
  // kernel computation
  // grid is a linear array
  dim3 blockDim(BLOCK_SIZE);
  dim3 gridDim((npair + blockDim.x - 1)/blockDim.x);
  // launch kernel
  print("launch kernel with  %d blocks of size %d each\n", gridDim.x, blockDim.x);
  polarization_prim_pairs_kernel<real, k, mx, my, mz, q>
    <<<gridDim,blockDim>>>(pairs_, npair, buffer_, alpha);
  //polarization_prim_pairs_kernel<real, k, mx, my, mz, q><<<1,1>>>(pairs_, npair, buffer_, alpha);
  // check for errors
  checkCuda( cudaGetLastError() );
  // Device -> Host copy
  checkCuda( cudaMemcpy(buffer, buffer_, sizeof(real) * buffer_size, cudaMemcpyDeviceToHost) );

  // release dynamic GPU memory
  cudaFree(pairs_);
  cudaFree(buffer_);
}
#endif // PINNED_MEMORY
    
// compiles specialized functions for double and single precision
// and for all combinations of operator powers  k,   mx, my, mz,   q

/**** BEGIN of automatically generated code (with code_generator.py) *****/

// Op(r) = 1/|r|^3

template void polarization_prim_pairs<double, 3,   0, 0, 0,   2>
        (const PrimitivePair<double> *pairs, int npair, double *buffer, double alpha);
template void polarization_prim_pairs<float,  3,   0, 0, 0,   2>
        (const PrimitivePair<float> *pairs, int npair, float *buffer, float alpha);

// Op(r) = 1/|r|^4

template void polarization_prim_pairs<double, 4,   0, 0, 0,   2>
        (const PrimitivePair<double> *pairs, int npair, double *buffer, double alpha);
template void polarization_prim_pairs<float,  4,   0, 0, 0,   2>
        (const PrimitivePair<float> *pairs, int npair, float *buffer, float alpha);

// Op(r) = r(i)/|r|^3

template void polarization_prim_pairs<double, 3,   0, 0, 1,   2>
        (const PrimitivePair<double> *pairs, int npair, double *buffer, double alpha);
template void polarization_prim_pairs<float,  3,   0, 0, 1,   2>
        (const PrimitivePair<float> *pairs, int npair, float *buffer, float alpha);

template void polarization_prim_pairs<double, 3,   0, 1, 0,   2>
        (const PrimitivePair<double> *pairs, int npair, double *buffer, double alpha);
template void polarization_prim_pairs<float,  3,   0, 1, 0,   2>
        (const PrimitivePair<float> *pairs, int npair, float *buffer, float alpha);

template void polarization_prim_pairs<double, 3,   1, 0, 0,   2>
        (const PrimitivePair<double> *pairs, int npair, double *buffer, double alpha);
template void polarization_prim_pairs<float,  3,   1, 0, 0,   2>
        (const PrimitivePair<float> *pairs, int npair, float *buffer, float alpha);

// Op(r) = r(i)r(j)/|r|^6

template void polarization_prim_pairs<double, 6,   0, 0, 2,   2>
        (const PrimitivePair<double> *pairs, int npair, double *buffer, double alpha);
template void polarization_prim_pairs<float,  6,   0, 0, 2,   2>
        (const PrimitivePair<float> *pairs, int npair, float *buffer, float alpha);

template void polarization_prim_pairs<double, 6,   0, 1, 1,   2>
        (const PrimitivePair<double> *pairs, int npair, double *buffer, double alpha);
template void polarization_prim_pairs<float,  6,   0, 1, 1,   2>
        (const PrimitivePair<float> *pairs, int npair, float *buffer, float alpha);

template void polarization_prim_pairs<double, 6,   0, 2, 0,   2>
        (const PrimitivePair<double> *pairs, int npair, double *buffer, double alpha);
template void polarization_prim_pairs<float,  6,   0, 2, 0,   2>
        (const PrimitivePair<float> *pairs, int npair, float *buffer, float alpha);

template void polarization_prim_pairs<double, 6,   1, 0, 1,   2>
        (const PrimitivePair<double> *pairs, int npair, double *buffer, double alpha);
template void polarization_prim_pairs<float,  6,   1, 0, 1,   2>
        (const PrimitivePair<float> *pairs, int npair, float *buffer, float alpha);

template void polarization_prim_pairs<double, 6,   1, 1, 0,   2>
        (const PrimitivePair<double> *pairs, int npair, double *buffer, double alpha);
template void polarization_prim_pairs<float,  6,   1, 1, 0,   2>
        (const PrimitivePair<float> *pairs, int npair, float *buffer, float alpha);

template void polarization_prim_pairs<double, 6,   2, 0, 0,   2>
        (const PrimitivePair<double> *pairs, int npair, double *buffer, double alpha);
template void polarization_prim_pairs<float,  6,   2, 0, 0,   2>
        (const PrimitivePair<float> *pairs, int npair, float *buffer, float alpha);

/**** END of automatically generated code *****/


template <typename real>
int integral_buffer_size(// array of pairs of primitives
			 const PrimitivePair<real> *pairs,
			 // number of pairs in array
			 int npair) {
  /* compute the size of the buffer to hold all polarization integrals */
  int buffer_size = 0;
  // loop over all pairs of primitives
  for(int ipair = 0; ipair < npair; ipair++) {
    const PrimitivePair<real> *pair = pairs+ipair;
    const Primitive<real> *primA = &(pair->primA);
    const Primitive<real> *primB = &(pair->primB);
    // Increase buffer by the amount needed for this pair, all combinations of angular momenta
    buffer_size += ANGL_FUNCS(primA->l) * ANGL_FUNCS(primB->l);
  }
  return buffer_size;
}

// templates for double and single precision
template int integral_buffer_size<double>(const PrimitivePair<double> *pairs, int npair);
template int integral_buffer_size<float>(const PrimitivePair<float> *pairs, int npair);
