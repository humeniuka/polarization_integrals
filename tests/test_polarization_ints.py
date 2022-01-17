#!/usr/bin/env python
"""
compare polarization integrals from C++ implementation with reference implementation
"""
import numpy as np
import numpy.linalg as la
from scipy import special

# slow python implementation
from polarization_ints_reference import d_func as d_func_reference
from polarization_ints_reference import h_func as h_func_reference
from polarization_ints_reference import polarization_integral as polarization_integral_reference
# fast C++ implementation
from polarization_integrals import PolarizationIntegral
# 
from polarization_integrals import _polarization

def d_func(x, pmin, pmax, w0):
    """
    evaluates the integrals (eqns. (23) and (29) in Ref. [CPP] with eta=0)
    
                                   /x      p-1/2
    d(a,x) = d(p+1/2,x) = exp(-w0) |  dw  w      exp(w)
                                   /0

    for a = p+1/2 with p an integer.
    The prefactor exp(-w0) allows to avoid overflows in the exponential.
    
    The function values are generated iteratively for all integers p in the
    range p = pmin,pmin+1,..., 0 ,1,2,...,pmax 

    1) by upward iteration for p=0,1,2,...,pmax (a=1/2,3/2,...,pmax+1/2)

      Starting value (p=0)

          d[0] = d(1/2,x) = 2 exp(x-w0) dawson(sqrt(x))
      
      Iteration  (p -> p+1)

                    p+1/2
          d[p+1] = x      exp(x-w0)  - (p+1/2) d[p]

    2) and by downward iteration for p=0,-1,-2,...,pmin

      Iteration (-p -> -(p+1))
                           -(p+1/2)
          d[-(p+1)] = - ( x         exp(x-w0) - d[-p] ) / (p+1/2)

    Parameters
    ----------
    x          : float >= 0
      upper limit of integration
    pmin, pmax : int
      defines range pmin <= p <= pmax
    w0         : float
      To avoid overflow in the exponential function, exp(-w0) * d(p+1/2,x) is calculated.

    Returns
    -------
    d          : array of size (|pmin|+pmax+1,)
      exp(-w0) * d(p+1/2,x) in the order p = 0,1,...,pmax,pmin,pmin+1,...,-2,-1
    """
    assert pmin <= 0 and pmax >= 0
    # output array
    d = np.zeros(-pmin+pmax+1)
    # constants during iteration
    expx = np.exp(x-w0)
    sqrtx = np.sqrt(x)
    dwsn = special.dawsn(sqrtx)

    # initialization p=0
    d[0] = 2*expx * dwsn

    # 1) upward iteration starting from p=0
    xp = sqrtx * expx
    for p in range(0,pmax):
        d[p+1] = xp - (p+0.5)*d[p]
        # x^(p+1/2)  * exp(x-w0)
        xp *= x

    # 2) downward iteration starting from p=0
    ixp = 1/sqrtx * expx
    for p in range(0,pmin,-1):
        d[p-1] = -(ixp - d[p])/(-p+0.5)
        # x^(-(p+1/2)) * exp(x-w0)
        ixp /= x

    return d

def d_func_zero_limit(x, pmin, pmax, w0):
    """
    The function \tilde{d} also computes d(p+1/2,x), however without the factor x^{p+1/2}:

      ~             p+1/2
      d(p+1/2,x) = x      d(p+1/2,x)          for all integers p

    This ensures that \tilde{d} has a finite value in the limit x -> 0.
    """
    assert pmin <= 0 and pmax >= 0
    # output array
    dt = np.zeros(-pmin+pmax+1)
    # constants during iterations
    expx = np.exp(x-w0)
    
    # 1) For p >= 0, \tilde{d} is calculated from the Taylor expansion around x=0.
    #
    #      ~          inf     x^k
    #      d (x) = sum     ------------
    #       p         k=0  k! (p+k+1/2)
    #
    #    The Taylor expansion is truncated at k_max = 20
    kmax = 20
    # y = x^k / k! * exp(-w0)
    y = np.exp(-w0)
    for k in range(0, kmax):
        for p in range(0, pmax+1):
            dt[p] += y/(p+k+0.5)
        y *= x/(k+1)

    # 2) For -p < 0, \tilde{d} is obtained by downward iteration starting from p=0
    #    according to the prescription
    #
    #      ~              1       x        ~
    #      d       = - ------- ( e   -  x  d   )
    #       -(p+1)      p+1/2               -p
    #
    for p in range(0, -pmin):
        dt[-(p+1)] = - (expx - x*dt[-p])/(p+0.5)

    return dt

        
def g_func(x, pmax):
    """
    evaluates the integral

                          /x      p-1/2
        gamma(p+1/2, x) = |  dw  w      exp(-w)
                          /0

    iteratively for all p=0,1,...,pmax. The closed-form expression for gamma is given in eqn. (33) in [CPP].

    Parameters
    ----------
    x    :  float >= 0
      upper integration limit
    pmax :  int >= 0
      integral is evaluated for all integers p=0,1,...,pmax

    Returns
    -------
    g    :  array of size (pmax+1,)
      gamma(p+1/2,x) in the order p=0,1,...,pmax
    """
    assert pmax >= 0
    # output array
    g = np.zeros(pmax+1)

    # constants during iteration
    sqrtx = np.sqrt(x)
    expmx = np.exp(-x)
    # The definition of the error function in eqn. (5) [CPP] differs from the usual definition,
    # which is also used in scipy (see https://en.wikipedia.org/wiki/Error_function )
    # 
    #  * eqn. (5)            erf(x) = integral(exp(-t^2), t=0...z)        (eqn. 5)
    #  * scipy's definition  erf(x) = 2/sqrt(pi) integral(exp(-t^2), t=0...z)
    #
    
    # initialization p=0
    g[0] = np.sqrt(np.pi) * special.erf(sqrtx)

    # upward iteration starting from p=0
    # x^(p+1/2) * exp(-x)
    xp = sqrtx * expmx
    for p in range(0, pmax):
        g[p+1] = -xp + (p+0.5)*g[p]
        xp *= x

    return g

def m_func(x):
    if x >= 6.0:
        return np.exp(x**2) * special.erf(x) * special.dawsn(x)
    else:
        # hard-coded values m(xi) and m'(xi) at the expansion points
        x0s = np.array([0., 0.5, 1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5, 5., 5.5])
        m0s = np.array([
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
            1.269164846178781e12])
        m1s = np.array([
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
            1.372170497746480e13])
        
        # select the expansion point i if x0(i) <= x < x0(i+1)
        i = int(2*x)
        x0 = x0s[i]

        # Taylor expansion is truncated after nmax+1 terms
        nmax = 20
        m_deriv = np.zeros(nmax+1)
        m_deriv[0] = m0s[i]
        m_deriv[1] = m1s[i]
        m_deriv[2] = 2*x0*m_deriv[1] + 2.0/np.sqrt(np.pi)

        # compute derivatives of m(x) at the expansion point iteratively
        #               d^k m(x)
        #  m_deriv[k] = -------- (x0)
        #                d x^k
        #
        for n in range(3, nmax+1):
            m_deriv[n] = 2*(n-2)*m_deriv[n-2] + 2*x0*m_deriv[n-1]

        # evaluate Taylor expansion
        dx = x-x0
        y = dx   # holds (x-x0)^n / n!
        # accumulator
        m = m_deriv[0]
        for n in range(1, nmax+1):
            m += y * m_deriv[n]
            y *= dx/(n+1)

        return m

def h_func(x, pmin, pmax):
    """
    evaluates the non-diverging part of the integral

               /1          -p
      H(p,x) = | dt (1-t^2)   exp(-x t^2) 
               /0

    according to the recursion relations in  eqns. (37) in [CPP] for p > 0 
    and eqn. (38) in [CPP] for p < 0 for all integers p in the range [pmin, pmax]

    Parameters
    ----------
    x       :   float > 0
      second argument
    pmin    :   int <= 0
      lower limit for p
    pmax    :   int >= 0
      upper limit for p

    Returns
    -------
    h       :   array of size (-pmin+pmax+1,)
      h[p] = H(p,x) in the order p=0,1,...,pmax,pmin,pmin+1,...,-1
    """
    assert pmin <= 0 and pmax >= 0
    # output array
    h = np.zeros(-pmin+pmax+1)

    # 1) upward recursion for positive p
    # initial values for upward recursion
    # H(0,x), eqn. (37b)
    sqrtx = np.sqrt(x)
    h[0] = 0.5 * np.sqrt(np.pi)/sqrtx * special.erf(sqrtx)
    if (pmax > 0):
        # H(1,x), eqn. (37c)
        h[1] = np.sqrt(np.pi) * np.exp(-x) * m_func(sqrtx)
    """
    # I think the following is wrong because of the missing factors sqrt(pi)/2,
    # but with these definitions for H(0,x) and H(1,x) I get the same result as Xiao for p > 0.
    h[0] = 1.0/sqrtx * special.erf(sqrtx)
    # H(1,x), eqn. (37c)
    h[1] = 2.0 * np.exp(-x) * m_func(sqrtx)
    """
    
    for p in range(2,pmax+1):
        # compute H(p,x) from H(p-1,x) and H(p-2,x)
        # eqn. (37a)
        h[p] = (2*(p+x)-3)/(2*(p-1)) * h[p-1] - x/(p-1) * h[p-2]

    # 2) for negative p we need gamma(k+1/2,x) for k=0,1,...,|pmin|
    g = g_func(x, -pmin)

    # eqn. (38)
    #                                                         k 
    # H(-p,x) = 1/(2 sqrt(x)) sum(k to p) binomial(p,k) (-1/x)   gamma(k+1/2,x)
    #         = 1/(2 sqrt(x)) sum(k to p) B_{p,k} g[k]
    invmx = (-1/x)
    for p in range(1,-pmin+1):
        acc = 0.0
        # B_{p,k} = binomial(p,k) (-1/x)^k
        y = 1.0  # holds B_{p,k}
        for k in range(0, p+1):
            acc += y * g[k]
            # B_{p,k+1} = (-1/x) (p-k)/(k+1) B_{p,k}
            y *= invmx * (p-k)/(k+1)
        acc *= 1.0/(2*sqrtx)

        h[-p] = acc
        
    return h

def h_func_small_x(x, pmin, pmax):
    """
    compute H(p,x) for small x or x=0
    """
    assert pmin <= 0 and pmax >= 0
    # output array
    h = np.zeros(-pmin+pmax+1)

    # 1) upward recursion for positive p
    # Taylor expansion of H(0,x) around x=0, which is truncated at kmax
    #
    #                  kmax    (-2x)^k
    #  H(0,x) = 1 + sum     -------------
    #                  k=1  (2k+1) (2k)!!
    #
    kmax = 20
    y = (-2*x)
    # yk = (-2x)^k / (2k)!!
    yk = y/2
    h0 = 1.0
    for k in range(1, kmax+1):
        h0 += yk/(2*k+1)
        assert(abs(yk - pow(-2*x, k)/special.factorial2(2*k)) < 1.0e-10)
        yk *= y / (2*(k+1))
        
    # initial values for upward recursion
    h[0] = h0
    
    # H(1,x), eqn. (37c)
    sqrtx = np.sqrt(x)
    h[1] = np.sqrt(np.pi) * np.exp(-x) * m_func(sqrtx)

    for p in range(2,pmax+1):
        # compute H(p,x) from H(p-1,x) and H(p-2,x)
        # eqn. (37a)
        h[p] = (2*(p+x)-3)/(2*(p-1)) * h[p-1] - x/(p-1) * h[p-2]

    # 2) for negative p we need \tilde{d}(k+1/2,-x) for k=0,1,...,|pmin|
    d_tilde = d_func_zero_limit(-x, 0, -pmin, 0.0)

    # For small x we rewrite eqn. (38) as
    #                   p
    #  H(-p,x) = 1/2 sum    binom(p,k) (-1)^k \tilde{d}(k+1/2,-x)
    #                   k=0
    #
    for p in range(1,-pmin+1):
        acc = 0.0
        # y holds B_{p,k} = binomial(p,k) (-1)^k
        y = 1.0
        for k in range(0, p+1):
            acc += y * d_tilde[k]
            # B_{p,k+1} = (-1) (p-k)/(k+1) B_{p,k}
            y *= (-1) * (p-k)/(k+1)
        acc *= 0.5

        h[-p] = acc

    return h
    
    

def partition3(l):
    """
    enumerate all partitions of the integer l into 3 integers nx, ny, nz
    such that nx+ny+nz = l
    """
    for nx in range(0, l+1):
        for ny in range(0, l-nx+1):
            nz = l-nx-ny
            yield (nx,ny,nz)

def kappa(n):
    if (n % 2 == 0):
        return n/2
    else:
        return (n+1)/2

    
### TESTS ###
import unittest

class TestSchwedtfegerSpecialFunctions(unittest.TestCase):
    def test_d_func(self):
        print("testing d(p+1/2,x)")

        pmin, pmax = -10, 10
        # random values for testing
        x = 0.2342
        w0 = 0.647

        # faster iterative algorithm computes d(p+1/2,x) for all p's in one go
        d = d_func(x, pmin, pmax, w0)
        print("iterative implementation      : ", d)
        
        # Take Xiao's implementation of d_func according to eqns. (25) and (29) as reference
        d_reference = np.zeros(-pmin+pmax+1)
        for p in range(pmin,pmax+1):
            d_reference[p] = d_func_reference(p+0.5, x, -w0)
        print("reference implementation      : ", d_reference)

        # compare the two implementations
        self.assertLess(la.norm(d - d_reference), 1.0e-8)

    def test_d_func_zero_limit(self):
        """
        check that \tilde{d} satisfies

                         p+1/2   ~
           d(p+1/2,x) = x        d(p+1/2,x)

        """
        print("testing \tilde{d}(p+1/2,x)")

        pmin, pmax = -10, 10
        # random values for testing
        x = 0.02342
        w0 = 0.647
        
        d = d_func(x, pmin, pmax, w0)
        d_tilde = d_func_zero_limit(x, pmin, pmax, w0)
        
        for p in range(pmin, pmax+1):
            dp = pow(x, p+0.5) * d_tilde[p]
            #print("p= %+4.d    %+12.8e   ?=   %+12.8e    absolute error= %+12.8e     relative error= %+12.8e" % (p, d[p], dp, abs(dp - d[p]), abs(dp - d[p])/abs(d[p]) ) )
            with self.subTest(p=p):
                # absolute error < 10^-10 or relative error < 10^-7
                self.assertTrue( abs(dp - d[p]) < 1.0e-10 or abs(dp - d[p])/abs(d[p]) < 1.0e-7 )

    def test_d_func_zero_limit_cpp(self):
        """
        test C++ implementation of d_func_zero_limit
        """
        pmin, pmax = -10, 10
        # random values for testing
        x = 0.02342
        w0 = 0.647

        d_tilde = d_func_zero_limit(x, pmin, pmax, w0)
        # compare python and C++ implementations
        for p in range(pmin, pmax+1):
            # call C++ extension
            d_tilde_p = _polarization.test_d_func_zero_limit(x, p, w0)
            
            #print("p= %+4.d    %+12.8e   ?=   %+12.8e    absolute error= %+12.8e     relative error= %+12.8e" % (p, d_tilde[p], d_tilde_p, abs(d_tilde[p] - d_tilde_p), abs(d_tilde[p] - d_tilde_p)/abs(d_tilde[p]) ) )
            with self.subTest(p=p):
                self.assertTrue( abs(d_tilde[p] - d_tilde_p) < 1.0e-10 )
                
    def test_g_func(self):
        print("testing gamma(p+1/2,x)")
        pmax = 10
        # random values for testing
        x = 0.235235

        # faster iterative algorithm computes g(p+1/2,x) for all p's in one go
        g = g_func(x, pmax)
        print("iterative implementation      : ", g)

        # gamma(p+1/2,x) can be written in terms of the Gamma function G(a) and the regularized
        # lower incomplete Gamma function P(a,x):
        #  gamma(p+1/2,x) = Gamma(p+1/2) P(p+1/2,x)
        g_reference = np.zeros(pmax+1)
        for p in range(0, pmax+1):
            g_reference[p] = special.gamma(p+0.5) * special.gammainc(p+0.5, x)
        print("reference implementation      : ", g_reference)

        # compare the two implementations
        self.assertLess(la.norm(g - g_reference), 1.0e-8)

    def test_g_func_cpp(self):
        print("testing C++ implementation gamma(p+1/2,x)")
        pmax = 10
        # random values for testing
        x = 0.86725

        # gamma(p+1/2,x) can be written in terms of the Gamma function G(a) and the regularized
        # lower incomplete Gamma function P(a,x):
        #  gamma(p+1/2,x) = Gamma(p+1/2) P(p+1/2,x)
        g_reference = np.zeros(pmax+1)
        g = np.zeros(pmax+1)
        for p in range(0, pmax+1):
            g[p] = _polarization.test_g_func(x, p)
            g_reference[p] = special.gamma(p+0.5) * special.gammainc(p+0.5, x)
        print("C++ implementation            : ", g)
        print("reference implementation      : ", g_reference)

        # compare the two implementations
        self.assertLess(la.norm(g - g_reference), 1.0e-8)

        
    def test_binomial_sums(self):
        """
        check that the iteration
                                         x k
            B    = 1        ;  B     = ---------  B
             n,n                n,k-1  n - k + 1   n,k

        produces the terms
                                 n-k
           B    = binomial(n,k) x
            n,k

        in reverse order k=n,n-1,...,0
        """
        # random values for testing
        n = 10
        x = 0.6262

        # initialization  B_{n,n}
        B = 1
        self.assertAlmostEqual(B, special.comb(n,n) * x**0)
        
        for k in range(n,0,-1):
            # B_{n,k-1}
            B *= (x * k)/(n-k+1)
            
            self.assertAlmostEqual(B, special.comb(n,k-1) * x**(n-(k-1)))

    def test_m_func(self):
        """
        check implementation of Dyson-error hybrid function m(x) 
        for small and large x by comparison with Mathematica
        """
        print("test Dawson-error hybrid function m(x)")
        # test points lie half way between expansion points, n/2+1/4
        xs_reference = np.array([
            0.25, 0.75, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75, 4.25, 4.75, 5.25,
            5.75, 6.25, 6.75, 7.25, 7.75, 15.0
        ])
        # The exact values of m(x) were computed as a reference
        # with the Mathematica script Schwerdtfeger_Dawson-error_hybrid_integral.nb
        ms_reference = np.array([
            0.03600888034034016, 0.3869800095043523, 1.619108065804630,
            6.777147720620485, 39.49564425980311, 379.6523082274102,
            6283.378330244620, 177564.3666528541, 8.472742354715874e6,
             6.780679537674443e8, 9.064107025088944e10,
             2.018368353479387e13, 7.472526220585603e15,
             4.593174392010666e18, 4.682417382497857e21,
             7.909964003438566e24, 1.738231811666687e96
        ])
        # compute m(x) using own implementation
        ms = np.array([ m_func(x) for x in xs_reference ])

        print("Taylor series + asymptotic : \n", ms)
        print("reference implementation (Mathematica)  : \n", ms_reference)

        relative_error = abs(ms-ms_reference)/abs(ms_reference)
        print("relative error : \n", relative_error)
        # relative error should be smaller than 10^-10 at all test points
        self.assertTrue( (relative_error < 1.0e-10).all() )
        
        def plot_m_function():
            import matplotlib
            matplotlib.rc('xtick', labelsize=16)
            matplotlib.rc('ytick', labelsize=16)
            matplotlib.rc('legend', fontsize=16)
            matplotlib.rc('axes', labelsize=16)
            
            import matplotlib.pyplot as plt

            npts = 1000
            xs = np.linspace(0.0, 15.5, npts)
            ms = np.zeros(npts)
            for i,x in enumerate(xs):
                ms[i] = m_func(x)

            plt.plot(xs_reference, ms_reference, "o", alpha=0.5, label="exact (Mathematica)")
            plt.plot(xs, ms, lw=2, label="this implementation")
            plt.xlabel("$x$")
            plt.ylabel("$m(x)$  (log)")
            plt.yscale('log')

            plt.legend()
            plt.tight_layout()

            plt.savefig("/tmp/dawson-error_hybrid_m-function.svg")
            plt.savefig("/tmp/dawson-error_hybrid_m-function.png", dpi=300)
            plt.show()

        # Disable plotting if test suite is run on cluster without graphics
        #plot_m_function()

    def test_m_func_cpp(self):
        """check C++ implementation of m(x)"""
        # test points lie half way between expansion points, n/2+1/4
        xs_reference = np.array([
            0.25, 0.75, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75, 4.25, 4.75, 5.25,
            5.75, 6.25, 6.75, 7.25, 7.75, 15.0
        ])
        # The exact values of m(x) were computed as a reference
        # with the Mathematica script Schwerdtfeger_Dawson-error_hybrid_integral.nb
        ms_reference = np.array([
            0.03600888034034016, 0.3869800095043523, 1.619108065804630,
            6.777147720620485, 39.49564425980311, 379.6523082274102,
            6283.378330244620, 177564.3666528541, 8.472742354715874e6,
             6.780679537674443e8, 9.064107025088944e10,
             2.018368353479387e13, 7.472526220585603e15,
             4.593174392010666e18, 4.682417382497857e21,
             7.909964003438566e24, 1.738231811666687e96
        ])
        # compute m(x) using own C++ implementation
        ms = np.array([ _polarization.m_func(x) for x in xs_reference ])
        
        # compute m(x) using own implementation
        ms = np.array([ m_func(x) for x in xs_reference ])

        print("Taylor series + asymptotic (C++)        : \n", ms)
        print("reference implementation (Mathematica)  : \n", ms_reference)

        relative_error = abs(ms-ms_reference)/abs(ms_reference)
        print("relative error : \n", relative_error)
        # relative error should be smaller than 10^-10 at all test points
        self.assertTrue( (relative_error < 1.0e-10).all() )
        
    def test_h_func(self):
        """
        check implementation of H(p,x) for positive and negative p
        """
        print("testing H(p,x)")

        pmin, pmax = -10, 10
        # random values for testing, sqrt(x) needs to be larger than 6, since
        # Xiao's h_func uses
        #   m(sqrt(x)) = exp(-x) erf(sqrt(x)) dawson(sqrt(x))
        # which is correct only for sqrt(x) > 6
        x = 6.5**2

        # faster iterative implementation
        h = h_func(x, pmin, pmax)

        print("iterative implementation      : ", h)
        
        # Xiao's recursive implementation only works for p >= 0
        h_reference = np.zeros(-pmin+pmax+1)
        for p in range(0,pmax+1):
            h_reference[p] = h_func_reference(p, x)

        # for negative p we use eqn. (38) directly
        g = g_func(x,pmax)
        for p in range(1,-pmin+1):
            for nu in range(0,p+1):
                h_reference[-p] += 0.5 * special.comb(p,nu) * (-1)**nu * x**(-nu-0.5) * g[nu]
            
        print("reference implementation      : ", h_reference)

        # compare the two implementations
        self.assertLess(la.norm(h - h_reference), 1.0e-7)

    def test_h_func_numerical(self):
        """
        compare analytical implementation of
 
                   /1          p
         H(-p,x) = | dt (1-t^2)  exp(-x t^2) 
                   /0
        
        with numerical integrals obtained using Mathematica
        """
        p = -2
        x = 2.253

        h = h_func(x, p, 0)[p]

        # got this with Mathematica
        h_numerical = 0.409357
        
        self.assertLess(abs(h - h_numerical), 1.0e-6)
        
    def test_h_func_cpp(self):
        """
        check C++ implementation of H(p,x) for positive and negative p
        """
        print("testing C++ implementation for H(p,x)")

        pmin, pmax = -10, 10
        # random values for testing, sqrt(x) needs to be larger than 6, since
        # Xiao's h_func uses
        #   m(sqrt(x)) = exp(-x) erf(sqrt(x)) dawson(sqrt(x))
        # which is correct only for sqrt(x) > 6
        x = 6.5**2

        # use python implementation as reference
        h_reference = h_func(x, pmin, pmax)

        print("python implementation      : ", h_reference)
        
        # compare with C++ implementation
        h = np.zeros(-pmin+pmax+1)
        for p in range(pmin, pmax+1):
            h[p] = _polarization.test_h_func(x, p)
            
        print("C++ implementation         : ", h)

        # compare the two implementations
        self.assertLess(la.norm(h - h_reference), 1.0e-7)

    def test_h_func_small_x(self):
        """
        check H(p,x) for x -> 0
        """
        pmin, pmax = -10, 10
        # random value
        # For x < 1.0, the small x expansion has a much higher accuracy than the h_func.
        x = 1.9515253

        h_reference = h_func(x, pmin, pmax)
        h_small_x = h_func_small_x(x, pmin, pmax)

        print("H(x,p) (reference)     = ", h_reference)
        print("H(x,p) (small x expr.) = ", h_small_x)
        
        self.assertLess(la.norm(h_small_x - h_reference), 1.0e-10)

    def test_h_func_small_x_cpp(self):
        """
        compary python and C++ implementations for `h_func_small`
        """
        pmin, pmax = -10, 10
        # random value
        # For x < 1.0, the small x expansion has a much higher accuracy than the h_func.
        x = 1.9515253

        # 
        h_reference = h_func_small_x(x, pmin, pmax)

        #
        h = np.zeros(-pmin+pmax+1)
        for p in range(pmin, pmax+1):
            h[p] = _polarization.test_h_func_small_x(x, p)

        print("H(x,p) (python) = ", h_reference)
        print("H(x,p) (C++)    = ", h)
        
        self.assertLess(la.norm(h - h_reference), 1.0e-10)
        
    def test_factorial2_even(self):
        """
        check that for even k

           (k-1)!!/2^(k/2)

        can be calculated iteratively from

          f[0] = 1
          f[1] = 1/2
          f[i+1] = (i+1/2) f[i]

        as f[k//2].
        """
        l_max = 10
        f = np.zeros(l_max+1)
        f[0] = 1.0
        f[1] = 0.5
        for i in range(1, (l_max+1)//2):
            f[i+1] = (i+0.5)*f[i]
            
        for k in range(0, l_max):
            if k % 2 == 0:
                self.assertLess( abs( f[k//2] - special.factorial2(k-1)/pow(2,k//2) ), 1.0e-10 )

class TestPolarizationIntegrals(unittest.TestCase):
    def test_polarization_integrals_case_1(self):
        """
        compare polarization integrals for the case `k=2*j` with the reference implementation
        """
        # make random numbers reproducible
        np.random.seed(0)

        # centers of Gaussian orbitals        
        xi,yi,zi = 2.0 * 2.0*np.array(np.random.rand(3)-0.5)
        xj,yj,zj = 2.0 * 2.0*np.array(np.random.rand(3)-0.5)
        # exponents of radial parts of Gaussians
        beta_i = 0.234
        beta_j = 1.255
        # cutoff function
        alpha = 50.0

        ## Check integrals up to g-functions
        #l_max = 4
        # Check integrals up to d-function
        l_max = 2
        
        # enumerate powers of polarization operator
        for k in [4]:  # [4,6]:
            for mx in [0,1,2]:
                for my in [0,1,2]:
                    for mz in [0,1,2]:
                        # power of cutoff function
                        q = max(2, int(kappa(k) - kappa(mx) - kappa(my) + kappa(mz) - 1))
                        # enumerate angular momenta of basis functions
                        for li in range(0, l_max+1):
                            for lj in range(0, l_max+1):
                                # prepare for integrals between shells with angular momenta li and lj
                                pol = PolarizationIntegral(xi,yi,zi, li, beta_i,
                                                           xj,yj,zj, lj, beta_j,
                                                           k, mx,my,mz,
                                                           alpha, q)
                                for nxi,nyi,nzi in partition3(li):
                                    for nxj,nyj,nzj in partition3(lj):
                                        label=f"k={k} mx={mx} my={my} mz={mz} , q={q} , nxi={nxi} nyi={nyi} nzi={nzi} , nxj={nxj} nyj={nyj} nzj={nzj}"
                                        with self.subTest(label=label):
                                            # fast C++ implementation
                                            pint = pol.compute_pair(nxi,nyi,nzi,
                                                                    nxj,nyj,nzj)
                                            
                                            # slow python implementation
                                            pint_reference = polarization_integral_reference(xi,yi,zi, nxi,nyi,nzi, beta_i,
                                                                                             xj,yj,zj, nxj,nyj,nzj, beta_j,
                                                                                             k, mx,my,mz,
                                                                                             alpha, q)

                                            print(label + f"  integral= {pint:+12.7f}  reference= {pint_reference:+12.7f}")
                                            absolute_error = abs(pint - pint_reference)
                                            relative_error = abs(pint - pint_reference)/abs(pint_reference)
                                            self.assertTrue( (absolute_error < 1.0e-7) or (relative_error < 1.0e-7) )

    def test_polarization_integrals_case_1_small_b(self):
        """
        compare polarization integrals for the case `k=2*j` and the limit b -> 0 
        with the reference implementation.

        b=0 cannot be checked because this means dividing by 0 in reference implementation.
        """
        # make random numbers reproducible
        np.random.seed(0)

        # First center
        xi,yi,zi = 2.0 * 2.0*np.array(np.random.rand(3)-0.5)
        # exponents of radial parts of Gaussians
        beta_i = 0.234
        beta_j = 1.255
        # The second center is chosen such that
        #  b = |beta_i * ri + beta_j * rj| = eps, such that x = b^2 / alpha << 1
        eps = 1.0
        xj,yj,zj = - beta_i/beta_j * np.array([xi,yi,zi]) + eps * 2.0*(np.random.rand(3)-0.5)
        # cutoff function
        alpha = 50.0

        ## Check integrals up to g-functions
        #l_max = 4
        # Check integrals up to d-function
        l_max = 2
        
        # enumerate powers of polarization operator
        for k in [4]: #[4,6]:
            for mx in [0,1]:
                for my in [0,1]:
                    for mz in [0,1]:
                        # power of cutoff function
                        q = max(2, int(kappa(k) - kappa(mx) - kappa(my) + kappa(mz) - 1))
                        # enumerate angular momenta of basis functions
                        for li in range(0, l_max+1):
                            for lj in range(0, l_max+1):
                                # prepare for integrals between shells with angular momenta li and lj
                                pol = PolarizationIntegral(xi,yi,zi, li, beta_i,
                                                           xj,yj,zj, lj, beta_j,
                                                           k, mx,my,mz,
                                                           alpha, q)
                                for nxi,nyi,nzi in partition3(li):
                                    for nxj,nyj,nzj in partition3(lj):
                                        label=f"k={k} mx={mx} my={my} mz={mz} , q={q} , nxi={nxi} nyi={nyi} nzi={nzi} , nxj={nxj} nyj={nyj} nzj={nzj}"
                                        with self.subTest(label=label):
                                            # fast C++ implementation
                                            pint = pol.compute_pair(nxi,nyi,nzi,
                                                                    nxj,nyj,nzj)
                                            
                                            # slow python implementation
                                            pint_reference = polarization_integral_reference(xi,yi,zi, nxi,nyi,nzi, beta_i,
                                                                                             xj,yj,zj, nxj,nyj,nzj, beta_j,
                                                                                             k, mx,my,mz,
                                                                                             alpha, q)

                                            print(label + f"  integral= {pint:+12.7f}  reference= {pint_reference:+12.7f}")
                                            absolute_error = abs(pint - pint_reference)
                                            relative_error = abs(pint - pint_reference)/abs(pint_reference)
                                            self.assertTrue( (absolute_error < 1.0e-7) or (relative_error < 1.0e-7) )

    def test_polarization_integrals_case_1_numerical(self):
        """
        compare polarization integrals for the case `k=2*j` with the numerical integrals (if available)
        """
        try:
            from polarization_ints_numerical import polarization_integral as polarization_integral_numerical
        except ImportError as err:
            print("numerical integrals not available, skip this test")
            return

        # increase resolution of integration grids
        from becke import settings
        settings.radial_grid_factor = 3  # increase number of radial points by factor 3
        settings.lebedev_order = 23      # angular Lebedev grid of order 23

        
        # make random numbers reproducible
        np.random.seed(0)

        # centers of Gaussian orbitals        
        xi,yi,zi = 2.0 * 2.0*np.array(np.random.rand(3)-0.5)
        xj,yj,zj = 2.0 * 2.0*np.array(np.random.rand(3)-0.5)
        # exponents of radial parts of Gaussians
        beta_i = 0.234
        beta_j = 1.255
        # cutoff function
        alpha = 50.0

        ## Check integrals up to g-functions
        #l_max = 4
        # Check integrals up to d-function
        l_max = 2
        
        # enumerate powers of polarization operator
        for k in [4]:  # [4,6]:
            for mx in [0,1,2]:
                for my in [0,1,2]:
                    for mz in [0,1,2]:
                        # power of cutoff function
                        q = max(2, int(kappa(k) - kappa(mx) - kappa(my) + kappa(mz) - 1))
                        # enumerate angular momenta of basis functions
                        for li in range(0, l_max+1):
                            for lj in range(0, l_max+1):
                                # prepare for integrals between shells with angular momenta li and lj
                                pol = PolarizationIntegral(xi,yi,zi, li, beta_i,
                                                           xj,yj,zj, lj, beta_j,
                                                           k, mx,my,mz,
                                                           alpha, q)
                                for nxi,nyi,nzi in partition3(li):
                                    for nxj,nyj,nzj in partition3(lj):
                                        label=f"k={k} mx={mx} my={my} mz={mz} , q={q} , nxi={nxi} nyi={nyi} nzi={nzi} , nxj={nxj} nyj={nyj} nzj={nzj}"
                                        with self.subTest(label=label):
                                            # fast C++ implementation
                                            pint = pol.compute_pair(nxi,nyi,nzi,
                                                                    nxj,nyj,nzj)
                                            
                                            # slow numerical integrals
                                            pint_numerical = polarization_integral_numerical(xi,yi,zi, nxi,nyi,nzi, beta_i,
                                                                                             xj,yj,zj, nxj,nyj,nzj, beta_j,
                                                                                             k, mx,my,mz,
                                                                                             alpha, q)

                                            print(label + f"  integral= {pint:+12.7f}  numerical= {pint_numerical:+12.7f}")
                                            absolute_error = abs(pint - pint_numerical)
                                            relative_error = abs(pint - pint_numerical)/abs(pint_numerical)
                                            # numerical accuracy is quite low
                                            self.assertTrue( (absolute_error < 1.0e-3) or (relative_error < 1.0e-3) )

    def test_polarization_integrals_case_2a(self):
        """
        compare polarization integrals for the case `k=2*j+1` with s-j >= 0 
        with the reference implementation (case 1, subcase 2)
        """
        # make random numbers reproducible
        np.random.seed(0)

        # centers of Gaussian orbitals        
        xi,yi,zi = 2.0 * 2.0*np.array(np.random.rand(3)-0.5)
        xj,yj,zj = 2.0 * 2.0*np.array(np.random.rand(3)-0.5)+1.0
        # exponents of radial parts of Gaussians
        beta_i = 0.234
        beta_j = 1.255
        # cutoff function
        alpha = 50.0

        ## Check integrals up to g-functions
        #l_max = 4
        # Check integrals up to d-function
        l_max = 2
        
        # enumerate powers of polarization operator
        for k in [3]:
            for mx,my,mz in partition3(1):
                # Since k = 3 and s >= mx+my+mz = 1, we have j = 1 and s-j >= 0
                # power of cutoff function
                q = max(2, int(kappa(k) - kappa(mx) - kappa(my) + kappa(mz) - 1))
                # enumerate angular momenta of basis functions
                for li in range(0, l_max+1):
                    for lj in range(0, l_max+1):
                        # prepare for integrals between shells with angular momenta li and lj
                        pol = PolarizationIntegral(xi,yi,zi, li, beta_i,
                                                   xj,yj,zj, lj, beta_j,
                                                   k, mx,my,mz,
                                                   alpha, q)
                        for nxi,nyi,nzi in partition3(li):
                            for nxj,nyj,nzj in partition3(lj):
                                label=f"k={k} mx={mx} my={my} mz={mz} , q={q} , nxi={nxi} nyi={nyi} nzi={nzi} , nxj={nxj} nyj={nyj} nzj={nzj}"
                                with self.subTest(label=label):
                                    # fast C++ implementation
                                    pint = pol.compute_pair(nxi,nyi,nzi,
                                                            nxj,nyj,nzj)
                                            
                                    # slow python implementation
                                    pint_reference = polarization_integral_reference(xi,yi,zi, nxi,nyi,nzi, beta_i,
                                                                                     xj,yj,zj, nxj,nyj,nzj, beta_j,
                                                                                     k, mx,my,mz,
                                                                                     alpha, q)
                                    print(label + f"  integral= {pint:+12.7f}  reference= {pint_reference:+12.7f}")
                                    absolute_error = abs(pint - pint_reference)
                                    relative_error = abs(pint - pint_reference)/abs(pint_reference)
                                    self.assertTrue( (absolute_error < 1.0e-7) or (relative_error < 1.0e-7) )
                                    
    def test_polarization_integrals_case_2a_numerical(self):
        """
        compare polarization integrals for the case `k=2*j+1` with s-j >= 0 
        with the numerical integrals (case 1, subcase 2)
        """
        try:
            from polarization_ints_numerical import polarization_integral as polarization_integral_numerical
        except ImportError as err:
            print("numerical integrals not available, skip this test")
            return

        # increase resolution of integration grids
        from becke import settings
        settings.radial_grid_factor = 3  # increase number of radial points by factor 3
        settings.lebedev_order = 23      # angular Lebedev grid of order 23


        # make random numbers reproducible
        np.random.seed(0)

        # centers of Gaussian orbitals        
        xi,yi,zi = 2.0 * 2.0*np.array(np.random.rand(3)-0.5)
        xj,yj,zj = 2.0 * 2.0*np.array(np.random.rand(3)-0.5)+1.0
        # exponents of radial parts of Gaussians
        beta_i = 0.234
        beta_j = 1.255
        # cutoff function
        alpha = 50.0

        ## Check integrals up to g-functions
        #l_max = 4
        # Check integrals up to d-function
        l_max = 2
        
        # enumerate powers of polarization operator
        for k in [3]:
            for mx,my,mz in partition3(1):
                # Since k = 3 and s >= mx+my+mz = 1, we have j = 1 and s-j >= 0
                # power of cutoff function
                q = max(2, int(kappa(k) - kappa(mx) - kappa(my) + kappa(mz) - 1))
                # enumerate angular momenta of basis functions
                for li in range(0, l_max+1):
                    for lj in range(0, l_max+1):
                        # prepare for integrals between shells with angular momenta li and lj
                        pol = PolarizationIntegral(xi,yi,zi, li, beta_i,
                                                   xj,yj,zj, lj, beta_j,
                                                   k, mx,my,mz,
                                                   alpha, q)
                        for nxi,nyi,nzi in partition3(li):
                            for nxj,nyj,nzj in partition3(lj):
                                label=f"k={k} mx={mx} my={my} mz={mz} , q={q} , nxi={nxi} nyi={nyi} nzi={nzi} , nxj={nxj} nyj={nyj} nzj={nzj}"
                                with self.subTest(label=label):
                                    # fast C++ implementation
                                    pint = pol.compute_pair(nxi,nyi,nzi,
                                                            nxj,nyj,nzj)
                                            
                                    # slow numerical integrals
                                    pint_numerical = polarization_integral_numerical(xi,yi,zi, nxi,nyi,nzi, beta_i,
                                                                                     xj,yj,zj, nxj,nyj,nzj, beta_j,
                                                                                     k, mx,my,mz,
                                                                                     alpha, q)
                                    
                                    print(label + f"  integral= {pint:+12.7f}  numerical= {pint_numerical:+12.7f}")
                                    absolute_error = abs(pint - pint_numerical)
                                    relative_error = abs(pint - pint_numerical)/abs(pint_numerical)
                                    # numerical accuracy is quite low
                                    self.assertTrue( (absolute_error < 1.0e-3) or (relative_error < 1.0e-3) )

    def test_polarization_integrals_case_2a_small_b(self):
        """
        compare polarization integrals for the case `k=2*j+1` with s-j >= 0 
        with the reference implementation (case 1, subcase 2) for the limit b -> 0
        """
        # make random numbers reproducible
        np.random.seed(0)

        # First center
        xi,yi,zi = 2.0 * 2.0*np.array(np.random.rand(3)-0.5)
        # exponents of radial parts of Gaussians
        beta_i = 0.234
        beta_j = 1.255
        # The second center is chosen such that
        #  b = |beta_i * ri + beta_j * rj| = eps, such that x = b^2 / alpha << 1
        eps = 1.0
        xj,yj,zj = - beta_i/beta_j * np.array([xi,yi,zi]) + eps * 2.0*(np.random.rand(3)-0.5)
        # cutoff function
        alpha = 50.0

        ## Check integrals up to g-functions
        #l_max = 4
        # Check integrals up to d-function
        l_max = 2
        
        # enumerate powers of polarization operator
        for k in [3]:
            for mx,my,mz in partition3(1):
                # Since k = 3 and s >= mx+my+mz = 1, we have j = 1 and s-j >= 0
                # power of cutoff function
                q = max(2, int(kappa(k) - kappa(mx) - kappa(my) + kappa(mz) - 1))
                # enumerate angular momenta of basis functions
                for li in range(0, l_max+1):
                    for lj in range(0, l_max+1):
                        # prepare for integrals between shells with angular momenta li and lj
                        pol = PolarizationIntegral(xi,yi,zi, li, beta_i,
                                                   xj,yj,zj, lj, beta_j,
                                                   k, mx,my,mz,
                                                   alpha, q)
                        for nxi,nyi,nzi in partition3(li):
                            for nxj,nyj,nzj in partition3(lj):
                                label=f"k={k} mx={mx} my={my} mz={mz} , q={q} , nxi={nxi} nyi={nyi} nzi={nzi} , nxj={nxj} nyj={nyj} nzj={nzj}"
                                with self.subTest(label=label):
                                    # fast C++ implementation
                                    pint = pol.compute_pair(nxi,nyi,nzi,
                                                            nxj,nyj,nzj)
                                            
                                    # slow python implementation
                                    pint_reference = polarization_integral_reference(xi,yi,zi, nxi,nyi,nzi, beta_i,
                                                                                     xj,yj,zj, nxj,nyj,nzj, beta_j,
                                                                                     k, mx,my,mz,
                                                                                     alpha, q)
                                    print(label + f"  integral= {pint:+12.7f}  reference= {pint_reference:+12.7f}")
                                    absolute_error = abs(pint - pint_reference)
                                    relative_error = abs(pint - pint_reference)/abs(pint_reference)
                                    self.assertTrue( (absolute_error < 1.0e-7) or (relative_error < 1.0e-7) )

    def test_polarization_integrals_case_2b_numerical(self):
        """
        compare polarization integrals for the case `k=2*j+1` with s-j < 0 
        with numerical integrals (case 1, subcase 2)
        """
        try:
            from polarization_ints_numerical import polarization_integral as polarization_integral_numerical
        except ImportError as err:
            print("numerical integrals not available, skip this test")
            return

        # increase resolution of integration grids
        from becke import settings
        settings.radial_grid_factor = 3  # increase number of radial points by factor 3
        settings.lebedev_order = 23      # angular Lebedev grid of order 23


        # make random numbers reproducible
        np.random.seed(0)

        # centers of Gaussian orbitals        
        xi,yi,zi = 2.0 * 2.0*np.array(np.random.rand(3)-0.5)
        xj,yj,zj = 2.0 * 2.0*np.array(np.random.rand(3)-0.5)+3.0
        # exponents of radial parts of Gaussians
        beta_i = 0.234
        beta_j = 1.255
        # cutoff function
        alpha = 50.0

        ## Check integrals up to g-functions
        #l_max = 4
        # Check integrals up to d-function
        l_max = 2
        
        # enumerate powers of polarization operator
        for k in [3]:
            for mx,my,mz in partition3(0):
                # Since k = 3 and s >= mx+my+mz = 1, we have j = 1 and s-j >= 0
                # power of cutoff function
                q = max(2, int(kappa(k) - kappa(mx) - kappa(my) + kappa(mz) - 1))
                # enumerate angular momenta of basis functions
                for li in range(0, l_max+1):
                    for lj in range(0, l_max+1):
                        # prepare for integrals between shells with angular momenta li and lj
                        pol = PolarizationIntegral(xi,yi,zi, li, beta_i,
                                                   xj,yj,zj, lj, beta_j,
                                                   k, mx,my,mz,
                                                   alpha, q)
                        for nxi,nyi,nzi in partition3(li):
                            for nxj,nyj,nzj in partition3(lj):
                                label=f"k={k} mx={mx} my={my} mz={mz} , q={q} , nxi={nxi} nyi={nyi} nzi={nzi} , nxj={nxj} nyj={nyj} nzj={nzj}"
                                with self.subTest(label=label):
                                    # fast C++ implementation
                                    pint = pol.compute_pair(nxi,nyi,nzi,
                                                            nxj,nyj,nzj)
                                            
                                    # slow numerical integrals
                                    pint_numerical = polarization_integral_numerical(xi,yi,zi, nxi,nyi,nzi, beta_i,
                                                                                     xj,yj,zj, nxj,nyj,nzj, beta_j,
                                                                                     k, mx,my,mz,
                                                                                     alpha, q)
                                    
                                    # slow python implementation
                                    pint_reference = polarization_integral_reference(xi,yi,zi, nxi,nyi,nzi, beta_i,
                                                                                     xj,yj,zj, nxj,nyj,nzj, beta_j,
                                                                                     k, mx,my,mz,
                                                                                     alpha, q)

                                    print(label + f"  integral= {pint:+12.7f}  numerical= {pint_numerical:+12.7f}  reference= {pint_reference:+12.7f}")
                                    absolute_error = abs(pint - pint_numerical)
                                    relative_error = abs(pint - pint_numerical)/abs(pint_numerical)
                                    # numerical accuracy is quite low
                                    self.assertTrue( (absolute_error < 1.0e-3) or (relative_error < 1.0e-3) )
    def test_polarization_integrals_case_2b_small_b_numerical(self):
        """
        compare polarization integrals for the case `k=2*j+1` with s-j < 0 
        with the numerical integrals for the limit b -> 0
        """
        try:
            from polarization_ints_numerical import polarization_integral as polarization_integral_numerical
        except ImportError as err:
            print("numerical integrals not available, skip this test")
            return

        # increase resolution of integration grids
        from becke import settings
        settings.radial_grid_factor = 3  # increase number of radial points by factor 3
        settings.lebedev_order = 23      # angular Lebedev grid of order 23

        # make random numbers reproducible
        np.random.seed(0)

        # First center
        xi,yi,zi = 2.0 * 2.0*np.array(np.random.rand(3)-0.5)
        # exponents of radial parts of Gaussians
        beta_i = 0.234
        beta_j = 1.255
        # The second center is chosen such that
        #  b = |beta_i * ri + beta_j * rj| = eps, such that x = b^2 / alpha << 1
        eps = 1.0
        xj,yj,zj = - beta_i/beta_j * np.array([xi,yi,zi]) + eps * 2.0*(np.random.rand(3)-0.5)
        # cutoff function
        alpha = 50.0

        ## Check integrals up to g-functions
        #l_max = 4
        # Check integrals up to d-function
        l_max = 2
        
        # enumerate powers of polarization operator
        for k in [3]:
            for mx,my,mz in partition3(0):
                # Since k = 3 and s >= mx+my+mz = 0, we have j = 1 and s-j < 0 for some integrals
                # power of cutoff function
                q = max(2, int(kappa(k) - kappa(mx) - kappa(my) + kappa(mz) - 1))
                # enumerate angular momenta of basis functions
                for li in range(0, l_max+1):
                    for lj in range(0, l_max+1):
                        # prepare for integrals between shells with angular momenta li and lj
                        pol = PolarizationIntegral(xi,yi,zi, li, beta_i,
                                                   xj,yj,zj, lj, beta_j,
                                                   k, mx,my,mz,
                                                   alpha, q)
                        for nxi,nyi,nzi in partition3(li):
                            for nxj,nyj,nzj in partition3(lj):
                                label=f"k={k} mx={mx} my={my} mz={mz} , q={q} , nxi={nxi} nyi={nyi} nzi={nzi} , nxj={nxj} nyj={nyj} nzj={nzj}"
                                with self.subTest(label=label):
                                    # fast C++ implementation
                                    pint = pol.compute_pair(nxi,nyi,nzi,
                                                            nxj,nyj,nzj)

                                    # slow numerical integrals
                                    pint_numerical = polarization_integral_numerical(xi,yi,zi, nxi,nyi,nzi, beta_i,
                                                                                     xj,yj,zj, nxj,nyj,nzj, beta_j,
                                                                                     k, mx,my,mz,
                                                                                     alpha, q)
                                    
                                    print(label + f"  integral= {pint:+12.7f}  numerical= {pint_numerical:+12.7f}")
                                    absolute_error = abs(pint - pint_numerical)
                                    relative_error = abs(pint - pint_numerical)/abs(pint_numerical)
                                    # numerical accuracy is quite low
                                    self.assertTrue( (absolute_error < 1.0e-3) or (relative_error < 1.0e-3) )
                                    
    def test_000_integral(self):
        """
        test integrals for case 2b for b=0
        """
        try:
            from polarization_ints_numerical import polarization_integral as polarization_integral_numerical
        except ImportError as err:
            print("numerical integrals not available, skip this test")
            return

        # increase resolution of integration grids
        from becke import settings
        settings.radial_grid_factor = 3  # increase number of radial points by factor 3
        settings.lebedev_order = 23      # angular Lebedev grid of order 23

        k = 3
        mx,my,mz = 0,0,0
        q = 2
        alpha = 1.0
        # two s-orbitals with exponent 1.0 at the origin
        xi,yi,zi = 0.0,0.0,0.00001
        nxi,nyi,nzi = 1,0,1
        beta_i = 2.0

        xj,yj,zj = 0.0,0.0001,0.0
        nxj,nyj,nzj = 1,0,1
        beta_j = 1.0

        li,lj = nxi+nyi+nzi,nxj+nyj+nzj

        pint_numerical = polarization_integral_numerical(xi,yi,zi, nxi,nyi,nzi, beta_i,
                                                         xj,yj,zj, nxj,nyj,nzj, beta_j,
                                                         k, mx,my,mz,
                                                         alpha, q)
        print(pint_numerical)

        pint_reference = polarization_integral_reference(xi,yi,zi, nxi,nyi,nzi, beta_i,
                                                         xj,yj,zj, nxj,nyj,nzj, beta_j,
                                                         k, mx,my,mz,
                                                         alpha, q)
        
        print(pint_reference)

        # prepare for integrals between shells with angular momenta li and lj
        pol = PolarizationIntegral(xi,yi,zi, li, beta_i,
                                   xj,yj,zj, lj, beta_j,
                                   k, mx,my,mz,
                                   alpha, q)

        # fast C++ implementation
        pint = pol.compute_pair(nxi,nyi,nzi,
                                nxj,nyj,nzj)

        print(pint)

    
if __name__ == '__main__':
    unittest.main()

        
