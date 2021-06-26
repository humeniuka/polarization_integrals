/* The following code has been generated automatically using the Mathematica script 'radial_integrals_upper_bound.nb'*/
#include <stdio.h>
#include <cassert>
#include <math.h>

template <typename real>
real radial_overlap(real xi, real yi, real zi, int li, real beta_i,
                    real xj, real yj, real zj, int lj, real beta_j) {

/*
    Analytical overlap integral between squares of spherically symmetric
    radial Gaussian basis functions of the form

                       li                      2
      RGTO (r) = (r-ri)    exp(-beta_i (r - ri) )
          i 

    The radial integrals

       (2)  /     2         2
      U   = | RGTO (r)  CGTO (r)
       ij   /     i         j

    are an upper bound for the integrals of the type

       (2)  /     2         2
      S   = | CGTO (r)  CGTO (r)
       ij   /     i         j

    between unnormalized primitive Cartesian basis functions of the form

                       nxi       nyi       nzi                     2
      CGTO (r) = (x-xi)    (y-yi)    (z-zi)    exp(-beta_i (r - ri) )
          i 

    with total angular momentum li = nxi+nyi+nzi.
*/
  const int lmax = 3;
  real dx = xi-xj;
  real dy = yi-yj;
  real dz = zi-zj;
  real r2_ij = dx*dx+dy*dy+dz*dz;
  real r_ij = sqrt(r2_ij);
  // compute gpow = (beta_i+beta_j)^(2*(li+lj)+3/2)
  real g = beta_i+beta_j;
  real sqg = sqrt(g);
  real g2=g*g;
  real gpow=g*sqg; // (beta_i+beta_j)^(3/2) 
  for(int n = 0; n < li+lj; n++) {
    gpow *= g2;
  }
  int pow2 = 1 << (2*(li+lj)+1);
  real prefactor = M_PI/M_2_SQRTPI*2 * M_SQRT1_2/pow2 * exp(- 2*beta_i*beta_j/g * r2_ij) / gpow;
  int lilj=li*(lmax+1)+lj;
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
  real olap;
  real bi[9];
  real bj[9];
  real r[12];
  switch(lilj) {
    case 0:
      // li=0 lj=0
      olap=1;
      break;
    case 1:
      // li=0 lj=1
      // precalculate powers of beta_i
      bi[0]=beta_i;
      bi[1]=bi[0]*beta_i;
      // precalculate powers of beta_j
      bj[0]=beta_j;
      // precalculate powers of R=|rj-ri|
      r[0]=r_ij;
      r[1]=r[0]*r_ij;
      olap=3*(bi[0] + bj[0]) + 4*bi[1]*r[1];
      break;
    case 2:
      // li=0 lj=2
      // precalculate powers of beta_i
      bi[0]=beta_i;
      bi[1]=bi[0]*beta_i;
      bi[2]=bi[1]*beta_i;
      bi[3]=bi[2]*beta_i;
      // precalculate powers of beta_j
      bj[0]=beta_j;
      bj[1]=bj[0]*beta_j;
      // precalculate powers of R=|rj-ri|
      r[0]=r_ij;
      r[1]=r[0]*r_ij;
      r[2]=r[1]*r_ij;
      r[3]=r[2]*r_ij;
      olap=30*bi[0]*bj[0] + 15*(bi[1] + bj[1]) + 40*(bj[0]*bi[1] + bi[2])*r[1] + 16*bi[3]*r[3];
      break;
    case 3:
      // li=0 lj=3
      // precalculate powers of beta_i
      bi[0]=beta_i;
      bi[1]=bi[0]*beta_i;
      bi[2]=bi[1]*beta_i;
      bi[3]=bi[2]*beta_i;
      bi[4]=bi[3]*beta_i;
      bi[5]=bi[4]*beta_i;
      // precalculate powers of beta_j
      bj[0]=beta_j;
      bj[1]=bj[0]*beta_j;
      bj[2]=bj[1]*beta_j;
      // precalculate powers of R=|rj-ri|
      r[0]=r_ij;
      r[1]=r[0]*r_ij;
      r[2]=r[1]*r_ij;
      r[3]=r[2]*r_ij;
      r[4]=r[3]*r_ij;
      r[5]=r[4]*r_ij;
      olap=315*bj[0]*bi[1] + 105*(bi[2] + 3*bi[0]*bj[1] + bj[2]) + 420*(2*bj[0]*bi[2] + bi[3] + bi[1]*bj[1])*r[1] + 336*(bj[0]*bi[3] + bi[4])*r[3] + 64*bi[5]*r[5];
      break;
    case 5:
      // li=1 lj=1
      // precalculate powers of beta_i
      bi[0]=beta_i;
      bi[1]=bi[0]*beta_i;
      bi[2]=bi[1]*beta_i;
      // precalculate powers of beta_j
      bj[0]=beta_j;
      bj[1]=bj[0]*beta_j;
      bj[2]=bj[1]*beta_j;
      // precalculate powers of R=|rj-ri|
      r[0]=r_ij;
      r[1]=r[0]*r_ij;
      r[2]=r[1]*r_ij;
      r[3]=r[2]*r_ij;
      olap=15*bj[1] + 12*(bi[2] + bj[2])*r[1] + bi[0]*(30*bj[0] - 4*bj[1]*r[1]) + bi[1]*(15 - 4*bj[0]*r[1] + 16*bj[1]*r[3]);
      break;
    case 6:
      // li=1 lj=2
      // precalculate powers of beta_i
      bi[0]=beta_i;
      bi[1]=bi[0]*beta_i;
      bi[2]=bi[1]*beta_i;
      bi[3]=bi[2]*beta_i;
      bi[4]=bi[3]*beta_i;
      // precalculate powers of beta_j
      bj[0]=beta_j;
      bj[1]=bj[0]*beta_j;
      bj[2]=bj[1]*beta_j;
      bj[3]=bj[2]*beta_j;
      // precalculate powers of R=|rj-ri|
      r[0]=r_ij;
      r[1]=r[0]*r_ij;
      r[2]=r[1]*r_ij;
      r[3]=r[2]*r_ij;
      r[4]=r[3]*r_ij;
      r[5]=r[4]*r_ij;
      olap=105*(bi[2] + 3*bi[0]*bj[1] + bj[2]) + 5*bj[0]*(63*bi[1] + 48*bi[2]*r[1] - 16*bi[3]*r[3]) + 4*(-5*(3*bi[1]*bj[1] + 2*bi[0]*bj[2] - 3*bj[3])*r[1] + 4*(3*bi[4] + 2*bi[2]*bj[1] + 10*bi[1]*bj[2])*r[3] + 2*bi[3]*(25*r[1] + 8*bj[1]*r[5]));
      break;
    case 7:
      // li=1 lj=3
      // precalculate powers of beta_i
      bi[0]=beta_i;
      bi[1]=bi[0]*beta_i;
      bi[2]=bi[1]*beta_i;
      bi[3]=bi[2]*beta_i;
      bi[4]=bi[3]*beta_i;
      bi[5]=bi[4]*beta_i;
      bi[6]=bi[5]*beta_i;
      // precalculate powers of beta_j
      bj[0]=beta_j;
      bj[1]=bj[0]*beta_j;
      bj[2]=bj[1]*beta_j;
      bj[3]=bj[2]*beta_j;
      bj[4]=bj[3]*beta_j;
      // precalculate powers of R=|rj-ri|
      r[0]=r_ij;
      r[1]=r[0]*r_ij;
      r[2]=r[1]*r_ij;
      r[3]=r[2]*r_ij;
      r[4]=r[3]*r_ij;
      r[5]=r[4]*r_ij;
      r[6]=r[5]*r_ij;
      r[7]=r[6]*r_ij;
      olap=945*(bi[3] + 6*bi[1]*bj[1] + 4*bi[0]*bj[2] + bj[3]) + 84*(5*(7*bi[4] + 10*bi[2]*bj[1] - 2*bi[1]*bj[2] - bi[0]*bj[3] + bj[4])*r[1] + 4*(5*bi[5] - 6*bi[3]*bj[1] + 2*bi[2]*bj[2] + 5*bi[1]*bj[3])*r[3] + bj[0]*(45*bi[2] + 85*bi[3]*r[1] + 8*bi[4]*r[3])) + 192*(-3*bj[0]*bi[5] + bi[6] + 3*bi[4]*bj[1] + 7*bi[3]*bj[2])*r[5] + 256*bi[5]*bj[1]*r[7];
      break;
    case 10:
      // li=2 lj=2
      // precalculate powers of beta_i
      bi[0]=beta_i;
      bi[1]=bi[0]*beta_i;
      bi[2]=bi[1]*beta_i;
      bi[3]=bi[2]*beta_i;
      bi[4]=bi[3]*beta_i;
      bi[5]=bi[4]*beta_i;
      // precalculate powers of beta_j
      bj[0]=beta_j;
      bj[1]=bj[0]*beta_j;
      bj[2]=bj[1]*beta_j;
      bj[3]=bj[2]*beta_j;
      bj[4]=bj[3]*beta_j;
      bj[5]=bj[4]*beta_j;
      // precalculate powers of R=|rj-ri|
      r[0]=r_ij;
      r[1]=r[0]*r_ij;
      r[2]=r[1]*r_ij;
      r[3]=r[2]*r_ij;
      r[4]=r[3]*r_ij;
      r[5]=r[4]*r_ij;
      r[6]=r[5]*r_ij;
      r[7]=r[6]*r_ij;
      olap=945*(6*bi[1]*bj[1] + 4*bi[0]*bj[2] + bj[3]) + 280*(5*bi[4] - 4*bi[2]*bj[1] - 4*bi[1]*bj[2] + 7*bi[0]*bj[3] + 5*bj[4])*r[1] + 240*bi[5]*r[3] + 20*bj[0]*(189*bi[2] + 98*bi[3]*r[1] - 40*bi[4]*r[3]) + 16*((216*bi[2]*bj[2] + 43*bi[1]*bj[3] - 50*bi[0]*bj[4] + 15*bj[5])*r[3] + 8*(5*bi[4]*bj[1] - 3*bi[2]*bj[3] + 5*bi[1]*bj[4])*r[5]) + bi[3]*(945 + 688*bj[1]*r[3] - 384*bj[2]*r[5] + 256*bj[3]*r[7]);
      break;
    case 11:
      // li=2 lj=3
      // precalculate powers of beta_i
      bi[0]=beta_i;
      bi[1]=bi[0]*beta_i;
      bi[2]=bi[1]*beta_i;
      bi[3]=bi[2]*beta_i;
      bi[4]=bi[3]*beta_i;
      bi[5]=bi[4]*beta_i;
      bi[6]=bi[5]*beta_i;
      bi[7]=bi[6]*beta_i;
      // precalculate powers of beta_j
      bj[0]=beta_j;
      bj[1]=bj[0]*beta_j;
      bj[2]=bj[1]*beta_j;
      bj[3]=bj[2]*beta_j;
      bj[4]=bj[3]*beta_j;
      bj[5]=bj[4]*beta_j;
      bj[6]=bj[5]*beta_j;
      // precalculate powers of R=|rj-ri|
      r[0]=r_ij;
      r[1]=r[0]*r_ij;
      r[2]=r[1]*r_ij;
      r[3]=r[2]*r_ij;
      r[4]=r[3]*r_ij;
      r[5]=r[4]*r_ij;
      r[6]=r[5]*r_ij;
      r[7]=r[6]*r_ij;
      r[8]=r[7]*r_ij;
      r[9]=r[8]*r_ij;
      olap=10395*(bi[4] + 10*bi[2]*bj[1] + 10*bi[1]*bj[2] + 5*bi[0]*bj[3] + bj[4]) + 3*bj[0]*(17325*bi[3] + 25200*bi[4]*r[1] - 784*bi[5]*r[3] - 1920*bi[6]*r[5]) + 4*(315*(40*bi[3]*bj[1] - 20*bi[2]*bj[2] - 15*bi[1]*bj[3] + 16*bi[0]*bj[4] + 10*bj[5])*r[1] + 84*(35*bi[6] - 93*bi[4]*bj[1] + 73*bi[3]*bj[2] + 187*bi[2]*bj[3] + 33*bi[1]*bj[4] - 25*bi[0]*bj[5] + 5*bj[6])*r[3] + 48*(5*bi[7] + 124*bi[4]*bj[2] - 51*bi[3]*bj[3] - 42*bi[2]*bj[4] + 35*bi[1]*bj[5])*r[5] + 64*(10*bi[6]*bj[1] - 3*bi[4]*bj[3] + 21*bi[3]*bj[4])*r[7] + bi[5]*(6615*r[1] + 3024*bj[1]*r[5] - 896*bj[2]*r[7] + 256*bj[3]*r[9]));
      break;
    case 15:
      // li=3 lj=3
      // precalculate powers of beta_i
      bi[0]=beta_i;
      bi[1]=bi[0]*beta_i;
      bi[2]=bi[1]*beta_i;
      bi[3]=bi[2]*beta_i;
      bi[4]=bi[3]*beta_i;
      bi[5]=bi[4]*beta_i;
      bi[6]=bi[5]*beta_i;
      bi[7]=bi[6]*beta_i;
      bi[8]=bi[7]*beta_i;
      // precalculate powers of beta_j
      bj[0]=beta_j;
      bj[1]=bj[0]*beta_j;
      bj[2]=bj[1]*beta_j;
      bj[3]=bj[2]*beta_j;
      bj[4]=bj[3]*beta_j;
      bj[5]=bj[4]*beta_j;
      bj[6]=bj[5]*beta_j;
      bj[7]=bj[6]*beta_j;
      bj[8]=bj[7]*beta_j;
      // precalculate powers of R=|rj-ri|
      r[0]=r_ij;
      r[1]=r[0]*r_ij;
      r[2]=r[1]*r_ij;
      r[3]=r[2]*r_ij;
      r[4]=r[3]*r_ij;
      r[5]=r[4]*r_ij;
      r[6]=r[5]*r_ij;
      r[7]=r[6]*r_ij;
      r[8]=r[7]*r_ij;
      r[9]=r[8]*r_ij;
      r[10]=r[9]*r_ij;
      r[11]=r[10]*r_ij;
      olap=945*(143*(15*bi[3]*bj[1] + 20*bi[2]*bj[2] + 15*bi[1]*bj[3] + 6*bi[0]*bj[4] + bj[5]) + 44*(7*bi[6] + 17*bi[4]*bj[1] - 15*bi[3]*bj[2] - 15*bi[2]*bj[3] + 17*bi[1]*bj[4] + 23*bi[0]*bj[5] + 7*bj[6])*r[1] + 112*bi[7]*r[3]) + 6720*bi[8]*r[5] + 378*bj[0]*(2145*bi[4] + 2530*bi[5]*r[1] - 32*(7*bi[6]*r[3] + 5*bi[7]*r[5])) + 3*(bi[5]*(45045 - 154224*bj[1]*r[3] + 122880*bj[2]*r[5] + 2048*bj[3]*r[7] - 5120*bj[4]*r[9]) + 16*(12600*bi[2]*bj[4]*r[3] - 9639*bi[1]*bj[5]*r[3] - 1764*bi[0]*bj[6]*r[3] + 2205*bj[7]*r[3] + 7680*bi[2]*bj[5]*r[5] + 4032*bi[1]*bj[6]*r[5] - 1260*bi[0]*bj[7]*r[5] + 140*bj[8]*r[5] + 112*(-14*bi[2]*bj[6] + 5*bi[1]*bj[7])*r[7] + 8*bi[4]*(1575*bj[2]*r[3] - 900*bj[3]*r[5] + 564*bj[4]*r[7] - 40*bj[5]*r[9]) + 112*(5*bi[7]*bj[1]*r[7] + 2*bi[6]*(18*bj[1]*r[5] - 7*bj[2]*r[7] + 2*bj[3]*r[9])) + 4*bi[3]*(9135*bj[3]*r[3] + 8*(-225*bj[4]*r[5] + 4*bj[5]*r[7] + 14*bj[6]*r[9])))) + 4096*bi[5]*bj[5]*r[11];
      break;
    default:
      printf("ERROR: radial integrals not implemented for li=%d  lj=%d\n",li,lj);
      printf("       If li > lj, swap the Gaussians i and j before calling this functions.\n");
      assert((li <= lmax) && (lj <= lmax));
      break;
  }
  return prefactor*olap;
# pragma GCC diagnostic pop
}

// template specializations
template double radial_overlap<double>(double xi, double yi, double zi, int li, double beta_i,
                                       double xj, double yj, double zj, int lj, double beta_j);
template float radial_overlap<float>(float xi, float yi, float zi, int li, float beta_i,
                                     float xj, float yj, float zj, int lj, float beta_j);
