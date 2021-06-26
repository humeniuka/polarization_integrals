
#ifndef _radial_overlap_h
#define _radial_overlap_h

template <typename real>
real radial_overlap(real xi, real yi, real zi, int li, real beta_i,
                    real xj, real yj, real zj, int lj, real beta_j
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
		    );

#endif // end of include guard
