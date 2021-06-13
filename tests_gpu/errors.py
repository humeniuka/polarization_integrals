#!/usr/bin/env python
"""
separate errors of GPU vs CPU integrals for different combinations of angular momenta
and print a table
"""

def ao_ordering(l):
    """list of Cartesian basis function with angular momentum l"""
    if l == 0:
        return [(0,0,0)]
    elif l == 1:
        return [(1,0,0), (0,1,0), (0,0,1)]
    elif l == 2:
        # ordering of d-functions in TeraChem: dxy,dxz,dyz,dxx,dyy,dzz
        return [(1,1,0), (1,0,1), (0,1,1), (2,0,0), (0,2,0), (0,0,2)]
    else:
        # The integral routines work for any angular momentum, but what ordering should
        # be used for f,g,etc. functions?
        raise NotImplemented("Angular momenta higher than s,p and d functions are not implemented!")


def group_errors_by_angmom(pairs, integs, integs_ref):
    """
    compare integrals with reference

    Parameters
    ----------
    pairs       :   list of PrimitivePair
    integs      :   np.ndarray
      integrals for all primitive pairs
    integs_ref  :   np.ndarray
      reference integrals in the same order as in `integs`

    Returns
    -------
    max_errors  :   dict
      dictionary with maximum errors for angular momentum combinations (lA,lB)
    """
    max_errors = {}
    for pair in pairs:
        primA = pair.primA
        primB = pair.primB
        # enumerate integrals for this pair of primitives
        ij = 0
        for i,(nxi,nyi,nzi) in enumerate(ao_ordering(primA.l)):
            for j,(nxj,nyj,nzj) in enumerate(ao_ordering(primB.l)):
                idx = pair.bufferIdx + ij
                error = abs(integs[idx] - integs_ref[idx])
                key = (primA.l, primB.l)
                max_errors[key] = max(max_errors.get(key, 0.0), error)

                ij += 1
                
    # convert angular momentum to spectroscopy notation s,p,d,...
    l2s = {0: "s", 1: "p", 2: "d", 3: "f", 4: "g", 5: "h"}
    # print table
    print("")
    print("""angmom A      angmom B         maximum error""")
    for (lA,lB), max_err in max_errors.items():
        print(f"   {l2s[lA]}             {l2s[lB]}              {max_err:e}")
    print("")
    
    return max_errors
