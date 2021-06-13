#!/usr/bin/env python
"""
Polarization integrals are computed on the CPU and GPU (with double precision) and are compared.

To profile the GPU kernel use
 $ nvprof --metrics achieved_occupancy,branch_efficiency -fo log_dp.nvprof python test_integrals_gpu_dp.py
"""
import time
import numpy as np
import numpy.linalg as la

import unittest

# try to import GPU implementation of integrals
try:
    from polarization_integrals._polarization_gpu_dp import Primitive, PrimitivePair, polarization_prim_pairs
except ImportError as e:
    print("NOTE: The module _polarization_gpu requires a GPU! It can be compiled by running `make` inside the folder src_gpu/ .")
    raise e

# import CPU implementation
from polarization_integrals import PolarizationIntegral

from errors import group_errors_by_angmom

def random_primitive(l):
    """Gaussian primitive of angular momentum l with random parameters"""
    # exponent has to be positive
    exp, = 0.001 + 2.0*np.random.rand(1)
    coef, x, y, z, = 2.0*(np.random.rand(4)-0.5)
    # shellIdx not needed
    shellIdx = 0
    return Primitive(coef, exp, l, x, y, z, shellIdx)

def random_pair_list(n=1000):
    """
    generate list of `n` random primitive pairs for testing

    Parameters
    ----------
    n     :   int
      desired number of random pairs
    
    Returns
    -------
    pairs :  list of PrimitivePair
    """
    # make random numbers reproducible
    np.random.seed(0)

    # up to d-functions
    lmax = 2
    
    pairs = []
    buffer_size = 0

    # As an additional check, the first pair of primitives is not random. 
    # Both primitives are s-orbitals with exponent 1.0 centered at origin.
    primA = Primitive(1.0, 1.0, 0, 0.0, 0.0, 0.0,  0)
    primB = Primitive(1.0, 1.0, 0, 0.0, 0.0, 0.0,  0)
    pair = PrimitivePair(primA, primB, 0)
    pairs.append(pair)
    buffer_size += 1

    # primitives should be sorted by angular momenta to avoid warp divergence
    for lA in range(0, lmax+1):
        num_angmomA = ((lA+1)*(lA+2))//2
        for lB in range(0, lmax+1):
            num_angmomB = ((lB+1)*(lB+2))//2
            for ipair in range(0, n):
                primA = random_primitive(lA)
                primB = random_primitive(lB)
                # Each pair of primitives needs to know where to put the integrals in the output buffer.
                # The integrals for this pair start at index `bufferIdx`.
                bufferIdx = buffer_size
                pair = PrimitivePair(primA, primB, bufferIdx)
                pairs.append(pair)
                # Increase size of buffer by the number of integrals generated by this pair.
                buffer_size += num_angmomA * num_angmomB

    print(f"number of integrals = {buffer_size}")

    return pairs

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

class TestGPUIntegrals(unittest.TestCase):
    def test_compare_with_CPU(self):
        """compare GPU with CPU integrals"""
        # random pair of primitives
        pairs = random_pair_list(n=1000)

        # polarization operator Op = x/r^3
        k = 3
        mx, my, mz = 1, 0, 0
        # cutoff function
        alpha = 50.0
        q = 2
        
        ##### compute integrals on the GPU ########

        tstart = time.time()
        integrals_gpu = np.array( polarization_prim_pairs(pairs, k, mx,my,mz, alpha, q) )
        tend = time.time()
        print("GPU integrals took %f seconds" % (tend-tstart))

        ##### compute integrals on the CPU ########
        integrals_cpu = np.zeros_like(integrals_gpu)

        tstart = time.time()
        for pair in pairs:
            primA = pair.primA
            primB = pair.primB
            # The coordinate system is shifted so that the polarizable site lies at the origin.
            pol = PolarizationIntegral(primA.x, primA.y, primA.z, primA.l, primA.exp,
                                       primB.x, primB.y, primB.z, primB.l, primB.exp,
                                       k, mx,my,mz,
                                       alpha, q)
            # enumerate integrals for this pair of primitives
            ij = 0
            # contraction coefficients
            cc = primA.coef * primB.coef
            for i,(nxi,nyi,nzi) in enumerate(ao_ordering(primA.l)):
                for j,(nxj,nyj,nzj) in enumerate(ao_ordering(primB.l)):
                    pint = cc * pol.compute_pair(nxi,nyi,nzi,
                                                 nxj,nyj,nzj)
                    integrals_cpu[pair.bufferIdx + ij] = pint
                    ij += 1
            tend = time.time()

        print("CPU integrals (with double precision) took %f seconds" % (tend-tstart))
            
        ###### compare GPU and CPU integrals ######

        print("GPU (first few integrals)")
        print(integrals_gpu[:5])
        print("CPU (first few integrals)")
        print(integrals_cpu[:5])

        print("GPU (last few integrals)")
        print(integrals_gpu[-5:])
        print("CPU (last few integrals)")
        print(integrals_cpu[-5:])

        print("")
        print("Errors of GPU (double precision) relative to CPU (double precision) implementation")
        group_errors_by_angmom(pairs, integrals_gpu, integrals_cpu)
        
        # absolute error
        max_abs_err = abs(integrals_gpu - integrals_cpu).max()

        print( "absolute error")
        print(f"  max |Integrals(GPU)-Integrals(CPU)|                  = {max_abs_err}")

        self.assertLess(max_abs_err, 1.0e-8)

if __name__ == '__main__':
    unittest.main()

