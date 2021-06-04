#!/usr/bin/env python
import time
import numpy as np
import numpy.linalg as la

# try to import GPU implementation of integrals
try:
    from polarization_integrals._polarization_gpu import Primitive, PrimitivePair, polarization_prim_pairs
except ImportError as e:
    print("NOTE: The module _polarization_gpu requires a GPU!")
    raise e

# import CPU implementation
from polarization_integrals import PolarizationIntegral

def random_primitive(l):
    """Gaussian primitive of angular momentum l with random parameters"""
    exp, = 0.01 + 2.0*np.random.rand(1)
    coef, x, y, z, = 2.0*(np.random.rand(4)-0.5)
    # shellIdx not needed
    shellIdx = 0
    return Primitive(exp, coef, l, x, y, z, shellIdx)

# generate list of random primitive pairs
n = 10000000

# up to d-functions
lmax = 2

pairs = []
buffer_size = 0
# primitives should be sorted by angular momenta to avoid warp divergence
for lA in range(0, lmax):
    num_angmomA = ((lA+1)*(lA+2))//2
    for lB in range(0, lmax):
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

# compute polarization integral for all primitives
origin = [0.0, 0.0, 0.0]
# polarization operator Op = x/r^3
k = 3
mx, my, mz = 1, 0, 0
# cutoff function
alpha = 50.0
q = 2

##### compute integrals on the GPU ########

tstart = time.time()
integrals_gpu = polarization_prim_pairs(pairs, origin, k, mx,my,mz, alpha, q)
tend = time.time()
print("GPU integrals took %f seconds" % (tend-tstart))

##### compute integrals on the CPU ########
integrals_cpu = np.zeros_like(integrals_gpu)

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

tstart = time.time()
for pair in pairs:
    primA = pair.primA
    primB = pair.primB
    # shift origin to center of polarizable atom
    xA, yA, zA = primA.x - origin[0], primA.y - origin[1], primA.z - origin[2]
    xB, yB, zB = primB.x - origin[0], primB.y - origin[1], primB.z - origin[2]
    pol = PolarizationIntegral(xA,yA,zA, primA.l, primA.exp,
                               xB,yB,zB, primB.l, primB.exp,
                               k, mx,my,mz,
                               alpha, q)
    # enumerate integrals for this pair of primitives
    ij = 0
    # contraction coefficients
    cc = primA.coef * primB.coef
    for i,(nxi,nyi,nzi) in enumerate(ao_ordering(primA.l)):
        for j,(nxj,nyj,nzj) in enumerate(ao_ordering(primB.l)):
            integrals_cpu[pair.bufferIdx + ij] = cc * pol.compute_pair(nxi,nyi,nzi,
                                                                       nxj,nyj,nzj)
            ij += 1
tend = time.time()

print("CPU integrals took %f seconds" % (tend-tstart))
            
###### compare GPU and CPU integrals ######
print("GPU integrals")
print(integrals_gpu[:20])
print("CPU integrals")
print(integrals_cpu[:20])

err = la.norm(integrals_gpu - integrals_cpu)

print(f"|Integrals(GPU)-Integrals(CPU)| = {err}")
                                                                    