#include <stdio.h>
#include <stdlib.h>

#include <ctime>
#include <sys/time.h>
#include <random>
#include <assert.h>

#include <cuda_profiler_api.h>

#include "polarization.h"

#ifdef SINGLE_PRECISION
typedef float real;
#else
typedef double real;
#endif

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

int main() {
  const int k = 3;
  const int mx = 1;
  const int my = 0;
  const int mz = 0;
  real alpha = 4.0;
  const int q = 2;

  int npair = 16000000;

  PrimitivePair<real> *pairs;
  // allocate memory on host for pairs of primitives
# ifdef PINNED_MEMORY
  // allocate pinned memory on host for pairs of primitives
  long int mem_bytes = sizeof(PrimitivePair<real>) * npair;
  printf("Allocate pinned host memory of %ld Kb = %ld Mb = %ld Gb \n",  
	 (mem_bytes >> 10), 
	 (mem_bytes >> 20),
	 (mem_bytes >> 30));
  checkCuda( cudaMallocHost((void**) &pairs, sizeof(PrimitivePair<real>) * npair) ); 
# else
  pairs = (PrimitivePair<real> *) malloc(sizeof(PrimitivePair<real>) * npair);
# endif

  // random numbers
  std::random_device rd;    //Will be used to obtain a seed for the random number engine
  std::mt19937 random(rd());  //Standard mersenne_twister_engine seeded with rd()
  std::uniform_real_distribution<> distribution(0.0, 2.0);

  printf("Generate random primitive pairs\n");
  // determine size of output buffer to accomodate all combinations of angular momenta
  int buffer_size = 0;
  // create random pairs of primitives
  for(int ipair = 0; ipair < npair; ipair++) {
    PrimitivePair<real> *pair = pairs+ipair;
    Primitive<real> *primA = &(pair->primA);
    Primitive<real> *primB = &(pair->primB);

    if (ipair == 0) {
      // For these parameters we know the exact integrals, just a quick check
      primA->coef = 1.0;
      primA->exp  = 1.0;
      primA->l    = 0;
      primA->x    = 0.0;
      primA->y    = 0.0;
      primA->z    = 0.0;
      
      primB->coef = 1.0;
      primB->exp  = 1.0;
      primB->l    = 0;
      primB->x    = 0.0;
      primB->y    = 0.0;
      primB->z    = 0.0;
    } else if (ipair == 1) {
      // Another check with parameters for which we know the right result
      primA->coef = -0.0507585;
      primA->exp = 0.63629;
      primA->l = 0;
      primA->x = -11.3384;
      primA->y = 0.0;
      primA->z = 0.0;

      primB->coef = -0.288858;
      primB->exp = 6.4648;
      primB->l = 0;        
      primB->x = -7.5589;
      primB->y = 0.0; 
      primB->z = 0.0;
    } else {
      // random parameters
      primA->coef = distribution(random);
      primA->exp  = distribution(random) + 0.01;
      primA->l    = 1;   // p-orbital
      //primA->l    = 0;     // s-orbital
      primA->x    = distribution(random);
      primA->y    = distribution(random);
      primA->z    = distribution(random);
      
      primB->coef = distribution(random);
      primB->exp  = distribution(random) + 0.01;
      primB->l    = 2;  // d-orbital
      //primB->l    = 0;    // s-orbital
      primB->x    = distribution(random);
      primB->y    = distribution(random);
      primB->z    = distribution(random);
    }
    // Index into buffer where the integrals for this pair start.
    pair->bufferIdx = buffer_size;
    // Increase buffer by the amount needed for this pair.
    buffer_size += ANGL_FUNCS(primA->l) * ANGL_FUNCS(primB->l);
  }

  printf("number of integrals : %d  %d\n", buffer_size, integral_buffer_size<real>(pairs, npair));

  real *buffer;
  // allocate host memory for integral
# ifdef PINNED_MEMORY
  // allocate pinned host memory for integrals
  mem_bytes = sizeof(real) * buffer_size;
  printf("Allocate pinned host memory of %ld Kb = %ld Mb = %ld Gb \n",  
	 (mem_bytes >> 10), 
	 (mem_bytes >> 20),
	 (mem_bytes >> 30));
  checkCuda( cudaMallocHost((void**)&buffer, sizeof(real) * buffer_size) ); 
# else
  buffer = (real *) malloc(sizeof(real) * buffer_size);
# endif

  cudaProfilerStart();
  // compute integrals
  polarization_prim_pairs<real, k, mx, my, mz, q>(pairs, npair, 
						  buffer, 
						  alpha);
  cudaProfilerStop();
  
  printf("buffer[0]= %e\n", buffer[0]);
  printf("buffer[1]= %e\n", buffer[1]);

  /*
  for(int i = 0; i < buffer_size; i++) {
    printf("buffer[%d] = %f\n", i, buffer[i]);
  }
  */

# ifdef PINNED_MEMORY
  // release dynamic pinned memory
  cudaFreeHost(pairs);
  cudaFreeHost(buffer);
# else
  // release dynamic memory
  free(pairs);
  free(buffer);
# endif

  printf("DONE\n");
}
