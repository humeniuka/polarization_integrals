#include <stdio.h>
#include <stdlib.h>

#include <ctime>
#include <sys/time.h>
#include <random>

#include <cuda_profiler_api.h>

#include "polarization.h"

int main() {
  double3 origin = {0.0, 0.0, 0.0};
  int k = 3;
  int mx = 0;
  int my = 0;
  int mz = 0;
  double alpha = 50.0;
  int q = 2;

  int npair = 100000;

  // allocate memory for pairs of primitives
  PrimitivePair *pairs, *pairs_;
  pairs = (PrimitivePair *) malloc(sizeof(PrimitivePair) * npair);
  cudaMalloc((void **) &pairs_, sizeof(PrimitivePair) * npair);

  // random numbers
  std::random_device rd;    //Will be used to obtain a seed for the random number engine
  std::mt19937 random(rd());  //Standard mersenne_twister_engine seeded with rd()
  std::uniform_real_distribution<> distribution(0.0, 2.0);

  // determine size of output buffer to accomodate all combinations of angular momenta
  int buffer_size = 0;
  // create random pairs of primitives
  for(int ipair = 0; ipair < npair; ipair++) {
    PrimitivePair *pair = pairs+ipair;
    Primitive *primA = &(pair->primA);
    Primitive *primB = &(pair->primB);

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
    } else {
      // random parameters
      primA->coef = distribution(random);
      primA->exp  = distribution(random) + 0.01;
      primA->l    = 1;   // p-orbital
      primA->x    = distribution(random);
      primA->y    = distribution(random);
      primA->z    = distribution(random);
      
      primB->coef = distribution(random);
      primB->exp  = distribution(random) + 0.01;
      primB->l    = 2;  // d-orbital
      primB->x    = distribution(random);
      primB->y    = distribution(random);
      primB->z    = distribution(random);
    }
    // Index into buffer where the integrals for this pair start.
    pair->bufferIdx = buffer_size;
    // Increase buffer by the amount needed for this pair.
    buffer_size += ANGL_FUNCS(primA->l) * ANGL_FUNCS(primB->l);
  }

  printf("number of integrals : %d  %d\n", buffer_size, integral_buffer_size(pairs, npair));
  // copy primitive data to device
  cudaMemcpy(pairs_, pairs, sizeof(PrimitivePair) * npair,  cudaMemcpyHostToDevice);

  // allocate memory for integrals
  double *buffer, *buffer_;
  buffer = (double *) malloc(sizeof(double) * buffer_size);
  cudaMalloc((void **) &buffer_, sizeof(double) * buffer_size);

  cudaProfilerStart();
  // compute integrals
  polarization_prim_pairs(pairs_, npair, 
			  buffer_, 
			  origin, 
			  k, mx, my, mz,
			  alpha, q);
  cudaProfilerStop();
  
  // copy integrals back
  cudaMemcpy(buffer, buffer_, sizeof(double) * buffer_size,  cudaMemcpyDeviceToHost);
  printf("buffer[0]= %f\n", buffer[0]);

  /*
  for(int i = 0; i < buffer_size; i++) {
    printf("buffer[%d] = %f\n", i, buffer[i]);
  }
  */

  // release dynamic memory
  free(pairs);
  cudaFree(pairs_);
  free(buffer);
  cudaFree(buffer_);

  printf("DONE\n");
}
