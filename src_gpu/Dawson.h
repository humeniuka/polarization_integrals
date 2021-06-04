#ifndef DAWSON_HH
#define DAWSON_HH

// compute Dawson(z) = sqrt(pi)/2  *  exp(-z^2) * erfi(z)
__host__ __device__ double Dawson(double x); // special case for real x

#endif // DAWSON_HH
