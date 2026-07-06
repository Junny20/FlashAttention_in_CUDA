
#ifndef NAIVE_MATMUL_CUH
#define NAIVE_MATMUL_CUH

__global__ void naive_matmul(float *a, float *b, float *c, int m, int k, int n);

#endif
