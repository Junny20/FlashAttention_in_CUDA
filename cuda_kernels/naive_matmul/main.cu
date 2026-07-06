
#include "naive_matmul.cuh"
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define M 512
#define K 256
#define N 512

void init_matrix(float *matrix, int m, int n) {
  for (int i = 0; i < m * n; i++) {
    matrix[i] = (float)rand() / RAND_MAX;
  }
}

void cpu_naive_matmul(float *a, float *b, float *c, int m, int k, int n) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      float dot = 0.0f;
      for (int l = 0; l < k; l++) {
        dot += a[i * k + l] * b[l * n + j];
      }
      c[i * n + j] = dot;
    }
  }
}

int main() {
  float *h_a, *h_b, *h_c, *d_a, *d_b, *d_c;

  size_t a_sz = sizeof(float) * M * K;
  size_t b_sz = sizeof(float) * K * N;
  size_t c_sz = sizeof(float) * M * N;

  h_a = (float*)malloc(a_sz);
  h_b = (float*)malloc(b_sz);
  h_c = (float*)malloc(c_sz);
  cudaMalloc((void**)&d_a, a_sz);
  cudaMalloc((void**)&d_b, b_sz);
  cudaMalloc((void**)&d_c, c_sz);

  init_matrix(h_a, M, K);
  init_matrix(h_b, K, N);

  cudaStream_t stream1, stream2;
  cudaStreamCreate(&stream1);
  cudaStreamCreate(&stream2);

  cudaMemcpyAsync(d_a, h_a, a_sz, cudaMemcpyHostToDevice, stream1);
  cudaMemcpyAsync(d_b, h_b, b_sz, cudaMemcpyHostToDevice, stream2);
  cudaDeviceSynchronize();

  dim3 threads(16, 16, 1);
  dim3 blocks((N + 15) / 16, (M + 15) / 16, 1); // x is col, y is row

  naive_matmul<<<blocks, threads>>>(d_a, d_b, d_c, M, K, N);
  
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Kernel error: %s", cudaGetErrorString(err));
    return 1;
  }
  cudaDeviceSynchronize();

  // correctness oracle

  cudaMemcpy(h_c, d_c, c_sz, cudaMemcpyDeviceToHost);
  float *h_c_cpy = (float*)malloc(c_sz);
  cpu_naive_matmul(h_a, h_b, h_c_cpy, M, K, N);

  bool pass = 1;
  float max_diff = -INFINITY;
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      float diff = fabsf(h_c_cpy[i * N + j] - h_c[i * N + j]);
      max_diff = fmaxf(max_diff, diff);
      if (diff > 1e-4) {
        pass = 0;
        break;
      }
    }

    if (!pass) break;
  }

  if (pass) {
    printf("Passed correctness test. Max diff: %f\n", max_diff);
  } else {
    printf("Failed correctness oracle. Max diff: %f\n", max_diff);
  }

  free(h_c_cpy);

  cudaEvent_t start, stop;
  cudaEventCreate(&start), cudaEventCreate(&stop);
  int iters = 16384;

  cudaEventRecord(start);
  for (int i = 0; i < iters; i++) {
    naive_matmul<<<blocks, threads>>>(d_a, d_b, d_c, M, K, N);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float ms;
  cudaEventElapsedTime(&ms, start, stop);
  printf("naive matmul avg: %.4f ms\n", ms / iters);

  cudaEventDestroy(start), cudaEventDestroy(stop);
  cudaStreamDestroy(stream1), cudaStreamDestroy(stream2);
  free(h_a), free(h_b), free(h_c);
  cudaFree(d_a), cudaFree(d_b), cudaFree(d_c);
}
