
#include "flash_attention.cuh"
#include <cstdio>
#include <cstdlib>

int main() {
    const int N = 1024;
    size_t bytes = N * D * sizeof(float);

    float *h_Q = (float*)malloc(bytes);
    float *h_K = (float*)malloc(bytes);
    float *h_V = (float*)malloc(bytes);
    float *h_O = (float*)malloc(bytes);
    for (int i = 0; i < N * D; i++) {
        h_Q[i] = (float)rand() / RAND_MAX - 0.5f;
        h_K[i] = (float)rand() / RAND_MAX - 0.5f;
        h_V[i] = (float)rand() / RAND_MAX - 0.5f;
    }

    size_t free_before, free_after, total;
    cudaMemGetInfo(&free_before, &total);

    float *d_Q, *d_K, *d_V, *d_O;
    cudaMalloc(&d_Q, bytes);
    cudaMalloc(&d_K, bytes);
    cudaMalloc(&d_V, bytes);
    cudaMalloc(&d_O, bytes);

    cudaMemGetInfo(&free_after, &total);
    printf("kernel allocation: %.2f MB\n", (free_before - free_after) / 1e6);

    cudaMemcpy(d_Q, h_Q, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, bytes, cudaMemcpyHostToDevice);

    dim3 threads(BR);
    dim3 blocks((N + BR - 1) / BR);
    flash_attention<<<blocks, threads>>>(d_Q, d_K, d_V, d_O, N);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) { 
      printf("Kernel error: %s\n", cudaGetErrorString(err)); 
      return 1; 
    }
    cudaDeviceSynchronize();
    cudaMemcpy(h_O, d_O, bytes, cudaMemcpyDeviceToHost);

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    int iters = 100;

    flash_attention<<<blocks, threads>>>(d_Q, d_K, d_V, d_O, N);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < iters; i++) {
      flash_attention<<<blocks, threads>>>(d_Q, d_K, d_V, d_O, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    printf("flash kernel avg: %.4f ms\n", ms / iters);

    cudaEventDestroy(start); cudaEventDestroy(stop);
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O);
    free(h_Q); free(h_K); free(h_V); free(h_O);
    return 0;
}
