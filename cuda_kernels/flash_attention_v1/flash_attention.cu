
#include "flash_attention.cuh"
#include <cmath>

__global__ void flash_attention(float *Q, float *K, float *V, float *O, int N) {
  // each thread owns 1 query row
  __shared__ float Qi[BR][D];
  __shared__ float Kj[BC][D];
  __shared__ float Vj[BC][D];

  int tx = threadIdx.x;
  int query_block = blockIdx.x;
  int query_row = query_block * BR + tx;

  if (query_row < N) {
    for (int i = 0; i < D; i++) {
      Qi[tx][i] = Q[query_row * D + i];
    }
  }
  __syncthreads(); 
  
  float m_i = -INFINITY;
  float l_i = 0.0f;
  float acc[D];
  for (int i = 0; i < D; i++) acc[i] = 0.0f;

  // Qi fully loaded
  int R = (N + BC - 1) / BC;
  for (int i = 0; i < R; i++) {
    int kv_row = i * BC + tx; // invariant: blockDim.x >= BC
    if (kv_row < N) {
      for (int j = 0; j < D; j++) {
        Kj[tx][j] = K[kv_row * D + j];
        Vj[tx][j] = V[kv_row * D + j]; 
      }
    }
    __syncthreads();

    // Kj, Vj fully loaded, calculate S
    float s[BC]; // (1 x D) @ (D x BC)
    for (int j = 0; j < BC; j++) {
      float dot = 0.0f;
      for (int k = 0; k < D; k++) {
        dot += Qi[tx][k] * Kj[j][k];
      }
      s[j] = dot * rsqrtf(D);
    }

    // running softmax
    float m_tile = -INFINITY;
    float l_tile = 0.0f;
    for (int j = 0; j < BC; j++) {
      m_tile = fmaxf(m_tile, s[j]);
    }
    float new_max = fmaxf(m_i, m_tile);
    float rescale = expf(m_i - new_max);
    for (int j = 0; j < BC; j++) {
      s[j] = expf(s[j] - new_max);
      l_tile += s[j];
    }
    l_i = l_i * rescale + l_tile;

    // compute output row - S @ V
    for (int j = 0; j < D; j++) {
      float dot = 0.0f;
      for (int k = 0; k < BC; k++) {
        dot += s[k] * Vj[k][j];
      }
      acc[j] = acc[j] * rescale + dot;
    }

    m_i = new_max;
    __syncthreads();
  }

  // write output row to HBM
  for (int i = 0; i < D; i++) {
    if (query_row < N) {
      O[query_row * D + i] = acc[i] / l_i;
    }
  }
}
