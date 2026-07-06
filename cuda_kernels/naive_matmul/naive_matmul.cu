
#include <cmath>

// a.shape = (m, k), b.shape = (k, n), c.shape = (m, n)
__global__ void naive_matmul(float *a, float *b, float *c, int m, int k, int n) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < m && col < n) {
    float dot = 0.0f;
    for (int i = 0; i < k; i++) {
      dot += a[row * k + i] * b[i * n + col];
    }
    c[row * n + col] = dot;
  }
}
