
#ifndef FLASH_ATTENTION_CUH
#define FLASH_ATTENTION_CUH

#define BR 16
#define BC 16
#define D 16

__global__ void flash_attention(float *Q, float *K, float *V, float *O, int N);

#endif
