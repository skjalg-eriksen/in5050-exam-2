#include <inttypes.h>
#include <math.h>
#include <stdlib.h>

//extern "C" {}

#include "tables.cuh"

#include "dsp.cuh"
#include <cuda_profiler_api.h>
#include <stdio.h>

__device__ static void transpose_block(float *in_data, float *out_data)
{
  int i, j;

  for (i = 0; i < 8; ++i)
  {
    for (j = 0; j < 8; ++j)
    {
      out_data[i*8+j] = in_data[j*8+i];
    }
  }
}

__device__ static void dct_1d(float *in_data, float *out_data)
{
  int i, j;

  for (i = 0; i < 8; ++i)
  {
    float dct = 0;

    for (j = 0; j < 8; ++j)
    {
      dct += in_data[j] * dctlookup[j][i];
    }

    out_data[i] = dct;
  }
}

__device__ static void idct_1d(float *in_data, float *out_data)
{
  int i, j;

  for (i = 0; i < 8; ++i)
  {
    float idct = 0;

    for (j = 0; j < 8; ++j)
    {
      idct += in_data[j] * dctlookup[i][j];
    }

    out_data[i] = idct;
  }
}

__device__ static void scale_block(float *in_data, float *out_data)
{
  int u, v;

  for (v = 0; v < 8; ++v)
  {
    for (u = 0; u < 8; ++u)
    {
      float a1 = !u ? ISQRT2 : 1.0f;
      float a2 = !v ? ISQRT2 : 1.0f;

      /* Scale according to normalizing function */
      out_data[v*8+u] = in_data[v*8+u] * a1 * a2;
    }
  }
}

__device__ static void quantize_block(float *in_data, float *out_data, uint8_t *quant_tbl)
{
  int zigzag;

  for (zigzag = 0; zigzag < 64; ++zigzag)
  {
    uint8_t u = zigzag_U[zigzag];
    uint8_t v = zigzag_V[zigzag];

    float dct = in_data[v*8+u];

    /* Zig-zag and quantize */
    out_data[zigzag] = (float) round((dct / 4.0) / quant_tbl[zigzag]);
  }
}

__device__ static void dequantize_block(float *in_data, float *out_data,
    uint8_t *quant_tbl)
{
  int zigzag;

  for (zigzag = 0; zigzag < 64; ++zigzag)
  {
    uint8_t u = zigzag_U[zigzag];
    uint8_t v = zigzag_V[zigzag];

    float dct = in_data[zigzag];

    /* Zig-zag and de-quantize */
    out_data[v*8+u] = (float) round((dct * quant_tbl[zigzag]) / 4.0);
  }
}

__device__ void dct_quant_block_8x8(int16_t *in_data, int16_t *out_data,
    uint8_t *quant_tbl)
{
  float mb[8*8] __attribute((aligned(16)));
  float mb2[8*8] __attribute((aligned(16)));

  int i, v;

  for (i = 0; i < 64; ++i) { mb2[i] = in_data[i]; }

  /* Two 1D DCT operations with transpose */
  for (v = 0; v < 8; ++v) {
    dct_1d(mb2+v*8, mb+v*8);
  }
  transpose_block(mb, mb2);

  for (v = 0; v < 8; ++v) {
    dct_1d(mb2+v*8, mb+v*8);
  }
  transpose_block(mb, mb2);

  scale_block(mb2, mb);
  quantize_block(mb, mb2, quant_tbl);

  for (i = 0; i < 64; ++i) { out_data[i] = mb2[i]; }
}



__device__ void dequant_idct_block_8x8(int16_t *in_data, int16_t *out_data,
    uint8_t *quant_tbl)
{
  float mb[8*8] __attribute((aligned(16)));
  float mb2[8*8] __attribute((aligned(16)));

  int i, v;

  for (i = 0; i < 64; ++i) { mb[i] = in_data[i]; }

  dequantize_block(mb, mb2, quant_tbl);
  scale_block(mb2, mb);

  /* Two 1D inverse DCT operations with transpose */
  for (v = 0; v < 8; ++v) { idct_1d(mb+v*8, mb2+v*8); }
  transpose_block(mb2, mb);
  for (v = 0; v < 8; ++v) { idct_1d(mb+v*8, mb2+v*8); }
  transpose_block(mb2, mb);

  for (i = 0; i < 64; ++i) { out_data[i] = mb[i]; }
}



__global__ void sad_block(uint8_t *block1, uint8_t *block2, int stride, int *result){
  //int x = blockIdx.x * blockDim.x + threadIdx.x;
  //int y = blockIdx.y * blockDim.y + threadIdx.y;
  int x = threadIdx.x;
  int y = threadIdx.y;
  //printf("%s%d%s%d\n", "x:", x, ",\ty:", y);
  //printf("%s%d%s%d%s%d\n", "INDEX:", i%8, ", ", i/8, "\t ", abs(block2[(i%8)*stride+(i/8)] - block1[(i%8)*stride+(i/8)]));
  //result[i] = __vsadu4(block2[(i%8)*stride+(i/8)], block1[(i%8)*stride+(i/8)]);

  if(x < 8 && y < 8){
    //result[i] = abs(block2[(i%8)*stride+(i/8)] - block1[(i%8)*stride+(i/8)]);
    result[x*8+y] = abs(block2[x*stride+y] - block1[x*stride+y]);
  }
//  printf("%d\n", );
  //*result += abs(block2[(i%8)*stride+(i/8)] - block1[(i%8)*stride+(i/8)]);
  //__syncthreads();
}




void sad_block_8x8(uint8_t *block1, uint8_t *block2, int stride, int *result)
{
  int u, v;

  //printf("\n");
  int r[64];//=0;
  int *cuda_r;

  *result = 0;

  cudaMallocManaged((void**)&cuda_r, (sizeof(int)*64));

  dim3 dimGrid(1, 1);
  dim3 dimBlock(8, 8);

  sad_block<<<dimGrid,dimBlock>>>(block1, block2, stride, cuda_r);
  cudaDeviceSynchronize();

  //cudaMemcpy(&r, cuda_r, (sizeof(int)*64), cudaMemcpyDeviceToHost);

  for(int i=0; i < 64;i++){

    //  printf("%d\n", a[i]);
    //*result += r[i];
    *result += cuda_r[i];
  }
  //printf("\n%s%d\n", "CUDA-result:\t", *result);
  cudaFree(cuda_r);

/*

  *result = 0;
  for (v = 0; v < 8; ++v)
  {
    for (u = 0; u < 8; ++u)
    {
      *result += abs(block2[v*stride+u] - block1[v*stride+u]);
      //printf("%s%d%s%d%s%d\n", "sadidx:", u, ", ", v, "\t ", abs(block2[v*stride+u] - block1[v*stride+u]));
    }
  }*/
  //printf("%s%d\n", "result:\t", *result);
  //printf("\n");
//    exit(1);
}
