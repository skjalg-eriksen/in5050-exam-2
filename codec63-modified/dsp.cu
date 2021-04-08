#include <inttypes.h>
#include <math.h>
#include <stdlib.h>

#include "tables.cuh"
#include "dsp.cuh"
#include <stdio.h>

/*
  Moved all these functions over to device so they can run on the GPU
*/

__device__ static void transpose_block(float *in_data, float *out_data)
{
  int i, j;
  #pragma unroll
  for (i = 0; i < 8; ++i)
  {
    #pragma unroll
    for (j = 0; j < 8; ++j)
    {
      out_data[i*8+j] = in_data[j*8+i];
    }
  }

}

__device__ static void dct_1d(float *in_data, float *out_data)
{
  int i, j;
  #pragma unroll
  for (i = 0; i < 8; ++i)
  {
    float dct = 0;
    #pragma unroll
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
  #pragma unroll
  for (i = 0; i < 8; ++i)
  {
    float idct = 0;
    #pragma unroll
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
  #pragma unroll
  for (v = 0; v < 8; ++v)
  {
    #pragma unroll
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
  #pragma unroll
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
  #pragma unroll
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
  #pragma unroll
  for (i = 0; i < 64; ++i) { mb2[i] = in_data[i]; }
  /* Two 1D DCT operations with transpose */
  #pragma unroll
  for (v = 0; v < 8; ++v) {
    dct_1d(mb2+v*8, mb+v*8);
  }
  transpose_block(mb, mb2);
  #pragma unroll
  for (v = 0; v < 8; ++v) {
    dct_1d(mb2+v*8, mb+v*8);
  }
  transpose_block(mb, mb2);
  scale_block(mb2, mb);
  quantize_block(mb, mb2, quant_tbl);

  #pragma unroll
  for (i = 0; i < 64; ++i) { out_data[i] = mb2[i]; }

}



__device__ void dequant_idct_block_8x8(int16_t *in_data, int16_t *out_data,
    uint8_t *quant_tbl)
{
  float mb[8*8] __attribute((aligned(16)));
  float mb2[8*8] __attribute((aligned(16)));

  int i, v;

  #pragma unroll
  for (i = 0; i < 64; ++i) { mb[i] = in_data[i]; }
  dequantize_block(mb, mb2, quant_tbl);
  scale_block(mb2, mb);

  /* Two 1D inverse DCT operations with transpose */
  #pragma unroll
  for (v = 0; v < 8; ++v) { idct_1d(mb+v*8, mb2+v*8); }
  transpose_block(mb2, mb);

  #pragma unroll
  for (v = 0; v < 8; ++v) { idct_1d(mb+v*8, mb2+v*8); }
  transpose_block(mb2, mb);

  #pragma unroll
  for (i = 0; i < 64; ++i) { out_data[i] = mb[i]; }
}
