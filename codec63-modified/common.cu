#include <assert.h>
#include <errno.h>
#include <getopt.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "common.cuh"
#include "dsp.cuh"
#include <cuda_runtime.h>


__device__ void dequantize_idct_row(int16_t *in_data, uint8_t *prediction, int w, int h,
    int y, uint8_t *out_data, uint8_t *quantization)
{
  int x;

  int16_t block[8*8];

  /* Perform the dequantization and iDCT */
  for(x = 0; x < w; x += 8)
  {
    int i, j;

    dequant_idct_block_8x8(in_data+(x*8), block, quantization);

    #pragma unroll
    for (i = 0; i < 8; ++i)
    {
      #pragma unroll
      for (j = 0; j < 8; ++j)
      {
        /* Add prediction block. Note: DCT is not precise -
           Clamp to legal values */
        int16_t tmp = block[i*8+j] + (int16_t)prediction[i*w+j+x];

        if (tmp < 0) { tmp = 0; }
        else if (tmp > 255) { tmp = 255; }
        out_data[i*w+j+x] = tmp;
      }
    }
  }

}

__device__ void dct_quantize_row(uint8_t *in_data, uint8_t *prediction, int w, int h,
    int16_t *out_data, uint8_t *quantization)
{
  int x;

  int16_t block[8*8];

  /* Perform the DCT and quantization */
  for(x = 0; x < w; x += 8)
  {
    int i, j;

    #pragma unroll
    for (i = 0; i < 8; ++i)
    {
      #pragma unroll
      for (j = 0; j < 8; ++j)
      {
        block[i*8+j] = ((int16_t)in_data[i*w+j+x] - prediction[i*w+j+x]);
      }
    }

    /* Store MBs linear in memory, i.e. the 64 coefficients are stored
       continous. This allows us to ignore stride in DCT/iDCT and other
       functions. */
    dct_quant_block_8x8(block, out_data+(x*8), quantization);

  }
}


__global__ void dct_quantize_dequantize_idct(yuv_t *image, struct c63_common *cm)
{
  // get the y value this thread is responsible for
  int y = threadIdx.x * 8;
  int block = blockIdx.x;

  // DCT IDCT ON Y COMPONENT
  if (block == 0){
    // DCT QUANTIZE Y
    if (y < cm->padh[Y_COMPONENT]){
      dct_quantize_row(
              image->Y+y*cm->padw[Y_COMPONENT],                     // in_data
              cm->curframe->predicted->Y+y*cm->padw[Y_COMPONENT],   // prediction
              cm->padw[Y_COMPONENT],                                // width
              cm->padh[Y_COMPONENT],                                // height
              cm->curframe->residuals->Ydct+y*cm->padw[Y_COMPONENT],// out_data
              cm->quanttbl[Y_COMPONENT]);                           // quantization
    }
   // IDCT DEQUNATIZE Y
    if (y < cm->yph){
      dequantize_idct_row(
              cm->curframe->residuals->Ydct+y*cm->ypw,          // in_data
              cm->curframe->predicted->Y+y*cm->ypw,             // prediction
              cm->ypw,                                          // Width
              cm->yph,                                          // height
              y,                                                // threadidx*8?
              cm->curframe->recons->Y+y*cm->ypw,                // out_data
              cm->quanttbl[Y_COMPONENT]);                       // quantization
    }
  }
  // DCT IDCT ON U COMPONENT
  else if (block == 1){
    // DCT QUANTIZE U
    if (y < cm->padh[U_COMPONENT]){
        dct_quantize_row(
                image->U+y*cm->padw[U_COMPONENT],                     // in_data
                cm->curframe->predicted->U+y*cm->padw[U_COMPONENT],   // prediction
                cm->padw[U_COMPONENT],                                // width
                cm->padh[U_COMPONENT],                                // height
                cm->curframe->residuals->Udct+y*cm->padw[U_COMPONENT],// out_data
                cm->quanttbl[U_COMPONENT]);                           // quantization
     }

     // IDCT DEQUNATIZE U
     if (y < cm->uph){
        dequantize_idct_row(
                cm->curframe->residuals->Udct+y*cm->upw,        // in_data
                cm->curframe->predicted->U+y*cm->upw,           // prediction
                cm->upw,                                        // Width
                cm->uph,                                        // height
                y,                                              // threadidx*8?
                cm->curframe->recons->U+y*cm->upw,              // out_data
                cm->quanttbl[U_COMPONENT]);                     // quantization
    }

  }
  // DCT IDCT ON V COMPONENT
  else {
    // DCT QUANTIZE V
    if (y < cm->padh[V_COMPONENT]){
        dct_quantize_row(
                image->V+y*cm->padw[V_COMPONENT],                     // in_data
                cm->curframe->predicted->V+y*cm->padw[V_COMPONENT],   // prediction
                cm->padw[V_COMPONENT],                                // width
                cm->padh[V_COMPONENT],                                // height
                cm->curframe->residuals->Vdct+y*cm->padw[V_COMPONENT],// out_data
                cm->quanttbl[V_COMPONENT]);                           // quantization
    }
     // IDCT DEQUNATIZE V
    if (y <  cm->vph){
        dequantize_idct_row(
                cm->curframe->residuals->Vdct+y*cm->vpw,        // in_data
                cm->curframe->predicted->V+y*cm->vpw,           // prediction
                cm->vpw,                                        // Width
                cm->vph,                                        // height
                y,                                              // threadidx*8?
                cm->curframe->recons->V+y*cm->vpw,              // out_data
                cm->quanttbl[V_COMPONENT]);                     // quantization
    }
  }
}


void destroy_frame(struct frame *f)
{
  /* First frame doesn't have a reconstructed frame to destroy */
  if (!f) { return; }

  /*
    Free memory from global memory
  */
  cudaFree(f->recons->Y);
  cudaFree(f->recons->U);
  cudaFree(f->recons->V);
  cudaFree(f->recons);

  cudaFree(f->residuals->Ydct);
  cudaFree(f->residuals->Udct);
  cudaFree(f->residuals->Vdct);
  cudaFree(f->residuals);

  cudaFree(f->predicted->Y);
  cudaFree(f->predicted->U);
  cudaFree(f->predicted->V);
  cudaFree(f->predicted);

  cudaFree(f->mbs[Y_COMPONENT]);
  cudaFree(f->mbs[U_COMPONENT]);
  cudaFree(f->mbs[V_COMPONENT]);

  cudaFree(f);
}

struct frame* create_frame(struct c63_common *cm, yuv_t *image)
{
  /*
    Allocate memory to global memory with cudaMallocManaged
  */
  struct frame *f;
  cudaMallocManaged(&f, sizeof(struct frame));

  f->orig = image;

  cudaMallocManaged(&f->recons, sizeof(yuv_t));
  cudaMallocManaged(&f->recons->Y, cm->ypw * cm->yph);
  cudaMallocManaged(&f->recons->U, cm->upw * cm->uph);
  cudaMallocManaged(&f->recons->V, cm->vpw * cm->vph);


  cudaMallocManaged(&f->predicted, sizeof(yuv_t));
  cudaMallocManaged(&f->predicted->Y, cm->ypw * cm->yph*sizeof(uint8_t));
  cudaMallocManaged(&f->predicted->U, cm->upw * cm->uph*sizeof(uint8_t));
  cudaMallocManaged(&f->predicted->V, cm->vpw * cm->vph*sizeof(uint8_t));

  cudaMallocManaged(&f->residuals, sizeof(yuv_t));
  cudaMallocManaged(&f->residuals->Ydct, cm->ypw * cm->yph*sizeof(int16_t));
  cudaMallocManaged(&f->residuals->Udct, cm->upw * cm->uph*sizeof(int16_t));
  cudaMallocManaged(&f->residuals->Vdct, cm->vpw * cm->vph*sizeof(int16_t));

  cudaMallocManaged(&f->mbs[Y_COMPONENT],
    cm->mb_rows * cm->mb_cols * sizeof(struct macroblock));
  cudaMallocManaged(&f->mbs[U_COMPONENT],
    cm->mb_rows/2 * cm->mb_cols/2*sizeof(struct macroblock));
  cudaMallocManaged(&f->mbs[V_COMPONENT],
    cm->mb_rows/2 * cm->mb_cols/2*sizeof(struct macroblock));

  return f;
}

void dump_image(yuv_t *image, int w, int h, FILE *fp)
{
  fwrite(image->Y, 1, w*h, fp);
  fwrite(image->U, 1, w*h/4, fp);
  fwrite(image->V, 1, w*h/4, fp);
}
