#include <assert.h>
#include <errno.h>
#include <getopt.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C" {
#include "common.h"
#include "dsp.h"
}
#include <cuda_runtime.h>


void dequantize_idct_row(int16_t *in_data, uint8_t *prediction, int w, int h,
    int y, uint8_t *out_data, uint8_t *quantization)
{
  int x;

  int16_t block[8*8];

  /* Perform the dequantization and iDCT */
  for(x = 0; x < w; x += 8)
  {
    int i, j;

    dequant_idct_block_8x8(in_data+(x*8), block, quantization);

    for (i = 0; i < 8; ++i)
    {
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

void dequantize_idct(int16_t *in_data, uint8_t *prediction, uint32_t width,
    uint32_t height, uint8_t *out_data, uint8_t *quantization)
{
  int y;

  for (y = 0; y < height; y += 8)
  {
    dequantize_idct_row(in_data+y*width, prediction+y*width, width, height, y,
        out_data+y*width, quantization);
  }
}

void dct_quantize_row(uint8_t *in_data, uint8_t *prediction, int w, int h,
    int16_t *out_data, uint8_t *quantization)
{
  int x;

  int16_t block[8*8];

  /* Perform the DCT and quantization */
  for(x = 0; x < w; x += 8)
  {
    int i, j;

    for (i = 0; i < 8; ++i)
    {
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

void dct_quantize(uint8_t *in_data, uint8_t *prediction, uint32_t width,
    uint32_t height, int16_t *out_data, uint8_t *quantization)
{
  int y;

  for (y = 0; y < height; y += 8)
  {
    dct_quantize_row(in_data+y*width, prediction+y*width, width, height,
        out_data+y*width, quantization);
  }
}

void destroy_frame(struct frame *f)
{
  /* First frame doesn't have a reconstructed frame to destroy */
  if (!f) { return; }

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

  //struct frame *f = (frame*)malloc(sizeof(struct frame));
  struct frame *f;
  cudaMallocManaged(&f, sizeof(struct frame), 0x02);

  f->orig = image;

  /* original
    f->recons = malloc(sizeof(yuv_t));
    f->recons->Y = malloc(cm->ypw * cm->yph);
    f->recons->U = malloc(cm->upw * cm->uph);
    f->recons->V = malloc(cm->vpw * cm->vph);
  */
  /*cudaMallocHost(&f->recons, sizeof(yuv_t));
  cudaMallocHost(&f->recons->Y, cm->ypw * cm->yph);
  cudaMallocHost(&f->recons->U, cm->upw * cm->uph);
  cudaMallocHost(&f->recons->V, cm->vpw * cm->vph);
*/
  cudaMallocManaged(&f->recons, sizeof(yuv_t));
  cudaMallocManaged(&f->recons->Y, cm->ypw * cm->yph);
  cudaMallocManaged(&f->recons->U, cm->upw * cm->uph);
  cudaMallocManaged(&f->recons->V, cm->vpw * cm->vph);


  cudaMallocManaged(&f->predicted, sizeof(yuv_t), 0x02);
  cudaMallocManaged(&f->predicted->Y, cm->ypw * cm->yph*sizeof(uint8_t));
  cudaMallocManaged(&f->predicted->U, cm->upw * cm->uph*sizeof(uint8_t));
  cudaMallocManaged(&f->predicted->V, cm->vpw * cm->vph*sizeof(uint8_t));

  cudaMallocManaged(&f->residuals, sizeof(yuv_t));
  cudaMallocManaged(&f->residuals->Ydct, cm->ypw * cm->yph*sizeof(int16_t));
  cudaMallocManaged(&f->residuals->Udct, cm->upw * cm->uph*sizeof(int16_t));
  cudaMallocManaged(&f->residuals->Vdct, cm->vpw * cm->vph*sizeof(int16_t));

  cudaMallocManaged(&f->mbs[Y_COMPONENT], cm->mb_rows * cm->mb_cols * sizeof(struct macroblock));
  cudaMallocManaged(&f->mbs[U_COMPONENT], cm->mb_rows/2 * cm->mb_cols/2*sizeof(struct macroblock));
  cudaMallocManaged(&f->mbs[V_COMPONENT], cm->mb_rows/2 * cm->mb_cols/2*sizeof(struct macroblock));

  /*
  //f->predicted = (yuv_t*)malloc(sizeof(yuv_t));
  f->predicted->Y = (uint8_t*)calloc(cm->ypw * cm->yph, sizeof(uint8_t));
  f->predicted->U = (uint8_t*)calloc(cm->upw * cm->uph, sizeof(uint8_t));
  f->predicted->V = (uint8_t*)calloc(cm->vpw * cm->vph, sizeof(uint8_t));
  */
  /*
  f->residuals = (dct_t*)malloc(sizeof(dct_t));
  f->residuals->Ydct = (int16_t*)calloc(cm->ypw * cm->yph, sizeof(int16_t));
  f->residuals->Udct = (int16_t*)calloc(cm->upw * cm->uph, sizeof(int16_t));
  f->residuals->Vdct = (int16_t*)calloc(cm->vpw * cm->vph, sizeof(int16_t));
*/
/*
  f->mbs[Y_COMPONENT] =
    (macroblock*)calloc(cm->mb_rows * cm->mb_cols, sizeof(struct macroblock));
  f->mbs[U_COMPONENT] =
    (macroblock*)calloc(cm->mb_rows/2 * cm->mb_cols/2, sizeof(struct macroblock));
  f->mbs[V_COMPONENT] =
    (macroblock*)calloc(cm->mb_rows/2 * cm->mb_cols/2, sizeof(struct macroblock));
*/
  return f;
}

void dump_image(yuv_t *image, int w, int h, FILE *fp)
{
  fwrite(image->Y, 1, w*h, fp);
  fwrite(image->U, 1, w*h/4, fp);
  fwrite(image->V, 1, w*h/4, fp);
}
