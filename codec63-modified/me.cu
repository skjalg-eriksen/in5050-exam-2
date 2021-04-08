#include <assert.h>
#include <errno.h>
#include <getopt.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

//extern "C"{}
#include "dsp.cuh"
#include "me.cuh"
// #include <cuda_fp16.h>

/* Motion estimation for 8x8 block */
__global__ static void me_block_8x8(struct c63_common *cm,
    uint8_t *orig, uint8_t *ref, int color_component)
{
  // retrive mb_y and mb_x from indexes.
  int mb_y = blockIdx.x * blockDim.x + threadIdx.x;
  int mb_x = blockIdx.y * blockDim.y + threadIdx.y;

  struct macroblock *mb =
    &cm->curframe->mbs[color_component][mb_y*cm->padw[color_component]/8+mb_x];

  int range = cm->me_search_range;

  //Quarter resolution for chroma channels.
  if (color_component > 0) { range /= 2; }

  int left = mb_x * 8 - range;
  int top = mb_y * 8 - range;
  int right = mb_x * 8 + range;
  int bottom = mb_y * 8 + range;

  int w = cm->padw[color_component];
  int h = cm->padh[color_component];

  // Make sure we are within bounds of reference frame. TODO: Support partial frame bounds.
  if (left < 0) { left = 0; }
  if (top < 0) { top = 0; }
  if (right > (w - 8)) { right = w - 8; }
  if (bottom > (h - 8)) { bottom = h - 8; }

  int x, y;
  // block pointers
  uint8_t *block1, *block2;

  int mx = mb_x * 8;
  int my = mb_y * 8;

  int best_sad = INT_MAX;
  int sad;

    for (y = top; y < bottom; ++y)
    {
      for (x = left; x < right; ++x)
      {

          sad = 0;
          // update block pointers
          block1 = orig + my*w+mx;
          block2 = ref + y*w+x;

            // calculate sad block 8x8, a lot faster here than in dsp.cu
            #pragma unroll
            for (int v = 0; v < 8; ++v)
            {
              #pragma unroll
              for (int u = 0; u < 8; ++u)
              {
                  sad  += abs(block2[v*w+u] - block1[v*w+u]);
              }
            }

          if  (sad< best_sad)
          {
            mb->mv_x = x - mx;
            mb->mv_y = y - my;
            best_sad = sad;
          }


      }
    }

  /* Here, there should be a threshold on SAD that checks if the motion vector
     is cheaper than intraprediction. We always assume MV to be beneficial */

  //printf("(%d,%d) Using motion vector (%d, %d) with SAD %d\n",mb_x, mb_y, mb->mv_x, mb->mv_y, best_sad);

  mb->use_mv = 1;
}


__global__ void c63_motion_estimate(struct c63_common *cm)
{
  // Dim3 for Y
  dim3 Y_dim(cm->mb_rows, cm->mb_cols);
  // Dim3 for UV
  dim3 UV_dim(cm->mb_rows / 2, cm->mb_cols / 2);

  /* Luma */
  // Calculate Y, kernel with Ydim (rows, cols) and 1 thread
  me_block_8x8 <<<Y_dim, 1>>>(cm, cm->curframe->orig->Y,
      cm->refframe->recons->Y, Y_COMPONENT);


  /* Chroma */
  // Calculate U, kernel with Ydim (rows, cols) and 1 thread
  me_block_8x8<<<UV_dim, 1>>> (cm, cm->curframe->orig->U,
      cm->refframe->recons->U, U_COMPONENT);

  // Calculate V, kernel with Ydim (rows, cols) and 1 thread
  me_block_8x8<<<UV_dim, 1>>> (cm, cm->curframe->orig->V,
      cm->refframe->recons->V, V_COMPONENT);
}


/* Motion compensation for 8x8 block */
__global__ static void mc_block_8x8(struct c63_common *cm,
    uint8_t *predicted, uint8_t *ref, int color_component)
{
  // retrive mb_y and mb_x from indexes.
  int mb_y = blockIdx.x * blockDim.x + threadIdx.x;
  int mb_x = blockIdx.y * blockDim.y + threadIdx.y;

  struct macroblock *mb =
    &cm->curframe->mbs[color_component][mb_y*cm->padw[color_component]/8+mb_x];

  if (!mb->use_mv) { return; }

  int left = mb_x * 8;
  int top = mb_y * 8;
  int right = left + 8;
  int bottom = top + 8;

  int w = cm->padw[color_component];

  /* Copy block from ref mandated by MV */
  int x, y;

  for (y = top; y < bottom; ++y)
  {
    for (x = left; x < right; ++x)
    {
      predicted[y*w+x] = ref[(y + mb->mv_y) * w + (x + mb->mv_x)];
    }
  }
}

__global__ void c63_motion_compensate(struct c63_common *cm)
{
  // Dim3 for Y
  dim3 Y_dim(cm->mb_rows, cm->mb_cols);
  // Dim3 for UV
  dim3 UV_dim(cm->mb_rows / 2, cm->mb_cols / 2);

  /* Luma */
  // Calculate Y, kernel with Ydim (rows, cols) and 1 thread
  mc_block_8x8 <<<Y_dim, 1>>> (cm, cm->curframe->predicted->Y, cm->refframe->recons->Y, Y_COMPONENT);

  /* Chroma */
  // Calculate U, kernel with Ydim (rows, cols) and 1 thread
  mc_block_8x8 <<<UV_dim, 1>>>  (cm, cm->curframe->predicted->U, cm->refframe->recons->U, U_COMPONENT);
  // Calculate V, kernel with Ydim (rows, cols) and 1 thread
  mc_block_8x8 <<<UV_dim, 1>>>  (cm, cm->curframe->predicted->V, cm->refframe->recons->V, V_COMPONENT);

}
