#include <assert.h>
#include <errno.h>
#include <getopt.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C"{
#include "dsp.h"
#include "me.h"
}


/* Motion estimation for 8x8 block */
__global__ static void me_block_8x8(struct c63_common *cm,
    uint8_t *orig, uint8_t *ref, int color_component)
{
  int mb_x = blockIdx.x * blockDim.x + threadIdx.x;
  int mb_y = blockIdx.y * blockDim.y + threadIdx.y;

  //int _x = blockIdx.x * blockDim.x + threadIdx.x;
  //int _y = blockIdx.y * blockDim.y + threadIdx.y;
  //printf("x:%d - y:%d\t block(%d, %d)\n", _x, _y, thread_x, thread_y);

  //mb_x = thread_x;
  //mb_y = thread_y;



  struct macroblock *mb =
    &cm->curframe->mbs[color_component][mb_y*cm->padw[color_component]/8+mb_x];

  int range = cm->me_search_range;

  /* Quarter resolution for chroma channels. */
  if (color_component > 0) { range /= 2; }

  int left = mb_x * 8 - range;
  int top = mb_y * 8 - range;
  int right = mb_x * 8 + range;
  int bottom = mb_y * 8 + range;

  int w = cm->padw[color_component];
  int h = cm->padh[color_component];

  /* Make sure we are within bounds of reference frame. TODO: Support partial
     frame bounds. */
  if (left < 0) { left = 0; }
  if (top < 0) { top = 0; }
  if (right > (w - 8)) { right = w - 8; }
  if (bottom > (h - 8)) { bottom = h - 8; }

  int x, y;

  int mx = mb_x * 8;
  int my = mb_y * 8;

  int best_sad = INT_MAX;


    for (y = top; y < bottom; ++y)
    {
      for (x = left; x < right; ++x)
      {

          //int index = y*right+x;

          int sad = 0;

          uint8_t *block1 = orig + my*w+mx;
          uint8_t *block2 = ref + y*w+x;

          //int *result = &sad;
          int stride = w;



          //  *result = 0;
            for (int v = 0; v < 8; ++v)
            {
              for (int u = 0; u < 8; ++u)
              {
              //  *result += abs(block2[v*stride+u] - block1[v*stride+u]);
                  sad  += abs(block2[v*stride+u] - block1[v*stride+u]);
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

     //printf("Using motion vector (%d, %d) with SAD %d\n", mb->mv_x, mb->mv_y, best_sad);

  //printf("ex--\n" );
  mb->use_mv = 1;

}



void c63_motion_estimate(struct c63_common *cm)
{
  /* Compare this frame with previous reconstructed frame */
  //int mb_x, mb_y;


  // <<<block_grid_UV, thread_grid>>>
  // block_grid_UV = (upw, uph)
  // thread_grid = (8,8)
  // Block grid: NUM_8x8BLOCKSxNUM_8x8BLOCKS U and V component
  /* Luma */
  dim3 Y_dim(cm->mb_rows, cm->mb_cols);
  dim3 block_size(8,8);
  me_block_8x8<<<1, Y_dim>>> (cm, cm->curframe->orig->Y,  cm->refframe->recons->Y, Y_COMPONENT);

  cudaDeviceSynchronize();
  //printf("done Y\n" );
  //exit(1);

  /* Chroma */
  dim3 UV_dim(cm->mb_rows / 2, cm->mb_cols / 2);
  me_block_8x8<<<1,UV_dim>>> (cm, cm->curframe->orig->U,  cm->refframe->recons->U, U_COMPONENT);
  cudaDeviceSynchronize();
  //printf("done U\n" );

  me_block_8x8<<<1, UV_dim>>> (cm, cm->curframe->orig->V,  cm->refframe->recons->V, V_COMPONENT);
  cudaDeviceSynchronize();
  //printf("done V\n" );

}

/* Motion compensation for 8x8 block */
static void mc_block_8x8(struct c63_common *cm, int mb_x, int mb_y,
    uint8_t *predicted, uint8_t *ref, int color_component)
{
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

void c63_motion_compensate(struct c63_common *cm)
{
  int mb_x, mb_y;

  /* Luma */
  for (mb_y = 0; mb_y < cm->mb_rows; ++mb_y)
  {
    for (mb_x = 0; mb_x < cm->mb_cols; ++mb_x)
    {
      mc_block_8x8(cm, mb_x, mb_y, cm->curframe->predicted->Y,
          cm->refframe->recons->Y, Y_COMPONENT);
    }
  }

  /* Chroma */
  for (mb_y = 0; mb_y < cm->mb_rows / 2; ++mb_y)
  {
    for (mb_x = 0; mb_x < cm->mb_cols / 2; ++mb_x)
    {
      mc_block_8x8(cm, mb_x, mb_y, cm->curframe->predicted->U,
          cm->refframe->recons->U, U_COMPONENT);
      mc_block_8x8(cm, mb_x, mb_y, cm->curframe->predicted->V,
          cm->refframe->recons->V, V_COMPONENT);
    }
  }
}
