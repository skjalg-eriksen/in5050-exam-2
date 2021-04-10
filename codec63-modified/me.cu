#include <assert.h>
#include <errno.h>
#include <getopt.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dsp.cuh"
#include "me.cuh"


/* Motion estimation for 8x8 block */
__global__ static void me_block_8x8(struct c63_common *cm,
    uint8_t *orig, uint8_t *ref, int color_component)
{
  __shared__ int s[8][8];

  // get  mb_y and mb_x from block indexes.
  int mb_y = blockIdx.x;
  int mb_x = blockIdx.y;


  // get x, y thread indexes
  int index_x = threadIdx.x;
  int index_y = threadIdx.y;

  struct macroblock *mb  = &cm->curframe->mbs[color_component][mb_y*cm->padw[color_component]/8+mb_x];


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

  for (y = top; y < bottom; y++)
  {
    for (x = left; x < right; x++)
    {
      //printf("top: %5d\t bottom %5d\t diff %2d\t thread %d\t (%d,%d)\t org diff: %d\n", top, bottom, bottom-top, threadIdx.z, x,y,diff);
      sad = 0;
      // update block pointers
      block1 = orig + my*w+mx;
      block2 = ref + y*w+x;

      // let each thread calculat their calculate sad value, this replaces sad_block_8x8
      __syncthreads();
       s[index_x][index_y] = abs(block2[index_x * w + index_y] - block1[index_x * w + index_y]);
      __syncthreads();


      // reduce along y axis
      if(index_y < 4){
        s[index_x][index_y] += s[index_x][7-index_y];
      }
      __syncthreads();
      if (index_y < 2) {
        s[index_x][index_y] += s[index_x][3-index_y];
      }
      __syncthreads();
      if (index_y < 1){
        s[index_x][index_y] += s[index_x][1];
      }
      __syncthreads();

      // reduce along x axis
      if(index_x < 4  && index_y == 0){
        s[index_x][0] += s[7-index_x][0];
      }
      __syncthreads();
      if (index_x < 2  && index_y == 0) {
        s[index_x][0] += s[3-index_x][0];
      }
      __syncthreads();
      if (index_x < 1 && index_y == 0){
        s[index_x][index_y] += s[1][index_y];
      }



      // sum and check best sad in thread index_x = 0 index_y =0
      if (index_x ==0 && index_y == 0) {
        // get sum sad value
        sad = s[0][0];

        // do the normal sad check
        if  (sad< best_sad)
        {
          mb->mv_x = x - mx;
          mb->mv_y = y - my;
          best_sad = sad;
        }
      }
    }
  }

  /* Here, there should be a threshold on SAD that checks if the motion vector
     is cheaper than intraprediction. We always assume MV to be beneficial */
     //if (index ==0)  printf("(%d,%d) Using motion vector (%d, %d) with SAD %d\n",mb_x, mb_y, mb->mv_x, mb->mv_y, best_sad);
  //return;
  mb->use_mv = 1;

}
/* Motion compensation for 8x8 block */
__global__ static void mc_block_8x8(struct c63_common *cm,
    uint8_t *predicted, uint8_t *ref, int color_component)
{
  // retrive mb_y and mb_x from indexes.
  int mb_y = blockIdx.x;
  int mb_x = blockIdx.y;

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

__global__ void c63_motion_estimate_motion_compensate(struct c63_common *cm)
{
  // 8 by 8 threads (64 total) ine each grid block
  dim3 threads(8,8);

  /* Luma */
  // Calculate Y, kernel with Ydim and 1 thread
  if (threadIdx.x == 0){
    // Dim3 for Y gridblock, (image rows, image cols
    dim3 Y_dim(cm->mb_rows, cm->mb_cols);

    me_block_8x8 <<<Y_dim, threads>>>(cm, cm->curframe->orig->Y,
        cm->refframe->recons->Y, Y_COMPONENT);
    mc_block_8x8 <<<Y_dim, 1>>> (cm, cm->curframe->predicted->Y,
        cm->refframe->recons->Y, Y_COMPONENT);

    return; // save a little bit of time by this thread returning early.
  }

  /* Chroma */
  // Calculate U, kernel with UVdim and 1 thread
  if (threadIdx.x == 1){
    // Dim3 for UV gridblock, (image rows / 2, image cols / 2)    half of Ydim
    dim3 UV_dim(cm->mb_rows / 2, cm->mb_cols / 2);

    me_block_8x8<<<UV_dim, threads>>> (cm, cm->curframe->orig->U,
        cm->refframe->recons->U, U_COMPONENT);
    mc_block_8x8 <<<UV_dim, 1>>>  (cm, cm->curframe->predicted->U,
        cm->refframe->recons->U, U_COMPONENT);

    return;  // save a little bit of time by this thread returning early.
  }

  // Calculate V, kernel with UVdim and 1 thread
  if (threadIdx.x == 2){
    // Dim3 for UV gridblock, (image rows / 2, image cols / 2)    half of Ydim
    dim3 UV_dim(cm->mb_rows / 2, cm->mb_cols / 2);

    // motion motion estimate
    me_block_8x8<<<UV_dim, threads>>> (cm, cm->curframe->orig->V,
      cm->refframe->recons->V, V_COMPONENT);
    // motion compensate
    mc_block_8x8 <<<UV_dim, 1>>>  (cm, cm->curframe->predicted->V,
      cm->refframe->recons->V, V_COMPONENT);

    return;  // save a little bit of time by this thread returning early.
  }

}
