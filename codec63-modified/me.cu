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


__global__ void cuda_sad(uint8_t *block1, uint8_t *block2_, int stride, int *result, int bottom, int right){

  int x = blockIdx.x * blockDim.x; + threadIdx.x;
  int y = blockIdx.y * blockDim.y; + threadIdx.y;





  //int v = threadIdx.y;
  //int u = threadIdx.x;
  //printf("indx:%d\tx:%d, y:%d\n",index, threadIdx.x, threadIdx.y );
  //return;
  //if (result[index] > 0 ){

  //}
  //int index = right > bottom ?  y*right+x : y*bottom+x ;
  int index = y*right+x;
  if( bottom*right < index){
    printf("indexing error index %d when max is %d \t (%d,%d)\n", index, bottom*right-1,x,y);
  }

  if (x < right && y < bottom) {

    //printf("\nidx:%d  kernel%d-%d  maxindex:%d.          (%d,%d)\n", index, right, bottom, bottom*right-1,x,y);

    uint8_t *block2 = block2_ + y*stride + x; //DO I NEED TO CHANGE THIS??
    /*__shared__ int tmp[8];

    for(int i =0; i < 8; i++){
      tmp[i] += abs(block2[v*stride+u]- block1[v*stride+u]);
    }

    __syncthreads();
    result[index] = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];

    printf("(%d)\n",result[index]  );*/
    //#pragma unroll
      //int local_result = 0;
      result[index] = 0;
      for (int v = 0; v < 8; ++v)
      {
        for (int u = 0; u < 8; ++u)
        {
            //printf("%s%d%s%d%s%d%s%d\n","u:", u, "\tv:", v, "\tx:", x, ",\ty:", y);
          //  local_result += __vsadu4(block2[v*stride+u], block1[v*stride+u]);


            result[index] += abs(block2[v*stride+u] - block1[v*stride+u]);
            //result[index] += __vsadu4(block2[v*stride+u], block1[v*stride+u]);

            //printf("%s%d%s%d%s%d\n", "sadidx:", u, ", ", v, "\t ", abs(block2[v*stride+u] - block1[v*stride+u]));
          //  printf("idx:%d_kernel (%4d,%4d) - %d\n", index, x, y, result[index]);
        }
      }

      //__syncthreads();
      //result[index] = local_result;


    //printf("%d \tkernel%d-%d\t %d\n", index, right, bottom, result[index]);
    /*
    if (result[index] < 10){
      printf("kernel (%4d,%4d) - %d\n", x, y, result[index]);
      printf("id:%d.\n", index);
    }*/
  }



}


/* Motion estimation for 8x8 block */
static void me_block_8x8(struct c63_common *cm, int mb_x, int mb_y,
    uint8_t *orig, uint8_t *ref, int color_component)
{
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
  int best_sad_2 = INT_MAX;


  dim3 dimGrid(right, bottom);
  /*if (right >= bottom){
    dimGrid=(right, bottom);
  }else{
    dimGrid=(bottom, right);
  }*/

  //dim3 dimBlock(8, 8);

  int *cuda_r;
  int size = sizeof(int)*bottom*right;;

  cudaMallocManaged((void**)&cuda_r, size);
  cuda_sad<<<dimGrid,1>>> (orig+ my*w+mx, ref, w, cuda_r, bottom, right);
  cudaDeviceSynchronize();

  //printf("RES%d\n", cuda_r[0*bottom+0]);

  //exit(1);

  //printf("\nbottom:%d\n", bottom);
//  printf("right:%d\n", right);



    for (y = top; y < bottom; ++y)
    {
      for (x = left; x < right; ++x)
      {
        //printf("Drad\n");
        //int index = right > bottom ?  y*right+x : y*bottom+x ;
        int index = y*right+x;

        //printf("XY (%4d,%4d)\n", x, y);
        //printf("aaatrin index: %d\tSIZE: %d\t\t", index, size);
        //printf("value: %d\n", cuda_r[index]);

        int sad = 0;

/*
        uint8_t *block1 = orig + my*w+mx;
        uint8_t *block2 = ref + y*w+x;

        int *result = &sad;
        int stride = w;


        //printf("Bfor after-sad fors\n");
        *result = 0;
        for (int v = 0; v < 8; ++v)
        {
          for (int u = 0; u < 8; ++u)
          {
            *result += abs(block2[v*stride+u] - block1[v*stride+u]);
          }
        }*/
    //    printf("after-sad fors\n");
/*
        if (cuda_r[index]==0){
          printf(" abs (%4d,%4d) - %d\n", x, y, sad);
          printf("Cuda (%4d,%4d) - %d\n", x, y, cuda_r[index]);
          printf("\nzero error id %d\n", index);
          exit(-100);
        }
         else if (cuda_r[index] != sad){
          printf(" abs (%4d,%4d) - %d\n", x, y, sad);
          printf("Cuda (%4d,%4d) - %d\n", x, y, cuda_r[index]);
          printf("\nvalue error id %d\n", index);
          exit(1);
        }
*/
        //printf("sad - %d\n",best_sad );


        if (cuda_r[index] < best_sad)
      //if  (sad< best_sad)
        {
        //  printf("-1-\t");
          mb->mv_x = x - mx;
          //printf("-2-\t");
          mb->mv_y = y - my;
          //printf("-3-\n"
      //    best_sad_2 = cuda_r[index];
        //  best_sad = sad;
          best_sad = cuda_r[index];
        }
      }
    }

    //    exit(1);




    //exit(1);
  /* Here, there should be a threshold on SAD that checks if the motion vector
     is cheaper than intraprediction. We always assume MV to be beneficial */

     //printf(" ABS Using motion vector (%d, %d) with SAD %d\n", mb->mv_x, mb->mv_y, best_sad);
//     printf("CUDA Using motion vector (%d, %d) with SAD %d\n\n", mb->mv_x, mb->mv_y, best_sad_2);
  //printf("ex--\n" );
  mb->use_mv = 1;
//  printf("EXITING\n" );
  cudaFree(cuda_r);
}

void c63_motion_estimate(struct c63_common *cm)
{
  /* Compare this frame with previous reconstructed frame */
  int mb_x, mb_y;

  /* Luma */
  for (mb_y = 0; mb_y < cm->mb_rows; ++mb_y)
  {
    for (mb_x = 0; mb_x < cm->mb_cols; ++mb_x)
    {
      // <<<block_grid_Y,  thread_grid>>>
      // thread_grid = (8,8)
      // block_grid_Y = (width, height)
      //Block grid: NUM_8x8BLOCKSxNUM_8x8BLOCKS Y component *
      me_block_8x8(cm, mb_x, mb_y, cm->curframe->orig->Y,  cm->refframe->recons->Y, Y_COMPONENT);
    }
  }

  /* Chroma */
  for (mb_y = 0; mb_y < cm->mb_rows / 2; ++mb_y)
  {
    for (mb_x = 0; mb_x < cm->mb_cols / 2; ++mb_x)
    {
      // <<<block_grid_UV, thread_grid>>>
      // block_grid_UV = (upw, uph)
      // thread_grid = (8,8)
      /* Block grid: NUM_8x8BLOCKSxNUM_8x8BLOCKS U and V component */
      me_block_8x8(cm, mb_x, mb_y, cm->curframe->orig->U,  cm->refframe->recons->U, U_COMPONENT);
      me_block_8x8(cm, mb_x, mb_y, cm->curframe->orig->V,  cm->refframe->recons->V, V_COMPONENT);
    }
  }
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
