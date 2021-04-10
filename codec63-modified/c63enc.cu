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
#include "c63.h"
#include "c63_write.h"
}

#include "me.cuh"
#include "common.cuh"
#include "tables.cuh"

static char *output_file, *input_file;
FILE *outfile;

static int limit_numframes = 0;

static uint32_t width;
static uint32_t height;

/* getopt */
extern int optind;
extern char *optarg;

/* Read planar YUV frames with 4:2:0 chroma sub-sampling */
static yuv_t* read_yuv(FILE *file, struct c63_common *cm)
{
  size_t len = 0;

  /*
    Allocate memory to global memory with cudaMallocManaged
  */
  yuv_t *image;
  cudaMallocManaged(&image, sizeof(*image));

  /* Read Y. The size of Y is the same as the size of the image. The indices
     represents the color component (0 is Y, 1 is U, and 2 is V) */
  cudaMallocManaged(&image->Y, cm->padw[Y_COMPONENT]*cm->padh[Y_COMPONENT]);
  len += fread(image->Y, 1, width*height, file);

  /* Read U. Given 4:2:0 chroma sub-sampling, the size is 1/4 of Y
     because (height/2)*(width/2) = (height*width)/4. */
  cudaMallocManaged(&image->U, cm->padw[U_COMPONENT]*cm->padh[U_COMPONENT]);
  len += fread(image->U, 1, (width*height)/4, file);

  /* Read V. Given 4:2:0 chroma sub-sampling, the size is 1/4 of Y. */
  cudaMallocManaged(&image->V, cm->padw[V_COMPONENT]*cm->padh[V_COMPONENT]);
  len += fread(image->V, 1, (width*height)/4, file);

  if (ferror(file))
  {
    perror("ferror");
    exit(EXIT_FAILURE);
  }

  if (feof(file))
  {
    /*
      Free memory from global memory
    */
    cudaFree(image->Y);
    cudaFree(image->U);
    cudaFree(image->V);
    cudaFree(image);

    return NULL;
  }
  else if (len != width*height*1.5)
  {
    fprintf(stderr, "Reached end of file, but incorrect bytes read.\n");
    fprintf(stderr, "Wrong input? (height: %d width: %d)\n", height, width);

    /*
      Free memory from global memory
    */
    cudaFree(image->Y);
    cudaFree(image->U);
    cudaFree(image->V);
    cudaFree(image);

    return NULL;
  }

  return image;
}




/*
  runs motion estimate, motion compensate and dct, idct.
  encoder runs a bit faster when all the kernels are nested
  that way we dont need to run cudaDeviceSynchronize() inbetween kernels
*/
__global__ static void runner(struct c63_common *cm, yuv_t *image){
  if (!cm->curframe->keyframe)
  {

    /* Motion Estimation and Motion Compensation */
    // run motion estimate kernel with 1 grid, 3 thread one pr Y, U, V
    c63_motion_estimate<<<1 ,3>>>(cm);

  }

    /*
      amount of threads for each dct, idct kernel.
      one grid pr kernel and (cm->yph/8) height/8 threads.
    */
    int threads = cm->yph/8;
    /* DCT and Quantization AND Reconstruct frame for inter-prediction  */
    dct_quantize_dequantize_idct<<<3, threads>>>(image, cm);

}


static void c63_encode_image(struct c63_common *cm, yuv_t *image)
{
  /* Advance to next frame */
  destroy_frame(cm->refframe);
  cm->refframe = cm->curframe;
  cm->curframe = create_frame(cm, image);

  /* Check if keyframe */
  if (cm->framenum == 0 || cm->frames_since_keyframe == cm->keyframe_interval)
  {
    cm->curframe->keyframe = 1;
    cm->frames_since_keyframe = 0;

    fprintf(stderr, " (keyframe) ");
  }
  else { cm->curframe->keyframe = 0; }

  /*
    runs motion estimate, motion compensate and dct, idct.
    encoder runs a bit faster when all the kernels are nested
    that way we dont need to run cudaDeviceSynchronize() inbetween kernels
  */
  runner<<<1,1>>>(cm, image);
  cudaDeviceSynchronize();
  //printf("SEARCH RANGE %d\n", cm->me_search_range);
  /* Function dump_image(), found in common.c, can be used here to check if the
     prediction is correct */

  write_frame(cm);

  ++cm->framenum;
  ++cm->frames_since_keyframe;
}

struct c63_common* init_c63_enc(int width, int height)
{
  int i;
  /*
    Allocate memory to global memory with cudaMallocManaged
  */
  struct c63_common *cm;
  cudaMallocManaged(&cm, sizeof(struct c63_common), cudaMemAttachGlobal);

  cm->width = width;
  cm->height = height;

  cm->padw[Y_COMPONENT] = cm->ypw = (uint32_t)(ceil(width/16.0f)*16);
  cm->padh[Y_COMPONENT] = cm->yph = (uint32_t)(ceil(height/16.0f)*16);
  cm->padw[U_COMPONENT] = cm->upw = (uint32_t)(ceil(width*UX/(YX*8.0f))*8);
  cm->padh[U_COMPONENT] = cm->uph = (uint32_t)(ceil(height*UY/(YY*8.0f))*8);
  cm->padw[V_COMPONENT] = cm->vpw = (uint32_t)(ceil(width*VX/(YX*8.0f))*8);
  cm->padh[V_COMPONENT] = cm->vph = (uint32_t)(ceil(height*VY/(YY*8.0f))*8);

  cm->mb_cols = cm->ypw / 8;
  cm->mb_rows = cm->yph / 8;

  /* Quality parameters -- Home exam deliveries should have original values,
   i.e., quantization factor should be 25, search range should be 16, and the
   keyframe interval should be 100. */
  cm->qp = 25;                  // Constant quantization factor. Range: [1..50]
  cm->me_search_range = 16;     // Pixels in every direction
  cm->keyframe_interval = 100;  // Distance between keyframes

  /* Initialize quantization tables */
  for (i = 0; i < 64; ++i)
  {
    cm->quanttbl[Y_COMPONENT][i] = yquanttbl_def[i] / (cm->qp / 10.0);
    cm->quanttbl[U_COMPONENT][i] = uvquanttbl_def[i] / (cm->qp / 10.0);
    cm->quanttbl[V_COMPONENT][i] = uvquanttbl_def[i] / (cm->qp / 10.0);
  }

  return cm;
}

void free_c63_enc(struct c63_common* cm)
{
  destroy_frame(cm->curframe);
  cudaFree(cm);
}

static void print_help()
{
  printf("Usage: ./c63enc [options] input_file\n");
  printf("Commandline options:\n");
  printf("  -h                             Height of images to compress\n");
  printf("  -w                             Width of images to compress\n");
  printf("  -o                             Output file (.c63)\n");
  printf("  [-f]                           Limit number of frames to encode\n");
  printf("\n");

  exit(EXIT_FAILURE);
}

int main(int argc, char **argv)
{
  //cudaProfilerStart();
  int c;
  yuv_t *image;

  if (argc == 1) { print_help(); }

  while ((c = getopt(argc, argv, "h:w:o:f:i:")) != -1)
  {
    switch (c)
    {
      case 'h':
        height = atoi(optarg);
        break;
      case 'w':
        width = atoi(optarg);
        break;
      case 'o':
        output_file = optarg;
        break;
      case 'f':
        limit_numframes = atoi(optarg);
        break;
      default:
        print_help();
        break;
    }
  }

  if (optind >= argc)
  {
    fprintf(stderr, "Error getting program options, try --help.\n");
    exit(EXIT_FAILURE);
  }

  outfile = fopen(output_file, "wb");

  if (outfile == NULL)
  {
    perror("fopen");
    exit(EXIT_FAILURE);
  }

  struct c63_common *cm = init_c63_enc(width, height);
  cm->e_ctx.fp = outfile;

  input_file = argv[optind];

  if (limit_numframes) { printf("Limited to %d frames.\n", limit_numframes); }

  FILE *infile = fopen(input_file, "rb");

  if (infile == NULL)
  {
    perror("fopen");
    exit(EXIT_FAILURE);
  }

  /* Encode input frames */
  int numframes = 0;

  while (1)
  {
    image = read_yuv(infile, cm);

    if (!image) { break; }

    printf("Encoding frame %d, ", numframes);
    c63_encode_image(cm, image);

    cudaFree(image->Y);
    cudaFree(image->U);
    cudaFree(image->V);
    cudaFree(image);

    printf("Done!\n");

    ++numframes;

    if (limit_numframes && numframes >= limit_numframes) { break; }
  }

  free_c63_enc(cm);
  fclose(outfile);
  fclose(infile);

  //int i, j;
  //for (i = 0; i < 2; ++i)
  //{
  //  printf("int freq[] = {");
  //  for (j = 0; j < ARRAY_SIZE(frequencies[i]); ++j)
  //  {
  //    printf("%d, ", frequencies[i][j]);
  //  }
  //  printf("};\n");
  //}

  return EXIT_SUCCESS;
}
