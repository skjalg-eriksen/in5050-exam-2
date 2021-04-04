#define N 4
#define T 1024 // max threads per block
#include <stdio.h>
#include <inttypes.h>
__global__ void vecAdd (int *a, int *b, int *c);



int main() {
	uint8_t a[N] = {2, 4, 5, 1}; // 12
	uint8_t b[N] = {3, 7, 8, 1}; // 19
	int c[N];//=0;
	int *dev_a, *dev_b, *dev_c;
	// initialize a and b with real values (NOT SHOWN)
	int size = N * sizeof(int);
	cudaMalloc((void**)&dev_a, size);
	cudaMalloc((void**)&dev_b, size);
	cudaMalloc((void**)&dev_c, size);
	cudaMemcpy(dev_a, a, size,cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, size,cudaMemcpyHostToDevice);
	
	vecAdd<<<1,4>>>(dev_a,dev_b,dev_c);
	printf("\naaa\n");
	cudaMemcpy(&c, dev_c, size, cudaMemcpyDeviceToHost);
	
	for (int i=0; i < sizeof(*c); i++){
		printf("vec %d\n", c[i]);		
	}
	//printf("vec %d\n", *c);		
	
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
	
	//printf("\nvec %d\n", c);		
	
	exit (0);
}


__global__ void vecAdd (int *a, int *b, int *c) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < N) {
		//c[0] += a[i] + b[i];
//		atomicAdd(&c[3], (int)(a[i]+b[i]));
		c[0] = __vsadu4(a[i], b[i]);
		
		printf("abs:%d\n", __vsadu4(a[i], b[i]));
	}


}

