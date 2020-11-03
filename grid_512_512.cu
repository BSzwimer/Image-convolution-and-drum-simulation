#include <stdio.h>
#include <stdlib.h>
#include <algorithm>  
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

__global__ void boundaryAndCorner(int num_of_elem, int N, float* u, float* u1, float* u2) {
	int index = (threadIdx.x + blockIdx.x * blockDim.x) * num_of_elem;

	float G = 0.75;

	for (int i = index; i < index + num_of_elem; i++) {
		// TR
		if (i < N) {
			u[i] = G * u[i + N];
			u2[i] = u1[i];
			u1[i] = u[i];
		}
		// LC
		else if (i % N == 0) {
			u[i] = G * u[i + 1];
			u2[i] = u1[i];
			u1[i] = u[i];
		}

		// BR
		else if (i > N * (N - 1)) {
			u[i] = G * u[i - N];
			u2[i] = u1[i];
			u1[i] = u[i];
		}


		// RC
		else if (i % N == N - 1) {
			u[i] = G * u[i - 1];
			u2[i] = u1[i];
			u1[i] = u[i];
		}

	}

	//Corners
	for (int i = index; i < index + num_of_elem; i++) {
		// TL
		if (i == 0) {
			u[i] = G * u[N];
			u2[i] = u1[i];
			u1[i] = u[i];
		}
		// TR
		else if (i == N - 1) {
			u[i] = G * u[N - 2];
			u2[i] = u1[i];
			u1[i] = u[i];
		}

		// BL
		else if (i == (N - 1) * N) {
			u[i] = G * u[(N - 2) * N];
			u2[i] = u1[i];
			u1[i] = u[i];
		}


		//BR
		else if (i == (N - 1) * N + N - 1) {

			u[i] = G * u[(N - 1) * N + N - 2];
			u2[i] = u1[i];
			u1[i] = u[i];
		}
	}



}
__global__ void interior(int num_of_elem, int N, float* u, float* u1, float* u2 ) {
	int index = (threadIdx.x + blockIdx.x * blockDim.x) * num_of_elem;
	float p = 0.5;

	float n = 0.0002;

	for (int i = index; i < index + num_of_elem; i++) {

		if (!(i < N || i % N == 0 || i % N == N - 1 || i > N * (N - 1))) {
			u[i] = (p * (u1[i - N] + u1[i + N] + u1[i - 1] + u1[i + 1] - 4 * u1[i]) + 2 * u1[i] - (1 - n) * u2[i]) / (1 + n);
		}

	}

	// update after calculation
	for (int i = index; i < index + num_of_elem; i++) {

		u2[i] = u1[i];
		u1[i] = u[i];
	}
}



int main(int argc, char* argv[])
{
	
	float memsettime;
	cudaEvent_t start, stop;
	int num_of_iterations = strtol(argv[1], NULL, 10);
	int num_of_threads = 1024;
	int num_of_blocks = 16;
	int N = 512;
	
	float* u;
	float* u1;
	float* u2;

	int size_of_grid = N * N;
	int num_of_elem = size_of_grid / (num_of_threads * num_of_blocks);
	long size_of_arr = size_of_grid * sizeof(float);

	

	cudaMallocManaged((void**)&u, size_of_arr);

	cudaMallocManaged((void**)&u1, size_of_arr);

	cudaMallocManaged((void**)&u2, size_of_arr);


	
	
	


	u1[N * N / 2 + N / 2] = 1;

	printf("\n N is: %d \n Number of blocks is: %d \n Number of threads is: %d \n Number of elements is: %d \n", N, num_of_blocks, num_of_threads,num_of_elem);
// Timer creation
	cudaEventCreate(&start); 
	cudaEventCreate(&stop); 
	cudaEventRecord(start, 0);
	cudaEventSynchronize(start);
	for (int counter = 0; counter < num_of_iterations; counter++) {

		//// CUDA 
		interior << <num_of_blocks, num_of_threads >> > (num_of_elem,N, u, u1, u2);

		cudaDeviceSynchronize();

		boundaryAndCorner << <num_of_blocks, num_of_threads >> > (num_of_elem, N, u, u1, u2);

		cudaDeviceSynchronize();

		//output to console
		printf("This is iteration %d: %f\n", counter, u[N * N / 2 + N / 2]);
	}


	// Stop timer
	cudaEventRecord(stop, 0); 
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&memsettime, start, stop);
	printf("execution time: %f \n", memsettime);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cudaFree(u);
	cudaFree(u1);
	cudaFree(u2);

	return 0;
}
