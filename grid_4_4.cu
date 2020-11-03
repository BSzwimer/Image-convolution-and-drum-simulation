#include <stdio.h>
#include <stdlib.h>
#include <algorithm>  
#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace std;
__global__ void gridin(double* u, double* u1, double* u2, int N) {
	int i = threadIdx.x;
	double G = 0.75;
	double n = 0.0002;
	double p = 0.5;
	

	u[i] = (p * (u1[i - N] + u1[i + N] + u1[i - 1] + u1[i + 1] - 4 * u1[i]) + 2 * u1[i] - (1 - n) * u2[i]) / (1 + n);
	u2[i] = u1[i];
	u1[i] = u[i];

	//BOUNDARIES
	// TR
	if (i < N) {
		u[i] = G * u[i + N];
		u2[i] = u1[i];
		u1[i] = u[i];
	}

	// BR
	else if (i > N * (N - 1)) {
		u[i] = G * u[i - N];
		u2[i] = u1[i];
		u1[i] = u[i];
	}

	// LC
	else if (i % N == 0) {
		u[i] = G * u[i + 1];
		u2[i] = u1[i];
		u1[i] = u[i];
	}

	// RC
	else if (i % N == N - 1) {
		u[i] = G * u[i - 1];
		u2[i] = u1[i];
		u1[i] = u[i];
	}

	//CORNERS
	//TL 
	if (i == 0) {
		u[i] = G * u[N];
		u2[i] = u1[i];
		u1[i] = u[i];
	}

	// BL
	else if (i == (N - 1) * N) {
		u[i] = G * u[(N - 2) * N];
		u2[i] = u1[i];
		u1[i] = u[i];
	}

	// TR
	else if (i == N - 1) {
		u[i] = G * u[N - 2];
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




int main(int argc, char* argv[])
{
	float memsettime;
	cudaEvent_t start, stop;
	int num_of_iterations = strtol(argv[1], NULL, 10);
	double* u;
	double* u1;
	double* u2;

	int N = 4;
	int grid = N * N;
	int arr_len = grid * sizeof(double);

	cudaMallocManaged((void**)&u, arr_len);
	cudaMallocManaged((void**)&u1, arr_len);
	cudaMallocManaged((void**)&u2, arr_len);

	//timer
	
	cudaEventCreate(&start); 
	cudaEventCreate(&stop); 
	cudaEventRecord(start, 0); 
	cudaEventSynchronize(start);

	// hit drum 
	u1[N * N / 2 + N / 2] = 1;

	for (int t = 0; t < num_of_iterations; t++) {

		// CUDA
		gridin << <1, grid >> > (u, u1, u2, N);
		cudaDeviceSynchronize();

	
		printf("u(2,2) is: %f \n", u[10]);
		
	}


	// Stop CUDA timer
	cudaEventRecord(stop, 0); 
	cudaEventSynchronize(stop); 
	cudaEventElapsedTime(&memsettime, start, stop);
	printf("4X4 executed in parallel %d iterations: %f \n", num_of_iterations, memsettime);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cudaFree(u);
	cudaFree(u1);
	cudaFree(u2);




	return 0;
}
