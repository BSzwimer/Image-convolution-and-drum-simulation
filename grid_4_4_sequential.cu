#include <stdio.h>
#include <stdlib.h>
#include <algorithm>  
#include <iostream>
using namespace std;
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

int main(int argc, char* argv[])
{
	float memsettime;
	cudaEvent_t start, stop;
	int num_of_iterations = strtol(argv[1], NULL, 10);
	double u[4][4] = { 0 };
	double u1[4][4] = { 0 };
	double u2[4][4] = { 0 };
	double G = 0.75;
	double n = 0.0002;
	double p = 0.5;
	int N = 4;
	

	// Initialize CUDA timer
	
	cudaEventCreate(&start); 
	cudaEventCreate(&stop); 
	cudaEventRecord(start, 0); 
	cudaEventSynchronize(start);

	u1[2][2] = 1;

	for (int counter = 0; counter < num_of_iterations; counter++) {
		
		for (int i = 1; i <= N - 2; i++) {

			for (int j = 1; j <= N - 2; j++) {
				u[i][j] = (p * (u1[i - 1][j] + u1[i + 1][j] + u1[i][j - 1] + u1[i][j + 1] - 4 * u1[i][j]) + 2 * u1[i][j] - (1 - n) * u2[i][j]) / (1 + n);

			}


		}
		// set corner elements
		u[0][0] = G * u[1][0];
		u[N - 1][0] = G * u[N - 2][0];
		u[0][N - 1] = G * u[0][N - 2];
		u[N - 1][N - 1] = G * u[N - 1][N - 2];

		for (int i = 1; i <= N - 2; i++) {
			// set boundary
			u[0][i] = G * u[1][i];
			u[N - 1][i] = G * u[N - 2][i];
			u[i][0] = G * u[i][1];
			u[i][N - 1] = G * u[i][N - 2];
		}

		

		// copy u1 to u2
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < N; j++) {
				u2[i][j] = u1[i][j];
			}
		}

		// copy u to u1
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < N; j++) {
				u1[i][j] = u[i][j];
			}
		}

		// print drum output
		printf("u(2,2) is  %f \n", u[2][2]);

	
	}

	cudaEventRecord(stop, 0); 
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&memsettime, start, stop);
	printf(" Running sequentially on %d : %f \n", num_of_iterations ,memsettime);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	return 0;
}
