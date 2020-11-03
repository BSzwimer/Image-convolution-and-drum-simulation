#include "lodepng.h"
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>  
#include <math.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


__global__ void convolution(unsigned char* test_image, unsigned char* output_image, int size, int width, int height) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int new_index = index * 4 + ((index / width) * -8);
	int other_index = index * 4;
	bool right_padding = !((index + 2) % width < 2);

	/*if (index < 3*width) {
	printf("index : %d and new Index : %d \n", index, new_index);
	}*/

	if (index < (width * height)) {
		if (index < ((height -2) * width) && right_padding){
			for (int i = 0; i < 3; i++) {
				int pOne = test_image[other_index + i];
				int pTwo = test_image[other_index + 4 + i];
				int pThree = test_image[other_index + 8 + i];
				int pFour = test_image[other_index + width * 4 + i];
				int pFive = test_image[other_index + (width * 4) + 4 + i];
				int pSix = test_image[other_index + (width * 4) + 8 + i];
				int pSeven = test_image[other_index + (2 * width * 4) + i];
				int pEight = test_image[other_index + (2 * width * 4) + 4 + i];
				int pNine = test_image[other_index + (2 * width * 4) + 8 + i];


				//weights got from the wm.h file
				float temp = (pFive * 0.25);

				float sum = (pOne * 1.0) + (pTwo * 2.0) + (pThree * (-1.0)) + (pFour * 2.0) + (pFive * 0.25) + (pSix * (-2.0)) + (pSeven * 1.0) + (pEight * (-2.0)) + (pNine * (-1.0));


				if (sum <= 0.0) {
					sum = 0;
				}
				else if (sum >= 255.0) {
					sum = 255;
				}
				sum = round(sum);
				output_image[new_index + i] = (unsigned char)sum;
			}

			output_image[new_index + 3] = test_image[other_index + 3];


		}

	}



}

int main(int argc, char* argv[])
{

	char* input_filename = argv[1];
	char* output_filename = argv[2];
	unsigned int num_of_threads = atoi(argv[3]);

	/*char* input_filename = "Test_1.png";
	char* output_filename = "convolution.png";
	unsigned int num_of_threads = 2;*/

	unsigned char* image;
	unsigned width, height;
	unsigned error = lodepng_decode32_file(&image, &width, &height, input_filename);
	if (error) printf("error %u: %s\n", error, lodepng_error_text(error));

	unsigned int size = width * height * 4 * sizeof(unsigned char);
	unsigned int newSize = ((width - 2) * (height - 2)) * 4 * sizeof(unsigned char);
	unsigned char* og_image, * output_image;
	cudaMalloc((void**)&og_image, size);
	cudaMalloc((void**)&output_image, newSize);


	// CPU copies input data from CPU to GPU
	cudaMemcpy(og_image, image, size, cudaMemcpyHostToDevice);

	//unsigned int num_of_blocks = (((width) * (height) +num_of_threads - 1) / num_of_threads);
	unsigned int num_of_blocks = (((width) * (height)) / num_of_threads) + 1;

	// Initialize CUDA timer
	float memsettime;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	cudaEventSynchronize(start);

	//CUDA kernel call
	convolution << < num_of_blocks, num_of_threads >> > (og_image, output_image, size, width, height);

	// Stop CUDA timer
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&memsettime, start, stop);
	printf("execution time: %f \n", memsettime);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);


	// CPU copies input data from GPU back to CPU
	unsigned char* convolution_image = (unsigned char*)malloc(newSize);
	cudaMemcpy(convolution_image, output_image, newSize, cudaMemcpyDeviceToHost);
	cudaFree(og_image);
	cudaFree(output_image);

	lodepng_encode32_file(output_filename, convolution_image, width - 2, height - 2);

	return 0;
}