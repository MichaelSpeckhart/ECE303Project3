/*
Luke Hale and Michael Speckhart
Project #3
Part #2.6
3/8/2024
*/
#include <iostream>
#include <cmath>
#include <iomanip>
#include "files.h"
#include "timer.h"

const size_t BLUR_SIZE = 1;
const size_t NUM_BLUR = 10;
const size_t IMAGE_SIZE = 28;
const size_t NUMBER_IMAGES = 1000;

// Function to generate a 2D Gaussian kernel
__global__
void generateGaussianKernel(double* kernel, int kernelSize, double sigma) {
    int center = kernelSize / 2;
    double sum = 0.0;

    for (int i = 0; i < kernelSize; ++i) {
        for (int j = 0; j < kernelSize; ++j) {
            int x = i - center;
            int y = j - center;
            kernel[i * kernelSize + j] = exp(-(x * x + y * y) / (2 * sigma * sigma)) / (2 * M_PI * sigma * sigma);
            sum += kernel[i * kernelSize + j];
        }
    }

    // Normalize kernel values
    for (int i = 0; i < kernelSize * kernelSize; ++i) {
        kernel[i] /= sum;
    }
}

__global__ void blur_kernel(unsigned char *device_in_buffer, unsigned char *device_out_buffer, double* gaussianKernel, int kernelSize) {
    unsigned char (*inDeviceImage)[IMAGE_SIZE][IMAGE_SIZE] = (unsigned char (*)[IMAGE_SIZE][IMAGE_SIZE])device_in_buffer;
    unsigned char (*outDeviceImage)[IMAGE_SIZE][IMAGE_SIZE] = (unsigned char (*)[IMAGE_SIZE][IMAGE_SIZE])device_out_buffer;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    
    for (size_t image_number = idx; image_number < NUMBER_IMAGES; image_number += stride) {
        
        for (size_t i = 0; i < IMAGE_SIZE; i++) {
            for (size_t j = 0; j < IMAGE_SIZE; j++) {
                double sum = 0.0;

                
                for (int k = -BLUR_SIZE; k <= BLUR_SIZE; k++) {
                    for (int l = -BLUR_SIZE; l <= BLUR_SIZE; l++) {
                        
                        int x_kernel = k + BLUR_SIZE;
                        int y_kernel = l + BLUR_SIZE;

                        
                        int x_image = i + k;
                        int y_image = j + l;

                        // Check boundary conditions
                        if (x_image >= 0 && x_image < IMAGE_SIZE && y_image >= 0 && y_image < IMAGE_SIZE) {
                            
                            int kernel_index = x_kernel * kernelSize + y_kernel;

                            
                            sum += gaussianKernel[kernel_index] * inDeviceImage[image_number][x_image][y_image];
                        }
                    }
                }

                
                outDeviceImage[image_number][i][j] = static_cast<unsigned char>(sum);
            }
        }
    }
}

int main() {
    // Read input image data
    size_t bytes = NUMBER_IMAGES * IMAGE_SIZE * IMAGE_SIZE * sizeof(unsigned char);
    unsigned char* hostBuf = (unsigned char*)malloc(bytes);
    unsigned char (*hostImage)[IMAGE_SIZE][IMAGE_SIZE] = (unsigned char (*)[IMAGE_SIZE][IMAGE_SIZE])hostBuf;
    read_values_from_file("data/data3.txt", hostImage, NUMBER_IMAGES);

    std::cout << "Image before blurring: \n";
    for(size_t x = 0 ; x < 1;x++){
        for(size_t i = 0; i < IMAGE_SIZE;i++){
            for(size_t j = 0; j<IMAGE_SIZE;j++){
                std::cout << std::setw(3) << static_cast<int>(hostImage[0][i][j]) << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    // Allocate device memory
    unsigned char *device_in_buffer;
    cudaMalloc(&device_in_buffer, bytes);
    unsigned char *device_out_buffer;
    cudaMalloc(&device_out_buffer, bytes);

    // Copy input data from host to device
    cudaMemcpy(device_in_buffer, hostBuf, bytes, cudaMemcpyHostToDevice);

    // Define Gaussian kernel size and sigma
    const int kernelSize = 5;  // Adjust as needed
    const double sigma = 1.0;  // Adjust as needed

    // Generate Gaussian kernel on host
    double* gaussianKernel;
    cudaMalloc(&gaussianKernel, kernelSize * kernelSize * sizeof(double));
    generateGaussianKernel<<<1, 1>>>(gaussianKernel, kernelSize, sigma);

    // Launch kernel
    int threads_per_block = 512;
    int deviceId;
    cudaGetDevice(&deviceId);
    int numberOfSMs;
    cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);
    int number_of_blocks = 32 * numberOfSMs;
    StartTimer();
    for (size_t blur = 0; blur < NUM_BLUR; blur++) {
        blur_kernel<<<number_of_blocks, threads_per_block>>>(device_in_buffer, device_out_buffer, gaussianKernel, kernelSize);
        unsigned char *temp = device_in_buffer;
        device_in_buffer = device_out_buffer;
        device_out_buffer = temp;
    }
    std::cout << GetTimer() / NUMBER_IMAGES / NUM_BLUR << " ms per image average" << std::endl;

    // Copy output data from device to host
    cudaMemcpy(hostBuf, device_in_buffer, bytes, cudaMemcpyDeviceToHost);

    for(size_t x = 0 ; x < 1;x++){
        for(size_t i = 0; i < IMAGE_SIZE;i++){
            for(size_t j = 0; j<IMAGE_SIZE;j++){
                std::cout << std::setw(3) << static_cast<int>(hostImage[0][i][j]) << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    // Write blurred image data to file
    write_values_to_file("dataOut/data3GPU.txt", hostImage, NUMBER_IMAGES);

    // Free memory
    free(hostBuf);
    cudaFree(device_in_buffer);
    cudaFree(device_out_buffer);
    cudaFree(gaussianKernel);

    return 0;
}
