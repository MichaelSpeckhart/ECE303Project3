/*
Luke Hale and Michael Speckhart
Project #3
Part #2.4
3/8/2024
*/
#include "files.h"
#include "timer.h"
#include <iostream>
#include <iomanip>
const size_t BLUR_SIZE = 1;
const size_t NUM_BLUR = 10;
const size_t IMAGE_SIZE = 28;
const size_t NUMBER_IMAGES = 1000;

__global__ void blur_kernel(unsigned char *device_in_buffer,unsigned char *device_out_buffer)
{
    unsigned char (*inDeviceImage)[IMAGE_SIZE][IMAGE_SIZE] = (unsigned char (*)[IMAGE_SIZE][IMAGE_SIZE])device_in_buffer;
    unsigned char (*outDeviceImage)[IMAGE_SIZE][IMAGE_SIZE] = (unsigned char (*)[IMAGE_SIZE][IMAGE_SIZE])device_out_buffer;


    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    //increment by stride
    for(size_t image_number = idx; image_number < NUMBER_IMAGES; image_number += stride) {
        // For each pixel
        for(size_t i = 0; i < IMAGE_SIZE; i++) {
            for(size_t j = 0; j < IMAGE_SIZE; j++) {
                size_t sum = 0;
                size_t pixels = 0;
                // Define blur region boundaries
                size_t startX = (i >= BLUR_SIZE) ? (i - BLUR_SIZE) : 0;
                size_t endX = (i + BLUR_SIZE < IMAGE_SIZE) ? (i + BLUR_SIZE) : (IMAGE_SIZE - 1);
                size_t startY = (j >= BLUR_SIZE) ? (j - BLUR_SIZE) : 0;
                size_t endY = (j + BLUR_SIZE < IMAGE_SIZE) ? (j + BLUR_SIZE) : (IMAGE_SIZE - 1);
                // Calculate sum of neighboring pixels
                for (size_t x = startX; x <= endX; x++) {
                    for (size_t y = startY; y <= endY; y++) {
                        sum += inDeviceImage[image_number][x][y];
                        pixels++;
                    }
                }
                outDeviceImage[image_number][i][j] = static_cast<unsigned char>(sum / pixels);
            }
        }
    }
}

//the first step is to read in the MINST data set
// Each file has 1000 training examples. Each training example is of size 28x28 pixels. 
// The pixels are stored as unsigned chars (1 byte) and take values from 0 to 255. 
// The first 28x28 bytes of the file correspond to the first training example, the next 28x28 bytes correspond to the next example and so on.
int main(){

    size_t bytes = NUMBER_IMAGES * IMAGE_SIZE * IMAGE_SIZE * sizeof(unsigned char);
    // Allocate memory for input and output buffers
    unsigned char* hostBuf = (unsigned char*)malloc(bytes);

    // Allocate memory for input and output images
    unsigned char (*hostImage)[IMAGE_SIZE][IMAGE_SIZE] = (unsigned char (*)[IMAGE_SIZE][IMAGE_SIZE])hostBuf;
    
    read_values_from_file("data/data3.txt", hostImage,NUMBER_IMAGES);
    for(size_t x = 0 ; x < 1;x++){
        for(size_t i = 0; i < IMAGE_SIZE;i++){
            for(size_t j = 0; j<IMAGE_SIZE;j++){
                std::cout << std::setw(3) << static_cast<int>(hostImage[0][i][j]) << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
    //get the memory onto the GPU
    unsigned char *device_in_buffer;
    cudaMalloc(&device_in_buffer, bytes);

    unsigned char *device_out_buffer;
    cudaMalloc(&device_out_buffer, bytes);
  
    int threads_per_block = 512;
    int deviceId;
    cudaGetDevice(&deviceId);
  
    int numberOfSMs;
    cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);
    
    int number_of_blocks = 32 * numberOfSMs;

    StartTimer();
    cudaMemcpy(device_in_buffer, hostBuf, bytes, cudaMemcpyHostToDevice);
    //for each input image

    for (size_t blur = 0; blur < NUM_BLUR; blur++) {
        blur_kernel<<<number_of_blocks, threads_per_block>>>(device_in_buffer, device_out_buffer);
        unsigned char *temp = device_in_buffer;
        device_in_buffer = device_out_buffer;
        device_out_buffer = temp;
    }
    cudaMemcpy(hostBuf, device_in_buffer, bytes, cudaMemcpyDeviceToHost);
    std::cout << GetTimer()/NUMBER_IMAGES/NUM_BLUR << " ms per image average" << std::endl;


    for(size_t x = 0 ; x < 1;x++){
        for(size_t i = 0; i < IMAGE_SIZE;i++){
            for(size_t j = 0; j<IMAGE_SIZE;j++){
                std::cout << std::setw(3) << static_cast<int>(hostImage[0][i][j]) << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
    write_values_to_file("dataOut/data3GPU.txt", hostImage,NUMBER_IMAGES);
    free(hostBuf);
    //CUDA FREE
    cudaFree( device_in_buffer );
    cudaFree( device_out_buffer );
    return 0;
}
