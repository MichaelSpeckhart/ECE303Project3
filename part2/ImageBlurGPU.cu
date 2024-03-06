#include "files.h"
#include "timer.h"
#include <iostream>
#include <iomanip>
const size_t BLUR_SIZE = 1;
const size_t IMAGE_SIZE = 28;
const size_t NUMBER_IMAGES = 1;

__global__ void blur_kernel(unsigned char *device_in_buffer,unsigned char *device_out_buffer)
{
    unsigned char (*inDeviceImage)[IMAGE_SIZE][IMAGE_SIZE] = (unsigned char (*)[IMAGE_SIZE][IMAGE_SIZE])device_in_buffer;
    unsigned char (*outDeviceImage)[IMAGE_SIZE][IMAGE_SIZE] = (unsigned char (*)[IMAGE_SIZE][IMAGE_SIZE])device_out_buffer;


    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
//  for (int i = idx; i < n; i += stride) {
    for(size_t image_number = idx; image_number < NUMBER_IMAGES; stride++) {
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
    unsigned char* inBuf = (unsigned char*)malloc(bytes);
    unsigned char* outBuf = (unsigned char*)malloc(bytes);

    // Allocate memory for input and output images
    unsigned char (*inImage)[IMAGE_SIZE][IMAGE_SIZE] = (unsigned char (*)[IMAGE_SIZE][IMAGE_SIZE])inBuf;
    unsigned char (*outImage)[IMAGE_SIZE][IMAGE_SIZE] = (unsigned char (*)[IMAGE_SIZE][IMAGE_SIZE])outBuf;
    
    read_values_from_file("data/data3.txt", inImage,NUMBER_IMAGES);
    for(size_t x = 0 ; x < 1;x++){
        for(size_t i = 0; i < IMAGE_SIZE;i++){
            for(size_t j = 0; j<IMAGE_SIZE;j++){
                std::cout << std::setw(3) << static_cast<int>(inImage[0][i][j]) << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
    //get the memory onto the GPU
    unsigned char *device_in_buffer;
    cudaMalloc(&device_in_buffer, bytes);
    //cudaMemcpy(device_in_buffer, inBuf, bytes, cudaMemcpyHostToDevice);

    unsigned char *device_out_buffer;
    cudaMalloc(&device_out_buffer, bytes);
    cudaMemcpy(device_out_buffer, inBuf, bytes, cudaMemcpyHostToDevice);
  
    int threads_per_block = 512;
    int deviceId;
    cudaGetDevice(&deviceId);
  
    int numberOfSMs;
    cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);
    
    int number_of_blocks = 32 * numberOfSMs;


    //for each input image
    StartTimer();
    for(size_t many = 0; many <10;many++){
        std::cerr << "HI!" << std::endl;
        //call my kernel here
        cudaMemcpy(device_in_buffer, inBuf, bytes, cudaMemcpyHostToDevice);
        std::cerr << "Hello" << std::endl;
        blur_kernel<<<number_of_blocks, threads_per_block>>>(device_in_buffer, device_out_buffer);
        std::cerr << "goodBye" << std::endl;
        cudaMemcpy(outBuf, device_out_buffer, bytes, cudaMemcpyDeviceToHost);
        std::cerr << "bye" << std::endl;
                // Copy outImage to inImage
        for (size_t i = 0; i < NUMBER_IMAGES; i++) {
            for (size_t j = 0; j < IMAGE_SIZE; j++) {
                for (size_t k = 0; k < IMAGE_SIZE; k++) {
                    inImage[i][j][k] = outImage[i][j][k];
                }
            }
        }
    }
    std::cout << GetTimer()/NUMBER_IMAGES/10 << " ms per image average" << std::endl;

    for(size_t x = 0 ; x < 1;x++){
        for(size_t i = 0; i < IMAGE_SIZE;i++){
            for(size_t j = 0; j<IMAGE_SIZE;j++){
                std::cout << std::setw(3) << static_cast<int>(inImage[0][i][j]) << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
    write_values_to_file("dataOut/data3.txt", inImage,NUMBER_IMAGES);
    free(inBuf);
    free(outBuf);
    //CUDA FREE
    cudaFree( device_in_buffer );
    cudaFree( device_out_buffer );
    return 0;
}
