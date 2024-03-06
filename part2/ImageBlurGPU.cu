#include "files.h"
#include "timer.h"
#include <iostream>
#include <iomanip>


__global__ void blur_kernel(size_t BLUR_SIZE,size_t IMAGE_SIZE,size_t NUMBER_IMAGES)
{

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
                        sum += inImage[image_number][x][y];
                        pixels++;
                    }
                }
                outImage[image_number][i][j] = static_cast<unsigned char>(sum / pixels);
            }
        }
    }
}

//the first step is to read in the MINST data set
// Each file has 1000 training examples. Each training example is of size 28x28 pixels. 
// The pixels are stored as unsigned chars (1 byte) and take values from 0 to 255. 
// The first 28x28 bytes of the file correspond to the first training example, the next 28x28 bytes correspond to the next example and so on.
int main(){
    const size_t BLUR_SIZE = 1;
    const size_t IMAGE_SIZE = 28;
    const size_t NUMBER_IMAGES = 1000;
    unsigned char inImage[NUMBER_IMAGES][IMAGE_SIZE][IMAGE_SIZE];
    
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
    unsigned char outImage[NUMBER_IMAGES][IMAGE_SIZE][IMAGE_SIZE];
    //for each input image
    StartTimer();
    for(size_t many = 0; many <10;many++){
        //call my kernel here

        
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
                std::cout << std::setw(3) << static_cast<int>(outImage[0][i][j]) << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
    write_values_to_file("dataOut/data3.txt", outImage,NUMBER_IMAGES);
    return 0;
}
