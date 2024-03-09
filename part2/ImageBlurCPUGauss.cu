/*
Luke Hale and Michael Speckhart
Project #3
Part #2.2
3/8/2024
*/
#include "files.h"
#include "timer.h"
#include <iostream>
#include <iomanip>

const size_t BLUR_SIZE = 2;
const size_t NUM_BLUR = 10;
const size_t IMAGE_SIZE = 28;
const size_t NUMBER_IMAGES = 1000;
const int GAUSS_FILTER[5][5] = {
    { 1,  4,  6,  4,  1 },
    { 4, 16, 24, 16,  4 },
    { 6, 24, 36, 24,  6 },
    { 4, 16, 24, 16,  4 },
    { 1,  4,  6,  4,  1 }
};
//the first step is to read in the MINST data set
// Each file has 1000 training examples. Each training example is of size 28x28 pixels. 
// The pixels are stored as unsigned chars (1 byte) and take values from 0 to 255. 
// The first 28x28 bytes of the file correspond to the first training example, the next 28x28 bytes correspond to the next example and so on.
int main(){

    // Allocate memory for input and output buffers
    unsigned char* inBuf = (unsigned char*)malloc(NUMBER_IMAGES * IMAGE_SIZE * IMAGE_SIZE * sizeof(unsigned char));
    unsigned char* outBuf = (unsigned char*)malloc(NUMBER_IMAGES * IMAGE_SIZE * IMAGE_SIZE * sizeof(unsigned char));

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
    
    //for each input image
    StartTimer();
    for(size_t blur = 0; blur <NUM_BLUR;blur++){
        for(size_t image_number = 0; image_number < NUMBER_IMAGES; image_number++) {
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
                            size_t filterX = x - (i - BLUR_SIZE);
                            size_t filterY = y - (j - BLUR_SIZE);
                            sum += inImage[image_number][x][y] * GAUSS_FILTER[filterX][filterY];
                            pixels += GAUSS_FILTER[filterX][filterY];
                        }
                    }
                    outImage[image_number][i][j] = static_cast<unsigned char>(sum / pixels);
                }
            }
        }
        for (size_t i = 0; i < NUMBER_IMAGES; i++) {
            for (size_t j = 0; j < IMAGE_SIZE; j++) {
                for (size_t k = 0; k < IMAGE_SIZE; k++) {
                    inImage[i][j][k] = outImage[i][j][k];
                }
            }
        }
    }
    std::cout << GetTimer()/NUMBER_IMAGES/NUM_BLUR << " ms per image average" << std::endl;

    for(size_t x = 0 ; x < 1;x++){
        for(size_t i = 0; i < IMAGE_SIZE;i++){
            for(size_t j = 0; j<IMAGE_SIZE;j++){
                std::cout << std::setw(3) << static_cast<int>(outImage[0][i][j]) << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
    write_values_to_file("dataOut/data3CPUGauss.txt", outImage,NUMBER_IMAGES);
    free(inBuf);
    free(outBuf);
    return 0;
}
