/*
Luke Hale and Michael Speckhart
Project #3
Part #2.7
3/8/2024
*/
#include "timer.h"
#include <iostream>
#include <iomanip>
#include <random>

using namespace std;

const size_t MATRIX_SIZE = 100;

__global__ void mult_matrix_kernel(double* m1Buf,double* m2Buf,double* m3Buf)
{
    double (*m1)[MATRIX_SIZE] = (double (*)[MATRIX_SIZE])m1Buf;
    double (*m2)[MATRIX_SIZE] = (double (*)[MATRIX_SIZE])m2Buf;
    double (*m3)[MATRIX_SIZE] = (double (*)[MATRIX_SIZE])m3Buf;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < MATRIX_SIZE; i += stride)
        for (size_t j = 0; j < MATRIX_SIZE; ++j){
            for (size_t k = 0; k < MATRIX_SIZE; ++k)
                m3[i][j] += m1[i][k] * m2[k][j];
        }
}

int main(){

    size_t bytes = MATRIX_SIZE * MATRIX_SIZE * sizeof(double);
    //random num generator
    random_device rd;
    default_random_engine eng(rd());
    uniform_real_distribution<double> distr(-10000, 10000);
    //malloc m1
    double* m1Buf = (double*)malloc(bytes);
    double (*m1)[MATRIX_SIZE] = (double (*)[MATRIX_SIZE])m1Buf;
    //fill m1
    for(size_t i = 0 ; i < MATRIX_SIZE;i++){
        for(size_t j = 0 ; j <MATRIX_SIZE; j++){
            m1[i][j] = distr(eng);
        }
    }

    //malloc m2
    double* m2Buf = (double*)malloc(bytes);
    double (*m2)[MATRIX_SIZE] = (double (*)[MATRIX_SIZE])m2Buf;
    //fill m1
    for(size_t i = 0 ; i < MATRIX_SIZE;i++){
        for(size_t j = 0 ; j <MATRIX_SIZE; j++){
            m2[i][j] = distr(eng);
        }
    }

    //malloc m3
    double* m3BufDevice;
    cudaMalloc(&m3BufDevice, bytes);

    double* m1BufDevice;
    cudaMalloc(&m1BufDevice, bytes);
    cudaMemcpy(m1BufDevice, m1Buf, bytes, cudaMemcpyHostToDevice);

    double* m2BufDevice;
    cudaMalloc(&m2BufDevice, bytes);
    cudaMemcpy(m2BufDevice, m2Buf, bytes, cudaMemcpyHostToDevice);

    //double (*m3)[MATRIX_SIZE] = (double (*)[MATRIX_SIZE])m3Buf;
    int threads_per_block = 512;
    int deviceId;
    cudaGetDevice(&deviceId);
  
    int numberOfSMs;
    cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);
    
    int number_of_blocks = 32 * numberOfSMs;

    StartTimer();
    mult_matrix_kernel<<<number_of_blocks, threads_per_block>>>(m1BufDevice, m2BufDevice, m3BufDevice);
    std::cout << GetTimer() << " ms for multiplication of " <<MATRIX_SIZE <<" size"<< std::endl;\

    double* m3Buf = (double*)malloc(bytes);
    cudaMemcpy(m3Buf, m3BufDevice, bytes, cudaMemcpyDeviceToHost);
    //double (*m3)[MATRIX_SIZE] = (double (*)[MATRIX_SIZE])m3Buf;

    free(m1Buf);
    free(m2Buf);
    free(m3Buf);
    cudaFree(m1BufDevice);
    cudaFree(m2BufDevice);
    cudaFree(m3BufDevice);
}
