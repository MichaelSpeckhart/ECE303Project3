#include "timer.h"
#include <iostream>
#include <iomanip>
#include <random>

using namespace std;

const size_t MATRIX_SIZE = 500;

int main(){
    //random num generator
    random_device rd;
    default_random_engine eng(rd());
    uniform_real_distribution<double> distr(-10000, 10000);
    //malloc m1
    double* m1Buf = (double*)malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(double));
    double (*m1)[MATRIX_SIZE] = (double (*)[MATRIX_SIZE])m1Buf;
    //fill m1
    for(size_t i = 0 ; i < MATRIX_SIZE;i++){
        for(size_t j = 0 ; j <MATRIX_SIZE; j++){
            m1[i][j] = distr(eng);
        }
    }

    //malloc m2
    double* m2Buf = (double*)malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(double));
    double (*m2)[MATRIX_SIZE] = (double (*)[MATRIX_SIZE])m2Buf;
    //fill m1
    for(size_t i = 0 ; i < MATRIX_SIZE;i++){
        for(size_t j = 0 ; j <MATRIX_SIZE; j++){
            m2[i][j] = distr(eng);
        }
    }

    //malloc m3
    double* m3Buf = (double*)malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(double));
    double (*m3)[MATRIX_SIZE] = (double (*)[MATRIX_SIZE])m3Buf;
    
    StartTimer();
    for (size_t i = 0; i < MATRIX_SIZE; ++i)
        for (size_t j = 0; j < MATRIX_SIZE; ++j){
            for (size_t k = 0; k < MATRIX_SIZE; ++k)
                m3[i][j] += m1[i][k] * m2[k][j];
        }

    std::cout << GetTimer() << " ms for multiplication of " <<MATRIX_SIZE <<" size"<< std::endl;

    free(m1Buf);
    free(m2Buf);
    free(m3Buf);
}
