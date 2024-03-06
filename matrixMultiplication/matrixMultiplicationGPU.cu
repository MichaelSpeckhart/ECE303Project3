#include "timer.h"
#include <iostream>
#include <iomanip>
#include <random>
#include <vector>
#include <climits>

typedef std::vector<std::vector<double>> matrix;
const size_t MATRIX_SIZE = 100;

using namespace std;

/// error with dimensions in a matrix
vector<vector<double>> d_err{{INT_MAX, INT_MAX, INT_MIN},
                             {INT_MIN, INT_MIN, INT_MAX}};

/// @brief Creates a random matrix of given dimensions filled with values in
/// specified range.
/// @param d1 Number of rows.
/// @param d2 Number of columns.
/// @param min Lower bound of values.
/// @param max Upper bound of values.
/// @return The generated matrix.
vector<vector<double>> generate_random_matrix(const int d1, const int d2,
                                              const double min,
                                              const double max)
{
    // must be a 1x1 matrix at the minimum
    if (d1 < 1 || d2 < 1)
        return d_err;

    vector<vector<double>> matrix;
    vector<double> line;

    random_device rd;
    default_random_engine eng(rd());
    uniform_real_distribution<double> distr(min, max);

    for (int i = 0; i < d1; ++i)
    {
        for (int j = 0; j < d2; ++j)
            line.push_back(distr(eng));
        matrix.push_back(line);
        line.clear();
    }

    return matrix;
}

// /// @brief Multiplies two matrices together.
// /// @param m1 First matrix.
// /// @param m2 Second matrix.
// /// @return Product matrix.
// vector<vector<double>> mult_matrix(const vector<vector<double>> m1,
//                                    const vector<vector<double>> m2)
// {
//     const int r1 = static_cast<int>(m1.size()),
//               c1 = static_cast<int>(m1[0].size()),
//               r2 = static_cast<int>(m2.size()),
//               c2 = static_cast<int>(m2[0].size());
//     //   columns of first matrix must equal rows of second
//     if (c1 != r2)
//         return d_err;

//     vector<vector<double>> m3(r1);

//     for (auto i = 0; i < r1; i++)
//         m3[i] = vector<double>(c2, 0);

//     for (int i = 0; i < r1; ++i)
//         for (int j = 0; j < c2; ++j)
//             for (int k = 0; k < r2; ++k)
//                 m3[i][j] += m1[i][k] * m2[k][j];
//     return m3;
// }

__global__ void mult_matrix_kernel(const vector<vector<double>> m1,const vector<vector<double>> m2,vector<vector<double>> m3)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    const int r1 = static_cast<int>(m1.size());
    const int c1 = static_cast<int>(m1[0].size());
    const int r2 = static_cast<int>(m2.size());
    const int c2 = static_cast<int>(m2[0].size());

    for (int i = idx; i < r1; i += stride)
        for (int j = 0; j < c2; ++j)
            for (int k = 0; k < r2; ++k)
                m3[i][j] += m1[i][k] * m2[k][j];
}

int main(){
    vector<vector<double>> m1 = generate_random_matrix(MATRIX_SIZE,MATRIX_SIZE,-1000,1000);
    vector<vector<double>> m2 = generate_random_matrix(MATRIX_SIZE,MATRIX_SIZE,-1000,1000);
    int threads_per_block = 512;
    int deviceId;
    cudaGetDevice(&deviceId);
  
    int numberOfSMs;
    cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);
    
    int number_of_blocks = 32 * numberOfSMs;
    StartTimer();
    vector<vector<double>> m3(m1.size());
    for (auto i = 0; i < m1.size(); i++)
        m3[i] = vector<double>(m2[0].size(), 0);
    mult_matrix_kernel<<<number_of_blocks, threads_per_block>>>(m1,m2,m3);
    std::cout << GetTimer() << " ms for multiplication of " <<MATRIX_SIZE <<" size"<< std::endl;
}
