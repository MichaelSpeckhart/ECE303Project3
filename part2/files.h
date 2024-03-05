#include <stdio.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <cstdint>
#include <string>

// void read_values_from_file(const char * file, float * data, size_t size) {
//     std::ifstream values(file, std::ios::binary);
//     values.read(reinterpret_cast<char*>(data), size);
//     values.close();
// }

void read_values_from_file(const char *file, unsigned char data[][28*28], size_t num_values_to_read) {
    std::ifstream fid(file, std::ios::binary);
    if (!fid.is_open()) {
        std::cout << "Error opening file." << std::endl;
        return;
    }
    
    for (size_t i = 0; i < num_values_to_read; ++i) {
        fid.read(reinterpret_cast<char*>(data[i]), sizeof(unsigned char) * 28 * 28);
    }
    
    fid.close();
}

void write_values_to_file(const char * file, float * data, size_t size) {
    std::ofstream values(file, std::ios::binary);
    values.write(reinterpret_cast<char*>(data), size);
    values.close();
}
