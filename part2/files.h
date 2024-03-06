#include <stdio.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <cstdint>
#include <string>

void read_values_from_file(const char *file, unsigned char data[][28][28], size_t num_values_to_read) {
    std::ifstream values(file, std::ios::binary);
    for (size_t i = 0; i < num_values_to_read; ++i) {
        values.read(reinterpret_cast<char*>(data[i]), sizeof(unsigned char) * 28 * 28);
    }
    values.close();
}

void write_values_to_file(const char * file, unsigned char data[][28][28], size_t size) {
    std::ofstream values(file, std::ios::binary);
    for (size_t i = 0; i < size; ++i) {
        values.write(reinterpret_cast<char*>(data[i]), sizeof(unsigned char) * 28 * 28);
    }
    values.close();
}
