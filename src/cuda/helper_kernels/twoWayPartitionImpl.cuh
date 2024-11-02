#pragma once

#include <algorithm>
#include <iostream>
#include "../constants.cuh"
#include "../helper.h"




template<class T>
inline void TwoWayPartitionCpu(
    uint32_t bit_offs,
    uint32_t N,
    T* arr_inp,
    T* arr_out,
    bool debug = false
) {
    // Copy input to output first
    std::copy(arr_inp, arr_inp + N, arr_out);
    
    if (debug) {
        // Print original array
        std::cout << "Original array (showing bit " << bit_offs << "):\n";
        for(uint32_t i = 0; i < N; i++) {
        bool bit = isBitUnset(bit_offs, arr_inp[i]);
            std::cout << arr_inp[i] << "(" << (bit ? "0" : "1") << ") ";
        }
        std::cout << "\n\n";
    }
    
    printf("calling std::partition\n");
    // Use std::partition to do the partitioning
    std::partition(arr_out, arr_out + N, 
        [bit_offs](const T& val) {
            return isBitUnset(bit_offs, val);
        }
    );

    printf("std::partition done\n");

    if (debug) {
        // Print partitioned array
        std::cout << "Partitioned array:\n";
        for(uint32_t i = 0; i < N; i++) {
        bool bit = isBitUnset(bit_offs, arr_out[i]);
        std::cout << arr_out[i] << "(" << (bit ? "0" : "1") << ") ";
        }
        std::cout << "\n";
    }
    
    // Verify partition
    uint32_t partition_point = 0;
    while (partition_point < N && isBitUnset(bit_offs, arr_out[partition_point])) {
        partition_point++;
    }
    if (debug) {
        std::cout << "\nPartition point: " << partition_point << "\n";
        std::cout << "First part (0s): " << partition_point << " elements\n";
        std::cout << "Second part (1s): " << N - partition_point << " elements\n";
    }

    // Add debugging for each bit position
    for (uint32_t bit = 0; bit < 8; bit++) {  // Assuming 8 bits like in GPU version
        if (debug) {
            std::cout << "Examining bit position " << (bit_offs + bit) << ":\n";
            for(uint32_t i = 0; i < N; i++) {
                bool is_unset = isBitUnset(bit_offs + bit, arr_out[i]);
                std::cout << arr_out[i] << "(" << (is_unset ? "0" : "1") << ") ";
            }
            std::cout << "\n\n";
        }
    }
}