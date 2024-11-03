#pragma once

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <cstdint>
#include <limits>

#include "../constants.cuh"
#include "../helper.h"
#include "../helper_kernels/utils.cuh"
#include "../kernels.cuh"
#include "../helper_kernels/histogram.cuh"
#include <cuda_runtime.h>
#include <omp.h>
#include <limits>



template<class P>
void HistoCPU(
    typename P::ElementType* input,
    typename P::UintType* histogram,
    uint32_t input_size,
    uint32_t bit_pos
) {
    // Clear histogram array
    memset(histogram, 0, P::H * P::GRID_SIZE * sizeof(typename P::UintType));
    
    // Calculate mask to extract relevant bits
    typename P::ElementType mask = (P::H - 1) << bit_pos;
    
    // Process each input element
    for (uint32_t i = 0; i < input_size; i++) {
        // Extract the relevant bits and shift them to the right
        uint32_t bin = (input[i] & mask) >> bit_pos;
        
        // Calculate block_id based on the same distribution as GPU
        uint32_t elements_per_block = P::BLOCK_SIZE * P::Q;
        uint32_t block_id = i / elements_per_block;
        uint32_t hist_index = bin + (block_id * P::H);
        
        // Increment the counter for this bin
        histogram[hist_index]++;
    }
}



template<class P>
void testHistoKer(uint32_t input_size) {

    printf("Testing Histogram kernel\n");

    typename P::ElementType* h_in;
    typename P::ElementType* d_in;

    typename P::UintType* d_hist;
    typename P::UintType* h_hist;

    typename P::UintType* cpu_hist;


    uint32_t hist_len = P::H * P::GRID_SIZE;
    cpu_hist = (typename P::UintType*)calloc(hist_len, sizeof(typename P::UintType));
    
    allocateAndInitialize<typename P::ElementType, P::MAXNUMERIC_ElementType>(&h_in, &d_in, input_size, true);
    allocateAndInitialize<typename P::UintType, P::MAXNUMERIC_UintType>(&h_hist, &d_hist, hist_len, false);

    uint32_t bit_pos = 0;
    HistoCPU<P>(
        h_in, 
        cpu_hist, 
        input_size, 
        bit_pos
    );

    printf("CPU Histogram done\n");


    Histo<P, typename P::UintType><<<P::GRID_SIZE, P::BLOCK_SIZE>>>(
        d_in, 
        d_hist, 
        input_size, 
        bit_pos
    );

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Histogram kernel failed: %s\n", cudaGetErrorString(err));
        return;
    } 

    cudaMemcpy(h_hist, d_hist, sizeof(typename P::UintType) * hist_len, cudaMemcpyDeviceToHost);

    assert(validate<typename P::UintType>(h_hist, cpu_hist, hist_len));
    printf("histogram results validated\n");

    free(cpu_hist);
    free(h_in);
    free(h_hist);
    cudaFree(d_in);
    cudaFree(d_hist);
}
